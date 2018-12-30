# graph encoder

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from codes.net.base_net import Net
from codes.models.gnn.mpnn_model import MPNNModel
from codes.utils.util import one_hot_embedding, get_sinusoid_encoding_table, PositionwiseFeedForward
from codes.utils.encoder_utils import EncoderUtils
from codes.net.attention import SimpleSelfAttention
import pdb

class GraphEncoder(Net):
    """
    Graph Encoder

    1. Read the document with an LSTM
    2. Get intermediate encoder hidden states
    3. Pool the hidden states for the entities
    4. Pass the node embeddings as B x node x node_dim to MPNN
    5. Return the output of readout function to the decoder
    """
    def __init__(self, model_config, mpnn_model=None, shared_embeddings=None):
        super().__init__(model_config)

        if not shared_embeddings:
            self.init_embeddings()
        else:
            self.embedding = shared_embeddings

        self.freeze_embeddings()
        self.model_config = model_config
        # sinusoidal position embedding, from Vasvani et al
        n_position = self.model_config.max_sent_length + 1
        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, model_config.embedding.dim, padding_idx=0),
            freeze=True)

        if self.model_config.graph.pos_rep == 'random':
            self.position_embedding = nn.Embedding(
                num_embeddings=self.model_config.max_nodes,
                embedding_dim=self.model_config.graph.pos_dim
            )

            # freeze the position embeddings
            self.position_embedding.weight.requires_grad = False

        edge_dim = model_config.graph.edge_dim - model_config.graph.pos_dim * 2 - model_config.graph.feature_dim * 2

        if model_config.graph.edge_embedding == 'word':
            inp_dim = model_config.embedding.dim
            output_dim = edge_dim
            self.edge_fn = self.get_mlp(inp_dim, output_dim)
            #self.edge_fn = PositionwiseFeedForward(inp_dim, output_dim)
        else:
            hidden_size = edge_dim // 2
            self.lstm = nn.LSTM(
                model_config.embedding.dim,
                hidden_size,
                model_config.encoder.nlayers,
                bidirectional=model_config.encoder.bidirectional,
                batch_first=True,
                dropout=model_config.encoder.dropout
            )

        self.h_pos = None

        #self.graph_feature = self.get_mlp((model_config.max_nodes * model_config.max_nodes),
            # model_config.graph.feature_dim)

        self.graph_feature = nn.Parameter(torch.zeros(1, model_config.graph.feature_dim))

        self.mpnn = MPNNModel(model_config)
        self.self_attn = SimpleSelfAttention()


    def extract_edge_embedding(self, batch):
        """
        Given a batch of stories, extract edge embeddings
        :param batch:
        :return:
        """
        # raise error if not in sentence mode
        if not batch.config.dataset.sentence_mode:
            raise NotImplementedError("cannot extract edge embedding without sentence mode")

        enc_util = EncoderUtils()
        data = batch.inp
        sent_lengths = batch.sent_lengths
        batch_size, num_sentences, num_words = data.size()
        # sort according to the sent lengths
        data = enc_util.pack(data, sent_lengths)
        data_state = self.embedding(data) # (B x sent) x words x emb_dim
        if self.model_config.encoder.use_position_emb:
            pos_enc_util = EncoderUtils()
            data_pos = pos_enc_util.pack(batch.inp_row_pos, sent_lengths)
            data_pos_emb = self.position_enc(data_pos)
            data_state += data_pos_emb

        hidden_rep = None
        if batch.config.model.graph.edge_embedding == 'lstm':
            # since we padded empty sentences, we need to remove them before pad packing
            emb_data = torch.index_select(data_state, 0, enc_util.data_indices)
            data_lengths_nonzero = [l for l in enc_util.data_lengths_sorted if l > 0]
            data_pack = pack_padded_sequence(emb_data, data_lengths_nonzero, batch_first=True)
            outp, hidden_rep = self.lstm(data_pack)
            outp, _ = pad_packed_sequence(outp, batch_first=True)  # (batch x sents) x words x dim
            data_state = torch.zeros(batch_size * num_sentences, num_words, outp.size(-1), device=outp.device)
            data_state[enc_util.data_mask] = outp
        else:
            # word emb -> mlp -> edge dim
            b,w,d = data_state.size()
            data_state = data_state.view(b*w, d)
            data_state = self.edge_fn(data_state)
            data_state = data_state.view(b,w,-1)
            # replacing the above with Conv1d which does the same thing
            # on sequences
            # data_state = self.edge_fn(data_state)

        # unsort
        data_state = enc_util.unpack(data_state)

        # extract sentences for each entity pair
        max_entity_id = self.max_entity_id
        edge_dim = data_state.size(-1)
        inp_ents = batch.inp_ents # batch x sents x num_ents
        sent_pointer = batch.sentence_pointer # batch x max_id x max_id x num_sents
        assert sent_pointer.size(-1) == num_sentences

        true_edges = torch.bmm(
            sent_pointer.view(batch_size, -1, sent_pointer.size(-1)).float(),
            data_state.view(batch_size, sent_pointer.size(-1), -1))

        true_edges = true_edges.view(batch_size, max_entity_id, max_entity_id, -1, edge_dim)

        # convert sentences into fixed vectors
        if self.model_config.encoder.pooling == 'attention':
            true_edges = self.self_attn(true_edges.view((batch_size * max_entity_id * max_entity_id), -1, edge_dim))
            true_edges = true_edges.view(batch_size, max_entity_id, max_entity_id, edge_dim)
        elif self.model_config.encoder.pooling == 'maxpool':
            true_edges = torch.max(true_edges, 3)[0]
        else:
            raise NotImplementedError("Pooling method {} not implemented".format(self.model_config.encoder.pooling))

        return true_edges, hidden_rep

    def forward(self, batch):
        max_nodes = batch.sentence_pointer.size(1)
        h_in = torch.zeros(batch.batch_size, max_nodes, self.model_config.graph.node_dim).to(batch.inp.device)
        node_indices = torch.arange(0, max_nodes).unsqueeze(0).repeat(batch.batch_size, 1).long()
        # position embedding for directionality. one hot
        # Instead of a one-hot embedding, use a normal random embedding to denote position
        if self.model_config.graph.pos_rep == 'random':
            h_pos = self.position_embedding(node_indices.to(h_in.device))
        else:
            h_pos = one_hot_embedding(node_indices, max_nodes).to(h_in.device) # batch x max_nodes x max_nodes
        # first view
        true_edges, hidden_rep = self.extract_edge_embedding(batch)
        # get the queries
        ids = batch.outp_ents
        max_ents = batch.adj_mat.size(1)
        num_ents = batch.outp_ents.size(-1)
        num_abs = batch.outp.size(1)
        # correct ids for word2id which begins with 1
        ids = ids - 1
        # prune based on query
        start_ids = ids.squeeze(1)[:,0].cpu().numpy()
        end_ids = ids.squeeze(1)[:,-1].cpu().numpy()
        batch.prune_paths(start_ids, end_ids)
        g = batch.inp_graphs
        h_feature = self.graph_feature
        h_feature = h_feature.unsqueeze(1).expand(batch.batch_size, h_pos.size(1), h_feature.size(1))
        h_pos = torch.cat([h_pos, h_feature], dim=2)

        mask = batch.inp_ent_mask.unsqueeze(-1).float()
        resp = self.mpnn.forward((g, true_edges, h_in, h_pos, mask))
        # with the current graph state, predict a new g and do message passing over it.
        # then, pass that representation to the decoder
        self.h_pos = h_pos
        return resp, hidden_rep


class PrototypeGraphEncoder(Net):
    """
    Prototype Graph Encoder
    1. Read the text
    2. for each entity, get a softmax over d dimension as f(text, entity) = softmax(d)
    3. Use this prototype matrix as a starting point to run message passing
    """

    def __init__(self, model_config, shared_embeddings=None):
        super().__init__(model_config)
        inp_dim = model_config.embedding.dim * 2
        outp_dim = model_config.graph.prototypes
        self.prototype_extractor = self.get_mlp(inp_dim, outp_dim,
                                                num_layers=1)
        self.node_dictionary = nn.Parameter(torch.zeros(outp_dim, model_config.graph.node_dim))
        self.softmax = nn.Softmax()

        if not shared_embeddings:
            self.init_embeddings()
        else:
            self.embedding = shared_embeddings

        self.freeze_embeddings()
        self.model_config = model_config

        if self.model_config.graph.pos_rep == 'random':
            self.position_embedding = nn.Embedding(
                num_embeddings=self.model_config.max_nodes,
                embedding_dim=self.model_config.graph.pos_dim
            )

            # freeze the position embeddings
            self.position_embedding.weight.requires_grad = False

        if model_config.graph.edge_embedding == 'word':
            inp_dim = model_config.embedding.dim
            output_dim = model_config.graph.edge_dim - model_config.graph.pos_dim * 2 - model_config.graph.feature_dim * 2
            self.edge_fn = self.get_mlp(inp_dim, output_dim)
        else:
            hidden_size = (model_config.graph.edge_dim - model_config.graph.pos_dim * 2 - model_config.graph.feature_dim * 2) // 2
            self.lstm = nn.LSTM(
                model_config.embedding.dim,
                hidden_size,
                model_config.encoder.nlayers,
                bidirectional=model_config.encoder.bidirectional,
                batch_first=True,
                dropout=model_config.encoder.dropout
            )

        self.h_pos = None

        #self.graph_feature = self.get_mlp((model_config.max_nodes * model_config.max_nodes),
            # model_config.graph.feature_dim)

        self.graph_feature = nn.Parameter(torch.zeros(1, model_config.graph.feature_dim))

        self.mpnn = MPNNModel(model_config)

    def extract_node_embedding(self, batch):
        """
        Given the text sentences, indentify the sentences for each node
        Then learn a function f(sents, node),
            where node = node embedding of the particular node
            if in lstm mode, node is the maxpool over hidden states of the individual node
        :param batch:
        :return:
        """
        # raise error if not in sentence mode
        if not batch.config.dataset.sentence_mode:
            raise NotImplementedError("cannot extract node embedding without sentence mode")

        enc_util = EncoderUtils()
        data = batch.inp
        sent_lengths = batch.sent_lengths
        batch_size, num_sentences, num_words = data.size()
        # sort according to the sent lengths
        data = enc_util.pack(data, sent_lengths)
        # get the embeddings
        data_state = self.embedding(data)
        data_state = enc_util.unpack(data_state)
        sent_pointer = batch.sentence_pointer  # batch x max_id x max_id x num_sents

        # extract sentences for each entity pair
        max_entity_id = self.max_entity_id
        edge_dim = data_state.size(-1)
        assert sent_pointer.size(-1) == num_sentences
        true_edges = torch.bmm(
            sent_pointer.view(batch_size, -1, sent_pointer.size(-1)).float(),
            data_state.view(batch_size, sent_pointer.size(-1), -1))

        true_edges = true_edges.view(batch_size, max_entity_id,
                                     max_entity_id, -1, edge_dim)
        # now for each entity, take all the sentences and calculate
        # a softmax for each
        node_feature = true_edges.view(batch_size, max_entity_id, -1, edge_dim)
        node_feature = torch.max(node_feature, 2)[0]  # B x max_entity x dim
        query_node_ids = torch.arange(max_entity_id).unsqueeze(0).repeat(batch_size, 1).long() # B x max_entity
        query_node_ids = query_node_ids.to(data.device)
        query_node_ids = query_node_ids + 1 # adjusting for the padding
        query_node_emb = self.embedding(query_node_ids) # B x max_entity x dim
        mlp_inp = torch.cat([node_feature.view(batch_size * max_entity_id, -1),
                  query_node_emb.view(batch_size * max_entity_id, -1)], dim=1) # (B x max_entity) x dim
        proto = self.softmax(self.prototype_extractor(mlp_inp)) # B x p
        proto = proto.unsqueeze(1) # B x 1 x p
        p, d = self.node_dictionary.size()
        dictn = self.node_dictionary.unsqueeze(0).expand((batch_size * max_entity_id), p, d) # B x p x d
        node_init = torch.bmm(proto, dictn).squeeze(1) # B x dim
        node_init = node_init.view(batch_size, max_entity_id, -1)
        return node_init


    def forward(self, batch):
        max_nodes = batch.sentence_pointer.size(1)
        h_in = self.extract_node_embedding(batch)
        node_indices = torch.arange(0, max_nodes).unsqueeze(0).repeat(batch.batch_size, 1).long()
        # position embedding for directionality. one hot
        # Instead of a one-hot embedding, use a normal random embedding to denote position
        if self.model_config.graph.pos_rep == 'random':
            h_pos = self.position_embedding(node_indices.to(h_in.device))
        else:
            h_pos = one_hot_embedding(node_indices, max_nodes).to(h_in.device)  # batch x max_nodes x max_nodes
        # first view
        hidden_rep = None
        true_edges = torch.zeros(batch.batch_size, max_nodes, max_nodes,
                                 self.model_config.graph.edge_dim)
        # get the queries
        ids = batch.outp_ents
        max_ents = batch.adj_mat.size(1)
        num_ents = batch.outp_ents.size(-1)
        num_abs = batch.outp.size(1)
        # correct ids for word2id which begins with 1
        ids = ids - 1
        # prune based on query
        start_ids = ids.squeeze(1)[:, 0].cpu().numpy()
        end_ids = ids.squeeze(1)[:, -1].cpu().numpy()
        batch.prune_paths(start_ids, end_ids)
        g = batch.inp_graphs
        h_feature = self.graph_feature
        h_feature = h_feature.unsqueeze(1).expand(batch.batch_size, h_pos.size(1), h_feature.size(1))
        h_pos = torch.cat([h_pos, h_feature], dim=2)

        mask = batch.inp_ent_mask.unsqueeze(-1).float()
        resp = self.mpnn.forward((g, true_edges, h_in, h_pos, mask))
        self.h_pos = h_pos
        return resp, hidden_rep

