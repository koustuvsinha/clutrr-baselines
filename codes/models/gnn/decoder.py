# graph decoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from codes.net.base_net import Net
from codes.utils.util import check_id_emb
from codes.models.gnn.components.readout_function import AttentiveReadoutFunction, AverageReadout


class GraphLSTMDecoder(Net):
    """
    Graph Readout + Simple LSTM Decoder
    """
    def __init__(self, model_config, mpnn_model=None,
                 shared_embeddings=None, attn_module=None):
        super().__init__(model_config)

        if not shared_embeddings:
            self.init_embeddings()
        else:
            self.embedding = shared_embeddings

        self.model_config = model_config

        #self.decoder2vocab = nn.Linear(
        #    model_config.decoder.hidden_dim * self.num_directions,
        #    model_config.vocab_size
        #)

        if model_config.loss_type == 'classify':
            query_rep = model_config.graph.edge_dim + model_config.graph.pos_dim + model_config.graph.feature_dim
            query_rep *= 2

            if model_config.graph.readout_function.read_mode == 'only_query':
                input_dim = query_rep
            else:
                input_dim = query_rep + model_config.graph.node_dim

            output_dim = model_config.target_size
            self.decoder2vocab = self.get_mlp(input_dim, output_dim, dropout=model_config.graph.dropout)
        else:
            inp_dim = model_config.embedding.dim + model_config.graph.edge_dim

            hidden_dim = model_config.decoder.hidden_dim

            self.lstm = nn.LSTM(
                inp_dim,
                hidden_dim,
                model_config.decoder.nlayers,
                bidirectional=model_config.decoder.bidirectional,
                batch_first=True,
                dropout=model_config.decoder.dropout
            )
            self.decoder2vocab = self.get_mlp(
                hidden_dim,
                model_config.vocab_size, num_layers=1
            )
            self.copynet = self.get_mlp(
                model_config.graph.node_dim,
                model_config.decoder.hidden_dim, num_layers=1
            )
            self.switchnet = self.get_mlp(
                inp_dim + hidden_dim,
                1
            )
            self.sigmoid = nn.Sigmoid()

        if model_config.graph.readout_function.read_mode == 'attention':
            self.attn_readout = AttentiveReadoutFunction(model_config,
                    concat_size=(model_config.graph.node_dim + model_config.graph.pos_dim + model_config.graph.feature_dim)*2 + model_config.graph.node_dim)
        elif model_config.graph.readout_function.read_mode == 'average':
            self.avg_readout = AverageReadout(model_config)

    def init_hidden(self, encoder_states, batch_size):
        # initial hidden state of the decoder will be an average of encoder states
        avg_state = torch.mean(encoder_states, 1, keepdim=True) # B x 1 x dim
        avg_state = avg_state.transpose(0, 1).expand(self.model_config.decoder.nlayers, -1, -1).contiguous() # nlayers x B x dim
        encoder_hidden = []
        encoder_hidden.append(avg_state)
        encoder_hidden.append(torch.zeros_like(avg_state).to(avg_state.device))
        return encoder_hidden

    def calculate_query(self, batch):
        ### Calculate the query with respect to the entities. Concatenate the query one hot vector
        ### and the hidden messages, and pass it to the edge network
        encoder_inp, encoder_outp, query = batch.inp, batch.encoder_outputs, batch.query
        ids = query
        max_ents = batch.adj_mat.size(1)
        num_ents = query.size(-1)
        # correct ids for word2id which begins with 1
        ids = ids - 1
        ids = ids.view(-1, num_ents).unsqueeze(1) # ids : B x 1 x query_ents
        check_id_emb(ids, max_ents)
        id_mask = torch.zeros(query.size(0), max_ents, query.size(-1)).to(ids.device) # B x max_ents x query_ents
        id_mask.scatter_(1, ids, 1)
        id_mask = id_mask.long()
        encoder_model = batch.encoder_model
        h_pos = encoder_model.h_pos
        query_rep = encoder_model.mpnn.readout_function(
            encoder_outp, h_pos, id_mask, encoder_model.mpnn.message_function.edge_network)
        return query_rep

    def forward(self, batch, step_batch):
        decoder_inp = step_batch.decoder_inp
        hidden_rep = step_batch.hidden_rep
        query_rep = step_batch.query_rep
        encoder_outputs = batch.encoder_outputs
        mask = batch.inp_ent_mask.unsqueeze(-1) # Bx max_nodes x 1

        if self.model_config.loss_type == 'classify':
            if self.model_config.graph.readout_function.read_mode == 'average':
                graph_state = self.avg_readout(encoder_outputs, mask)
            elif self.model_config.graph.readout_function.read_mode == 'attention':
                # not all entities are useful (the work,school, etc entities), thus perform an attention
                graph_state = self.attn_readout(encoder_outputs, query_rep.transpose(0,1), mask.float())
            if self.model_config.graph.readout_function.read_mode == 'only_query':
                mlp_inp = query_rep.squeeze(1)
            else:
                mlp_inp = torch.cat([query_rep.squeeze(1), graph_state.squeeze(1)], -1)

            outp = self.decoder2vocab(mlp_inp)
        else:
            check_id_emb(decoder_inp, self.model_config.vocab_size)
            decoder_inp = self.embedding(decoder_inp)
            batch_size, max_nodes, enc_dim = batch.encoder_outputs.size()
            state = hidden_rep[0][-1]
            last_hidden_state = torch.cat([state, query_rep.squeeze(1)], dim=1).unsqueeze(0)
            #graph_state = encoder_outputs.contiguous().view(-1, encoder_outputs.size(-1))
            #outp_g = self.copynet(graph_state)
            lstm_inp = torch.cat([decoder_inp, query_rep], -1)
            mlp_inp, hidden_rep = self.lstm(lstm_inp, hidden_rep)
            #outp_vocab = self.decoder2vocab(mlp_inp)
            # arrange predictions w.r.t the vocabulary
            #outp_gt = torch.zeros(outp_vocab.size()).to(outp_vocab.device)
            #outp_gt[:,:,1:outp_g.size(-1)+1] = outp_g
            #switch_prob = self.sigmoid(self.switchnet(torch.cat([last_hidden_state.transpose(0,1).squeeze(1),
            #                                                     decoder_inp.squeeze(1)], dim=-1))).unsqueeze(2)
            #outp = switch_prob * outp_vocab + (1 - switch_prob) * outp_gt

            score_g = self.decoder2vocab(state)
            score_c = F.tanh(self.copynet(batch.encoder_outputs.contiguous().view(-1,encoder_outputs.size(-1))))
            score_c = score_c.view(encoder_outputs.size()) # B x max_nodes x size
            score_c = torch.bmm(score_c, state.unsqueeze(2)).squeeze() # B x max_nodes
            mask = (~mask.byte()).float()
            mask *= -1000
            score_c += mask.squeeze(-1)
            score_c = F.tanh(score_c)
            score = torch.cat([score_g, score_c], 1) # B x (vocab + max_nodes)
            probs = F.softmax(score)
            prob_g = probs[:,:self.model_config.vocab_size] # Batch x vocab
            prob_c = probs[:, self.model_config.vocab_size:] # batch x max_nodes
            prob_c_to_g = torch.zeros(batch_size, self.model_config.vocab_size).to(state.device)
            prob_c_to_g[:,1:prob_c.size(1)+1] = prob_c
            switch_prob = self.sigmoid(self.switchnet(torch.cat([last_hidden_state.transpose(0, 1).squeeze(1),
                                                                 decoder_inp.squeeze(1)], dim=-1)))
            outp = switch_prob * prob_g + (1-switch_prob) * prob_c_to_g
            outp = torch.log(outp).unsqueeze(1)


        return outp, None, hidden_rep

    def __handle_state__(self, tensor):
        """
        Cut the first dimension in half and concat it to last dimension
        :param tensor:
        :return:
        """
        size_mid = int(tensor.size(0) / 2)
        return torch.cat([tensor[:size_mid], tensor[size_mid:]], -1)

    def __expand_abs(self, tensor, max_abs):
        """
        Expand the tensor to match max abstract lines
        :param tensor:
        :param max_abs:
        :return:
        """
        _, seq_len, dim = tensor.size()
        return tensor.unsqueeze(1).expand(
            -1, max_abs, -1, -1).contiguous().view(-1, seq_len, dim)
