import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from codes.net.batch import Batch
from codes.net.base_net import Net
from codes.baselines.lstm.basic import SimpleEncoder

class MACNetworkEncoder(Net):
    """
    Input Unit and MAC Cell Unit of MAC Network.
    """

    def __init__(self, model_config, shared_embeddings=None):
        super(MACNetworkEncoder, self).__init__(model_config)

        if not shared_embeddings:
            self.init_embeddings()
        else:
            self.embedding = shared_embeddings

        bidirectional_mult = 1
        self.reader = SimpleEncoder(model_config, shared_embeddings=self.embedding)
        if model_config.encoder.bidirectional:
            bidirectional_mult = 2


    def calculate_query(self, batch):
        """
        :param batch
                encoder_outputs: B x seq_len x dim
                ent_mask: B x max_abs x num_ents x seq_len
                outp_ents: B x num_ents (usually 2)
        :return:
        """
        encoder_outputs, query_mask = batch.encoder_outputs, batch.query_mask
        # expand
        num_ents = query_mask.size(2)
        seq_len = query_mask.size(1)
        # encoder_outputs # B x seq_len x dim
        query_mask = query_mask.transpose(1, 2)  # B x num_ents x seq_len

        query_rep = torch.bmm(query_mask.float(), encoder_outputs)  # B x num_ents x dim
        query_rep = query_rep.transpose(1, 2)  # B x dim x num_ents
        hidden_size = self.model_config.encoder.hidden_dim
        ents = query_rep.size(-1)
        query_reps = []
        for i in range(ents):
            query_reps.append(query_rep.transpose(1, 2)[:, i, :].unsqueeze(1))
        query_rep = torch.cat(query_reps, -1)
        return query_rep


class MACNetworkDecoder(Net):
    """
    Output Unit of MAC Network, which is an MLP.
    """
    def __init__(self, model_config, share_embeddings=None):
        super(MACNetworkDecoder, self).__init(model_config)

        base_enc_dim = model_config.embedding.dim
        if model_config.encoder.bidirectional:
            base_enc_dim *= 2
        query_rep = base_enc_dim * model_config.decoder.query_ents

        self.output_unit = self.get_mlp(query_rep+base_enc_dim,
                                        model_config.target_size, num_layers=2)

    def forward(self, batch, step_batch):

        query_rep = batch.query_rep
        enc_outputs = batch.encoder_outputs

        out = self.output_unit(torch.cat([query_rep.squeeze(1), enc_outputs]))
        return out, None, None

    def init_hidden(self, encoder_outputs, batch_size):
        return None


    def calculate_query(self, batch):
        return None


class MACNetworkCell(Net):

    # TODO
    def __init__(self, model_config, hidden_size):
        super(MACNetworkCell, self).__init__(model_config)

        self.model_config = model_config

        base_enc_dim = model_config.embedding.dim
        if model_config.encoder.bidirectional:
            base_enc_dim *= 2
        query_rep_dim = base_enc_dim * model_config.decoder.query_ents

        # linear transformate query_rep into a position-aware vector
        # B, 2*dim -> B, dim
        self.question_transform = self.get_mlp(query_rep_dim,
                                               hidden_size, num_layers=2)

        # ---Control Unit
        self.contControl = nn.Linear(hidden_size*2, hidden_size)
        self.controlAttn = nn.Linear(hidden_size, 1)

        # Read Unit

        # Write Unit
        self.writeNewMemory = self.get_mlp(hidden_size*2, hidden_size, num_layers=1)

    # TODO
    def init_state(self, batch_size, hidden_size):
        # self.c = nn.Parameter(torch.rand(hidden_size))
        # self.m = nn.Parameter(torch.rand(hidden_size))

        c = torch.rand(batch_size, hidden_size)
        m = torch.rand(batch_size, hidden_size)

        return c, m


    def control(self, query_rep, query_rep_all, ctrl, contCtrl=None):
        """
        :param query_rep:     [B, dim], concatenation of two query entities after a time-step specific linear transformation
        :param query_rep_all: [B, entity_num, dim] (typically entity_num = 2)
        :param ctrl:          [B, dim], previous control state
        :param contCtrl:      [B, dim], previous continuous control state (before casting to softmax, optional)
        :return ctrl:         [B, dim], new control state
        :return contCtrl:     [B, dim], new continuous control state
        """
        n_batch = query_rep.size(0)
        n_entity = query_rep_all.size(1)

        # 1: compute "continuous" control state given previous control and question.
        newContCtrl = self.contControl(torch.cat([ctrl, query_rep], -1))  # B x dim

        # 2:   compute attention over query entities and sum them up.
        # 2.1: computer interactions between continuous control state and query entities.
        interactions = torch.unsqueeze(newContCtrl, 1) * query_rep_all     # B x entity_num x dim

        # optionally concatenate query entities with interactions.
        if self.model_config.controlConcatWords:
            interactions = torch.cat([interactions, query_rep_all], -1)   # B x entity_num x 2*dim

        # optionally projections
        if self.model_config.controlProj:
            interactions = self.controlProj(interactions)                 # B x entity_num x dim

        # compute attn distribution
        logits = self.controlAttn(interactions)                # B x entity_num x 1
        attn_weight = F.softmax(logits, dim=1)                  # B x entity_num x 1
        newCtrl = (attn_weight * query_rep_all).sum(dim=1).squeeze(1)  # B x dim

        return newCtrl, newContCtrl


    # TODO
    def read(self):
        pass


    # TODO: add optional self-attention to consider previous intermediate result
    # TODO: add optional memory gate to dynamically adjust reasoning length
    def write(self, memory, info, ctrl, contCtrl=None):
        """
        :param memory:    [B, dim], previous memory state
        :param info:      [B, dim], retrieved info from read unit
        :param ctrl:      [B, dim], current control state
        :param contCtrl:  [B, dim], current continuous state
        :return:
            new memory [B, dim]
        """

        newMemory = torch.cat([info, memory], dim=1)    # B x 2*dim
        newMemory = self.writeNewMemory(newMemory)      # B x dim

        return newMemory


    # TODO
    def forward(self, batch):
        pass