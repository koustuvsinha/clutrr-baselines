# Relation networks for CLUTRR
# Idea: relation networks would span either among words or among sentences
# RN(0) = f(\sum g_{\theta} (o_i, o_j))
# where, o_i could be words or o_i could be sentences
# let us first denote it as words

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from codes.net.base_net import Net
import pdb


class RelationNetworkEncoder(Net):
    """
    Relation Networks
    Paper: https://arxiv.org/pdf/1706.01427.pdf
    """
    def __init__(self, model_config, shared_embeddings=None):
        super().__init__(model_config)
        self.init_embeddings()

        self.reader = nn.LSTM(
            model_config.embedding.dim,
            model_config.encoder.hidden_dim,
            model_config.encoder.nlayers,
            bidirectional=model_config.encoder.bidirectional,
            batch_first=True,
            dropout=model_config.encoder.dropout
        )

        bidirectional_mult = 1
        if model_config.encoder.bidirectional:
            bidirectional_mult = 2
        self.g_theta = self.get_mlp_h(model_config.encoder.hidden_dim * bidirectional_mult * 4, model_config.encoder.rn.g_theta_dim,
                                      num_layers=4)
        self.f_theta_1 = self.get_mlp_h(model_config.encoder.rn.g_theta_dim, model_config.encoder.rn.f_theta.dim_1, num_layers=1)
        self.f_theta_2 = self.get_mlp_h(model_config.encoder.rn.f_theta.dim_1, model_config.encoder.rn.f_theta.dim_2, num_layers=1)

    def calculate_query(self, batch):
        """

        :param encoder_outputs: B x seq_len x dim
        :param ent_mask: B x max_abs x num_ents x seq_len
        :param outp_ents: B x num_ents (usually 2)
        :return:
        """
        encoder_outputs, query_mask = batch.encoder_outputs, batch.query_mask
        # expand
        num_ents = query_mask.size(2)
        seq_len = query_mask.size(1)
        # encoder_outputs # B x seq_len x dim
        query_mask = query_mask.transpose(1,2) # B x num_ents x seq_len

        query_rep = torch.bmm(query_mask.float(), encoder_outputs) # B x num_ents x dim
        query_rep = query_rep.transpose(1,2) # B x dim x num_ents
        hidden_size = self.model_config.encoder.hidden_dim
        ents = query_rep.size(-1)
        query_reps = []
        for i in range(ents):
            query_reps.append(query_rep.transpose(1,2)[:,i,:].unsqueeze(1))
        query_rep = torch.cat(query_reps, -1)
        return query_rep

    def forward(self, batch):
        """
        Reference: https://github.com/kimhc6028/relational-networks/blob/master/model.py
        :param batch:
        :return:
        """
        data = batch.inp
        data_lengths = batch.inp_lengths
        # embed the data
        data = self.embedding(data)
        # read the data through a bidirectional encoder
        data_pack = pack_padded_sequence(data, data_lengths, batch_first=True)
        outp, hidden_rep = self.reader(data_pack)
        outp, _ = pad_packed_sequence(outp, batch_first=True)
        outp = outp.contiguous() # B x length x dim

        max_len = outp.size(1)
        batch_size = outp.size(0)
        # how to add the question entity
        # lets add the question (for which we have two entities) directly concatenated
        # in the object pairs
        batch.encoder_outputs = outp
        query = self.calculate_query(batch) # B x 1 x (2*dim)
        query = query.repeat(1, max_len, 1) # B x len x (2*dim)
        query = torch.unsqueeze(query, 2) # B x len x 1 x (2*dim)

        # combine all pairs of words
        x_i = torch.unsqueeze(outp, 1) # B x 1 x length x dim
        x_i = x_i.repeat(1, max_len, 1, 1) # B x len x len x dim
        x_j = torch.unsqueeze(outp, 2) # B x len x 1 x dim
        x_j = torch.cat([x_j, query],3) # B x len x 1 x (dim * 3)
        x_j = x_j.repeat(1,1,max_len, 1) # B x len x len x (dim * 3)

        # concat all together
        x_full = torch.cat([x_i, x_j], 3)

        # reshape for passing through the network
        x_ = x_full.view(batch_size * max_len * max_len, -1) # (Bxlenxlen) x (dim*3)
        x_ = self.g_theta(x_)
        # reshape and sum
        x_g = x_.view(batch_size, -1, self.model_config.encoder.rn.g_theta_dim) # B x (len x len) x g_dim
        x_g = x_g.sum(1) # B x g_dim

        # apply f
        x_f = self.f_theta_2(self.f_theta_1(x_g)) # B x f_dim

        return x_f, hidden_rep

class RelationNetworkDecoder(Net):
    """ Simple MLP decoder"""
    def __init__(self, model_config, shared_embeddings=None):
        super().__init__(model_config)
        self.fout = self.get_mlp(model_config.encoder.rn.f_theta.dim_2,
                                 model_config.target_size, num_layers=2)

    def init_hidden(self, encoder_outputs, batch_size):
        return None

    def calculate_query(self, batch):
        return None

    def forward(self, batch, step_batch):
        out = self.fout(batch.encoder_outputs)
        return out, None, None

