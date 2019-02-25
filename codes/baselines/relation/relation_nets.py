# Relation networks for CLUTRR
# Idea: relation networks would span either among words or among sentences
# RN(0) = f(\sum g_{\theta} (o_i, o_j))
# where, o_i could be words or o_i could be sentences
# let us first denote it as words

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from codes.baselines.lstm.basic import SimpleEncoder
from codes.net.batch import Batch
from codes.net.base_net import Net
from addict import Dict
import numpy as np
import pdb


class TPREncoder(Net):
    def __init__(self, model_config, shared_embeddings=None):
        super().__init__(model_config)

        if not shared_embeddings:
            self.init_embeddings()
        else:
            self.embedding = shared_embeddings

        self.position_emb = nn.Embedding(num_embeddings=model_config.max_word_length,
                                     embedding_dim=model_config.embedding.dim)
        torch.nn.init.constant_(self.position_emb.weight, 1 / model_config.max_sent_length)

    def forward(self, batch:Batch):
        """
        Perform hadamard product between position embeddings
        https://arxiv.org/pdf/1811.12143.pdf
        :param batch:
        :return:
        """
        positions = torch.arange(batch.inp.shape[1]).unsqueeze(0).repeat(batch.inp.shape[0],1).to(batch.inp.device).long() # B x sent_len
        positions = self.position_emb(positions) # B x sent_len x dim
        data = self.embedding(batch.inp) # B x sent x dim
        return data * positions, None # B x sent_len x dim

class RNSentReader(Net):
    """
    Read sentences and return a sentence object for each sentences
    """
    def __init__(self, model_config, shared_embeddings=None):
        super().__init__(model_config)
        self.embedding = shared_embeddings

        if model_config.encoder.rn.reader == 'lstm':
            self.reader = SimpleEncoder(model_config, shared_embeddings=self.embedding)
        elif model_config.encoder.rn.reader == 'tpr':
            # Reader module as https://arxiv.org/pdf/1811.12143.pdf
            # basically add a position embedding to the input sentence
            self.reader = TPREncoder(model_config, shared_embeddings=self.embedding)

        # can be either max or mean
        self.pooling = model_config.encoder.pooling
        if self.pooling not in ['max', 'mean']:
            raise NotImplementedError("RNSentReader {} pooling not implemented".format(self.pooling))

    def forward(self, batch):
        inp = batch.s_inp # B x s x w
        B, sent_len, word_len = inp.size()
        inp = inp.view(-1, word_len) # (B x sent_len) x w
        inp_len = [s for sl in batch.sent_lengths for s in sl] # flatten
        reader_batch = Batch(inp=inp, inp_lengths=inp_len)
        outp,_ =  self.reader(reader_batch) # (B x s) x w x dim
        question_batch = Batch(inp=batch.inp, inp_lengths=batch.inp_lengths)
        q_outp,_ = self.reader(question_batch) # B x len x dim
        if self.pooling == 'mean':
            inp_len = np.array(inp_len)
            inp_len[inp_len == 0] = 1
            sent_len_a = torch.from_numpy(np.array(inp_len)).unsqueeze(1).to(outp.device).float()
            emb = torch.sum(outp, 1).squeeze(0)
            emb = emb / sent_len_a.expand_as(emb) # (B x s) x dim
        else:
            outp[outp == 0] = -1e9
            emb = torch.max(outp, 1)[0]
        outp = emb.view(B, sent_len, -1)  # B x s x dim
        return outp, q_outp




class RelationNetworkEncoder(Net):
    """
    Relation Networks
    Paper: https://arxiv.org/pdf/1706.01427.pdf
    """
    def __init__(self, model_config, shared_embeddings=None):
        super().__init__(model_config)
        self.init_embeddings()
        bidirectional_mult = 1
        if model_config.encoder.rn.reader == 'lstm':
            if model_config.encoder.bidirectional:
                bidirectional_mult = 2

        self.reader = RNSentReader(model_config, shared_embeddings=self.embedding)

        self.g_theta = self.get_mlp_h(model_config.embedding.dim * bidirectional_mult * 4, model_config.encoder.rn.g_theta_dim,
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
        hidden_size = self.model_config.embedding.dim
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
        # read the data through a bidirectional encoder
        #pdb.set_trace()
        outp,q_outp = self.reader(batch) # B x length x dim

        max_len = outp.size(1)
        batch_size = outp.size(0)
        # how to add the question entity
        # lets add the question (for which we have two entities) directly concatenated
        # in the object pairs
        min_batch = Dict()
        min_batch.encoder_outputs = q_outp
        min_batch.query_mask = batch.query_mask
        #batch.encoder_outputs = q_outp
        query = self.calculate_query(min_batch) # B x 1 x (2*dim)
        query = query.repeat(1, max_len, 1) # B x len x (2*dim)
        query = torch.unsqueeze(query, 2) # B x len x 1 x (2*dim)

        # problem: if max_len is high, then all pair combination is difficult
        # solution: or just run it per batch
        B = outp.size(0)

        """
        o_i = torch.arange(max_len).unsqueeze(0).unsqueeze(2).repeat(B, 1, 1) # B x length x 1
        o_i = torch.unsqueeze(o_i, 1) # B x 1 x length x 1
        o_i = o_i.repeat(1, max_len, 1, 1) # B x length x length x 1
        o_j = torch.arange(max_len).unsqueeze(0).unsqueeze(2).repeat(B, 1, 1) # B x length x 1
        o_j = torch.unsqueeze(o_j, 2) # B x length x 1 x 1
        o_j = o_j.repeat(1, 1, max_len, 1) # B x length x length x 1
        o_ij = torch.cat([o_i, o_j], 3)
        o_ij = o_ij.view(B * max_len * max_len, -1) # (Bxlenxlen) x (dim*3)

        # break this down into minibatches
        k = 5
        

        x_g = []
        for bi in range(B):
            x_ib = torch.unsqueeze(outp[bi], 0) # 1 x len x dim
            x_ib = x_ib.repeat(max_len, 1, 1)  # len x len x dim
            x_jb = torch.unsqueeze(outp[bi], 1)  # len x 1 x dim
            q_b = query[bi] # len x 1 x (dim * 2)
            x_jb = torch.cat([x_jb, q_b], 2)  # len x 1 x (dim * 3)
            x_jb = x_jb.repeat(1, max_len, 1)  # len x len x (dim * 3)

            # concat all together
            x_full_b = torch.cat([x_ib, x_jb], 2) # len x len x (dim * 4)

            # reshape for passing through the network
            x_b = x_full_b.view(max_len * max_len, -1)  # (lenxlen) x (dim*3)
            x_b = self.g_theta(x_b)
            # reshape and sum
            x_gb = x_b.view(-1, self.model_config.encoder.rn.g_theta_dim)  # (len x len) x g_dim
            x_gb = x_gb.sum(0).unsqueeze(0)  # 1 x g_dim
            x_g.append(x_gb)

        x_g = torch.cat(x_g, dim=0)


        """
        # combine all pairs of sentences
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


        return x_f, None

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

