# Sequence to Sequence net for abstractive summarization
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from codes.net.base_net import Net
from codes.utils.util import check_id_emb
from codes.net.attention import Attn
from torch.nn import functional as F
import numpy as np
import pdb

class SimpleEncoder(Net):
    """
    Simple LSTM Encoder
    """
    def __init__(self, model_config, shared_embeddings=None, use_embedding=True):
        super().__init__(model_config)

        if not shared_embeddings:
            self.init_embeddings()
        else:
            self.embedding = shared_embeddings

        self.lstm = nn.LSTM(
            model_config.embedding.dim,
            model_config.embedding.dim,
            model_config.encoder.nlayers,
            bidirectional=model_config.encoder.bidirectional,
            batch_first=True,
            dropout=model_config.encoder.dropout
        )

        self.use_embedding = use_embedding

    def forward(self, batch):
        data = batch.inp
        inp_len = np.array(batch.inp_lengths)
        if self.use_embedding:
            data = self.embedding(data)
        # sort
        inp_len_sorted, idx_sort = np.sort(inp_len)[::-1], np.argsort(-inp_len)
        inp_len_sorted = inp_len_sorted.copy()
        idx_unsort = np.argsort(idx_sort)

        idx_sort = torch.from_numpy(idx_sort).to(data.device)
        data = data.index_select(0, idx_sort)
        inp_len_sorted_nonzero_idx = np.nonzero(inp_len_sorted)[0]
        inp_len_sorted_nonzero_idx = torch.from_numpy(inp_len_sorted_nonzero_idx).to(data.device)
        inp_len_sorted = torch.from_numpy(inp_len_sorted).to(data.device)
        non_zero_data = data.index_select(0, inp_len_sorted_nonzero_idx)
        data_pack = pack_padded_sequence(non_zero_data, inp_len_sorted[inp_len_sorted_nonzero_idx], batch_first=True)
        outp, hidden_rep = self.lstm(data_pack)
        outp, _ = pad_packed_sequence(outp, batch_first=True)
        outp = outp.contiguous()
        outp_l = torch.zeros((data.size(0), data.size(1), outp.size(2))).to(outp.device)
        outp_l[inp_len_sorted_nonzero_idx] = outp
        # unsort
        idx_unsort = torch.from_numpy(idx_unsort).to(outp_l.device)
        outp_l = outp_l.index_select(0, idx_unsort)

        return outp_l.contiguous(), hidden_rep

class SimpleDecoder(Net):
    """
    Simple LSTM Decoder
    """
    def __init__(self, model_config, shared_embeddings=None, attn_module=None):
        super().__init__(model_config)

        if not shared_embeddings:
            self.init_embeddings()
        else:
            self.embedding = shared_embeddings

        self.pool_type = model_config.decoder.pool_type
        # set simple MLP classifier
        base_enc_dim = model_config.embedding.dim
        if model_config.encoder.bidirectional:
            base_enc_dim *=2
        query_rep = base_enc_dim * model_config.decoder.query_ents

        if self.pool_type == 'concat':
            base_enc_dim = base_enc_dim*2
        input_dim = query_rep + base_enc_dim
        output_dim = model_config.target_size
        self.decoder2vocab = self.get_mlp(input_dim, output_dim)

        self.attn_module = None
        if self.pool_type == 'attn':
            self.attn_module = LSTMAttn(base_enc_dim, input_dim)

    def init_hidden(self, encoder_states, batch_size):
        # initial hidden state of the decoder will be an average of encoder states
        avg_state = torch.mean(encoder_states, 1, keepdim=True)  # B x 1 x dim
        avg_state = avg_state.transpose(0, 1).expand(self.model_config.decoder.nlayers, -1, -1)  # nlayers x B x dim
        encoder_hidden = []
        encoder_hidden.append(avg_state)
        encoder_hidden.append(torch.zeros_like(avg_state).to(avg_state.device))
        return encoder_hidden

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
        query_mask = query_mask.transpose(1, 2) # B x num_ents x seq_len

        query_rep = torch.bmm(query_mask.float(), encoder_outputs) # B x num_ents x dim
        query_rep = query_rep.transpose(1, 2) # B x dim x num_ents
        hidden_size = self.model_config.embedding.dim
        ents = query_rep.size(-1)
        query_reps = []
        for i in range(ents):
            query_reps.append(query_rep.transpose(1, 2)[:, i, :].unsqueeze(1))
        query_rep = torch.cat(query_reps, -1)
        return query_rep

    def forward(self, batch, step_batch):
        hidden_rep = step_batch.hidden_rep
        query_rep = step_batch.query_rep
        encoder_outputs = batch.encoder_outputs
        _, seq_len, dim = encoder_outputs.size()
        if self.model_config.loss_type == 'classify':
            if self.pool_type == 'attn':
                emb = self.attn_module(query_rep, encoder_outputs)
            elif self.pool_type == 'max':
                # encoder_outputs[encoder_outputs == 0] = -1e9
                emb = torch.max(encoder_outputs, 1)[0]
            elif self.pool_type == 'mean':
                sent_len = torch.FloatTensor(batch.inp_lengths.copy()).unsqueeze(1).to(encoder_outputs.device)
                emb = torch.sum(encoder_outputs, 1)
                # BUG FIX: fails if batchsize is 1
                if emb.dim() > 2:
                    emb = emb.squeeze(0)
                emb = emb / sent_len.expand_as(emb)
            elif self.pool_type == 'concat':
                sent_len = torch.FloatTensor(batch.inp_lengths.copy()).unsqueeze(1).to(encoder_outputs.device)
                emb_mean = torch.sum(encoder_outputs, 1).squeeze(0)
                emb_mean = emb_mean / sent_len.expand_as(emb_mean)
                encoder_outputs[encoder_outputs == 0] = -1e9
                emb_max = torch.max(encoder_outputs, 1)[0]
                emb = torch.cat([emb_max, emb_mean], -1)
            else:
                emb = encoder_outputs[:, -1, :]
            if emb.dim() == 3:
                emb = emb.squeeze(0)
                assert emb.dim() == 2
            mlp_inp = torch.cat([query_rep.squeeze(1), emb], -1)
        else:
            decoder_inp = step_batch.decoder_inp
            check_id_emb(decoder_inp, self.model_config.vocab_size)
            decoder_inp = self.embedding(decoder_inp)
            lstm_inp = torch.cat([decoder_inp, query_rep], -1)
            mlp_inp, hidden_rep = self.lstm(lstm_inp, hidden_rep)

        outp = self.decoder2vocab(mlp_inp)
        return outp, None, hidden_rep

    def __handle_state__(self, tensor):
        """
        Cut the first dimension in half and concat it to last dimension
        :param tensor:
        :return:
        """
        size_mid = int(tensor.size(0) / 2)
        return torch.cat([tensor[:size_mid], tensor[size_mid:]], -1)

class LSTMAttn(nn.Module):
    '''
    Used by SimpleDecoder
    '''
    def __init__(self, hidden_size, concat_size=None):
        super(LSTMAttn, self).__init__()

        self.hidden_size = hidden_size

        self.concat_size = concat_size
        if not concat_size:
            self.concat_size = self.hidden_size*3
        self.attn1 = nn.Linear(self.concat_size, self.hidden_size)
        print(self.concat_size, self.hidden_size)
        self.attn2 = nn.Linear(self.hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        '''
        :param hidden:          B x 1 x H
        :param encoder_outputs: B x T x H
        :return:
        '''
        encoder_outputs = encoder_outputs.transpose(0, 1)   # T x B x H
        # print('---', encoder_outputs.shape)
        hidden = hidden.transpose(0, 1).repeat(encoder_outputs.shape[0], 1, 1) # T x B x H
        # print('---', hidden.shape)
        enc_hid = torch.cat([encoder_outputs, hidden], 2)  # T x B x 2H
        # print('---', enc_hid.shape)
        e1= F.tanh(self.attn1(enc_hid))
        e = self.attn2(e1)     # T x B x 1
        a = F.softmax(e, dim=0)    # T x B x 1

        return (a * encoder_outputs).sum(dim=0)    # B x H

