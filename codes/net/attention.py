# Attention calculations
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from codes.utils.util import merge_first_two_dims_batch, unmerge_first_two_dims_batch

class Attn(nn.Module):
    def __init__(self, method, hidden_size, concat_size=None):
        super(Attn, self).__init__()
        self.method = method # 'concat' by default
        self.hidden_size = hidden_size
        if not concat_size:
            concat_size = self.hidden_size * 4
        self.attn = nn.Linear(concat_size, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs, mask):
        '''
        :param hidden:
            previous hidden state of the decoder, in shape (B, T, H)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (B, T, T, H)
        :param mask:
            one-hot mask before computing softmax, shape (B,T)
        :return
            attention energies in shape (B,T, T)
        '''
        max_len = encoder_outputs.size(1)
        this_batch_size = encoder_outputs.size(1)
        H = hidden.unsqueeze(2).repeat(1, 1, max_len, 1)
        # H = hidden.repeat(max_len,1,1).transpose(0,1)
        # encoder_outputs = encoder_outputs.transpose(0,1) # [B*T*H]
        attn_energies = self.score(H,encoder_outputs) # compute attention score
        attn_energies = attn_energies * mask
        return F.softmax(attn_energies, dim=2) # normalize with softmax B x 1 x T

    def score(self, hidden, encoder_outputs):
        shape = hidden.shape
        batched_hidden = merge_first_two_dims_batch(hidden)
        batched_encoder_outputs = merge_first_two_dims_batch(encoder_outputs)
        energy = F.tanh(self.attn(torch.cat([batched_hidden, batched_encoder_outputs], 2))) # [B*T*2H]->[B*T*H]
        energy = energy.transpose(2,1) # [B*H*T]
        v = self.v.repeat(batched_encoder_outputs.data.shape[0],1).unsqueeze(1) #[B*1*H]
        energy = torch.bmm(v,energy) # [B*1*T]
        return unmerge_first_two_dims_batch(energy.squeeze(1), first_dim=shape[0]) #[B*T]

class SimpleSelfAttention(nn.Module):
    def __init__(self, temperature=1.0, dropout=0.1):
        super(SimpleSelfAttention, self).__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_in, mask=None, squeeze=True):
        q = torch.max(x_in, 1)[0].unsqueeze(1) # B x 1 x D
        k = x_in.transpose(1,2) # B x D x s
        attn = torch.bmm(q, k) # B x 1 x s
        attn = attn / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        x_out = torch.bmm(attn, x_in) # B x 1 x D
        if squeeze:
            x_out = x_out.squeeze(1)
        return x_out

if __name__ == '__main__':
    attn = SimpleSelfAttention(100)
    x = torch.randn(16, 30, 100)
    assert attn(x).size() == (16,100)