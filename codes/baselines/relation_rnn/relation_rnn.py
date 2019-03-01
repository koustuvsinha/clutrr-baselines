# The model is from `Relational recurrent neural networks (https://arxiv.org/abs/1806.01822)`
# The implementation is based on `https://github.com/L0SG/relational-rnn-pytorch`

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from codes.net.batch import Batch
from codes.net.base_net import Net


class RelationRNNEncoder(Net):
    """
    Args:
      mem_slots:    total number of memory slots (1st dim of memory matrix M).
      head_size:    size of an attention head.
      input_size:   size of input per step. i.e. the dimension of each input vector (embedding dim)
      num_heads:    number of attention heads to use. Defaults to 1.
      num_blocks:   number of times to compute attention per time step. Defaults 1.
      forget_bias:  bias to use for the forget gate, assuming we are using
                    some form of gating. Defaults to 1.
      input_bias:   bias to use for the input gate, assuming we are using
                    some form of gating. Defaults to 0.
      gate_style:   whether to use per-element gating ('unit'),
                    per-memory slot gating ('memory'), or no gating at all (None).
                    Defaults to `unit`.
      attention_mlp_layers: Number of layers to use in the post-attention
                    MLP. Defaults to 2.
      key_size:     size of vector to use for key & query vectors in the attention
                    computation. Defaults to None, in which case we use `head_size`.
      name:         Name of the module.
    """

    def __init__(self, model_config, shared_embeddings=None):
        super(RelationRNNEncoder, self).__init__(model_config)
        if not shared_embeddings:
            self.init_embeddings()
        else:
            self.embedding = shared_embeddings

        self.mem_slots = model_config.RMC.mem_slots
        self.head_size = model_config.RMC.head_size
        self.num_heads = model_config.RMC.num_heads
        self.num_blocks = model_config.RMC.num_blocks
        self.attention_mlp_layers = model_config.RMC.attention_mlp_layers
        self.mem_size = self.head_size * self.num_heads

        # a new fixed params needed for pytorch port of RMC
        # +1 is the concatenated input per time step: do self-attention with the concatenated memory & input
        # so if the mem_slots = 1, this value is 2
        self.mem_slots_plus_input = self.mem_slots + 1

        self.forget_bias = model_config.RMC.forget_bias
        self.input_bias = model_config.RMC.input_bias
        self.gate_style = model_config.RMC.gate_style
        assert self.gate_style in ['unit', 'memory', None]

        self.key_size = model_config.RMC.key_size  # if model_config.RMC.key_size else self.head_size
        ########## parameters for multihead attention ##########
        # value_size is same as head_size
        self.value_size = self.head_size
        # total size for query-key-value
        self.qkv_size = 2 * self.key_size + self.value_size
        self.total_qkv_size = self.qkv_size * self.num_heads  # denoted as F

        # each head has qkv_sized linear projector
        # just using one big param is more efficient, rather than this line
        # self.qkv_projector = [nn.Parameter(torch.randn((self.qkv_size, self.qkv_size))) for _ in range(self.num_heads)]
        self.qkv_projector = nn.Linear(self.mem_size, self.total_qkv_size)
        self.qkv_layernorm = nn.LayerNorm([self.mem_slots_plus_input, self.total_qkv_size])

        # used for attend_over_memory function
        self.attention_mlp = nn.ModuleList([nn.Linear(self.mem_size, self.mem_size)] * self.attention_mlp_layers)
        self.attended_memory_layernorm = nn.LayerNorm([self.mem_slots_plus_input, self.mem_size])
        self.attended_memory_layernorm2 = nn.LayerNorm([self.mem_slots_plus_input, self.mem_size])

        ########## parameters for initial embedded input projection ##########
        self.input_size = model_config.embedding.dim
        self.input_projector = nn.Linear(self.input_size, self.mem_size)

        ########## parameters for gating ##########
        self.num_gates = 2 * self.calculate_gate_size()
        self.input_gate_projector = nn.Linear(self.mem_size, self.num_gates)
        self.memory_gate_projector = nn.Linear(self.mem_size, self.num_gates)
        # trainable scalar gate bias tensors
        self.forget_bias = nn.Parameter(torch.tensor(self.forget_bias, dtype=torch.float32))
        self.input_bias = nn.Parameter(torch.tensor(self.input_bias, dtype=torch.float32))

        ########## number of outputs returned #####
        self.return_all_outputs = model_config.RMC.return_all_outputs

        ######### dropout #####
        self.dropout = nn.Dropout(model_config.RMC.dropout)


    def initial_state(self, batch, batch_size):
        """
        :param batch:
        :param batch_size:
        :return: (batch_size, self.mem_slots, self.mem_size)
        """
        init_state = torch.stack([torch.eye(self.mem_slots) for _ in range(batch_size)])

        # pad the matrix with zeros
        if self.mem_size > self.mem_slots:
            difference = self.mem_size - self.mem_slots
            pad = torch.zeros((batch_size, self.mem_slots, difference))
            init_state = torch.cat([init_state, pad], -1)

        # truncation. take the first 'self.mem_size' components
        elif self.mem_size < self.mem_slots:
            init_state = init_state[:, :, :self.mem_size]

        return init_state.to(batch.inp.device)


    def repackage_hidden(self, h):
        """Wraps hidden states in new Tensors, to detach them from their history."""
        # needed for truncated BPTT, called at every batch forward pass
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)


    def multihead_attention(self, memory):
        """
        Perform multi-head attention from 'Attention is All You Need'.
        Implementation of the attention mechanism from
        https://arxiv.org/abs/1706.03762.
        Args:
          memory: Memory tensor to perform attention on.
        Returns:
          new_memory: New memory tensor.
        """

        # First, a simple linear projection is used to construct queries
        # B x mem_slots x mem_size -> B x mem_slots x total qkv size (F)
        qkv = self.qkv_projector(memory)
        # apply layernorm for every dim except the batch dim
        qkv = self.qkv_layernorm(qkv)

        # mem_slots needs to be dynamically computed since mem_slots got concatenated with inputs
        # example: self.mem_slots=10 and seq_length is 3, and then mem_slots is 10 + 1 = 11 for each 3 step forward pass
        # this is the same as self.mem_slots_plus_input, but defined to keep the sonnet implementation code style
        mem_slots = memory.shape[1]  # denoted as N

        # split the qkv to multiple heads H
        # [B, N, F] => [B, N, H, F/H]
        qkv_reshape = qkv.view(qkv.shape[0], mem_slots, self.num_heads, self.qkv_size)

        # [B, N, H, F/H] => [B, H, N, F/H]
        qkv_transpose = qkv_reshape.permute(0, 2, 1, 3)

        # [B, H, N, key_size], [B, H, N, key_size], [B, H, N, value_size]
        q, k, v = torch.split(qkv_transpose, [self.key_size, self.key_size, self.value_size], -1)

        # scale q with d_k, the dimensionality of the key vectors
        q *= (self.key_size ** -0.5)

        # make it [B, H, N, N]
        dot_product = torch.matmul(q, k.permute(0, 1, 3, 2))
        weights = F.softmax(dot_product, dim=-1)

        # output is [B, H, N, V]
        output = torch.matmul(weights, v)

        # [B, H, N, V] => [B, N, H, V] => [B, N, H*V]
        output_transpose = output.permute(0, 2, 1, 3).contiguous()
        new_memory = output_transpose.view((output_transpose.shape[0], output_transpose.shape[1], -1))

        return new_memory


    def attend_over_memory(self, memory):
        """
        Perform multiheaded attention over `memory`.
            Args:
              memory: Current relational memory.
            Returns:
              The attended-over memory.
        """
        for _ in range(self.num_blocks):
            attended_memory = self.multihead_attention(memory)

            # Add a skip connection to the multiheaded attention's input.
            memory = self.attended_memory_layernorm(memory + attended_memory)

            # add a skip connection to the attention_mlp's input.
            attention_mlp = memory
            for i, l in enumerate(self.attention_mlp):
                attention_mlp = self.attention_mlp[i](attention_mlp)
                attention_mlp = F.relu(attention_mlp)
            memory = self.attended_memory_layernorm2(memory + attention_mlp)

        return memory


    def calculate_gate_size(self):
        """
        Calculate the gate size from the gate_style.
        Returns:
          The per sample, per head parameter size of each gate.
        """
        if self.gate_style == 'unit':
            return self.mem_size
        elif self.gate_style == 'memory':
            return 1
        else:  # self.gate_style == None
            return 0


    def create_gates(self, inputs, memory):
        """
        Create input and forget gates for this step using `inputs` and `memory`.
        Args:
          inputs: Tensor input.
          memory: The current state of memory.
        Returns:
          input_gate: A LSTM-like insert gate.
          forget_gate: A LSTM-like forget gate.
        """
        # We'll create the input and forget gates at once. Hence, calculate double
        # the gate size.

        # equation 8: since there is no output gate, h is just a tanh'ed m
        memory = torch.tanh(memory)

        # TODO: check this input flattening is correct
        # sonnet uses this, but i think it assumes time step of 1 for all cases
        # if inputs is (B, T, features) where T > 1, this gets incorrect
        # inputs = inputs.view(inputs.shape[0], -1)

        # fixed implementation
        if len(inputs.shape) == 3:
            if inputs.shape[1] > 1:
                raise ValueError(
                    "input seq length is larger than 1. create_gate function is meant to be called for each step, with input seq length of 1")
            inputs = inputs.view(inputs.shape[0], -1)
            # matmul for equation 4 and 5
            # there is no output gate, so equation 6 is not implemented
            gate_inputs = self.input_gate_projector(inputs)
            gate_inputs = gate_inputs.unsqueeze(dim=1)
            gate_memory = self.memory_gate_projector(memory)
        else:
            raise ValueError("input shape of create_gate function is 2, expects 3")

        # this completes the equation 4 and 5
        gates = gate_memory + gate_inputs
        gates = torch.split(gates, split_size_or_sections=int(gates.shape[2] / 2), dim=2)
        input_gate, forget_gate = gates
        assert input_gate.shape[2] == forget_gate.shape[2]

        # to be used for equation 7
        input_gate = torch.sigmoid(input_gate + self.input_bias)
        forget_gate = torch.sigmoid(forget_gate + self.forget_bias)

        return input_gate, forget_gate


    def forward_step(self, inputs, memory, treat_input_as_matrix=False):
        """
        Forward step of the relational memory core.
        Args:
          inputs: Tensor input. [B x 1 x embed_dim]
          memory: Memory output from the previous time step.
          treat_input_as_matrix: Optional, whether to treat `input` as a sequence
            of matrices. Default to False, in which case the input is flattened
            into a vector.
        Returns:
          output: This time step's output.
          next_memory: The next version of memory to use.
        """

        if treat_input_as_matrix:
            # keep (Batch, Seq, ...) dim (0, 1), flatten starting from dim 2
            inputs = inputs.view(inputs.shape[0], inputs.shape[1], -1)
            # apply linear layer for dim 2
            inputs_reshape = self.input_projector(inputs)
        else:
            # keep (Batch, ...) dim (0), flatten starting from dim 1
            inputs = inputs.view(inputs.shape[0], -1)
            # apply linear layer for dim 1
            inputs = self.input_projector(inputs)
            # unsqueeze the time step to dim 1
            inputs_reshape = inputs.unsqueeze(dim=1)        # B x 1 x mem_size

        memory_plus_input = torch.cat([memory, inputs_reshape], dim=1)  # B x (mem_slots+1) x mem_size
        next_memory = self.attend_over_memory(memory_plus_input)

        # cut out the concatenated input vectors from the original memory slots
        n = inputs_reshape.shape[1]
        next_memory = next_memory[:, :-n, :]

        if self.gate_style == 'unit' or self.gate_style == 'memory':
            # these gates are sigmoid-applied ones for equation 7
            input_gate, forget_gate = self.create_gates(inputs_reshape, memory)
            # equation 7 calculation
            next_memory = input_gate * torch.tanh(next_memory)
            next_memory += forget_gate * memory

        # outout is a flatten memory, B x mem_slots*mem_size
        output = next_memory.view(next_memory.shape[0], -1)

        return output, next_memory


    def forward(self, batch):
        import time
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        # memory = self.repackage_hidden(memory)

        # for loop implementation of (entire) recurrent forward pass of the model
        # inputs is batch first [batch, seq], and output logit per step is [batch, vocab]
        # so the concatenated logits are [seq * batch, vocab]

        # targets are flattened [seq, batch] => [seq * batch], so the dimension is correct
        inputs = batch.inp
        inputs_lengths = batch.inp_lengths
        inputs = self.embedding(inputs)
        inputs = self.dropout(inputs)

        batch_size = inputs.shape[0]
        time_steps = inputs.shape[1]
        memory = self.initial_state(batch, batch_size)
        logits = []
        # shape[1] is seq_lenth T
        for idx_step in range(time_steps):
            # input: B x embed_dim
            # memory: B x mem_slots x mem_size
            logit, memory = self.forward_step(inputs[:, idx_step], memory)
            if self.return_all_outputs:
                logits.append(logit.unsqueeze(1))
        if self.return_all_outputs:
            logits = torch.cat(logits, dim=1)
            return logits, memory.view(memory.shape[0], -1)
        else:
            return logit, memory.view(memory.shape[0], -1)


class RelationRNNDecoder(Net):
    def __init__(self, model_config):
        super(RelationRNNDecoder, self).__init__(model_config)
        base_enc_dim = model_config.RMC.mem_slots * model_config.RMC.num_heads * model_config.RMC.head_size
        query_rep = base_enc_dim * model_config.decoder.query_ents
        input_dim = query_rep + base_enc_dim
        output_dim = model_config.target_size
        self.decoder2vocab = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            nn.ReLU(),
            nn.Linear(input_dim//2, output_dim)
        )
        # self.decoder2vocab = self.get_mlp(input_dim, output_dim)

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
        encoder_hidden = batch.encoder_hidden
        query_rep = step_batch.query_rep
        encoder_outputs = batch.encoder_outputs
        # print(encoder_outputs.shape, encoder_hidden.shape, query_rep.shape)
        mlp_inp = torch.cat([query_rep.squeeze(1), encoder_hidden], -1)
        outp = self.decoder2vocab(mlp_inp)
        return outp, None, None


# def attention(query, key, value, mask=None, dropout=None):
#     "Compute 'Scaled Dot Product Attention'"
#     d_k = query.size(-1)
#     scores = torch.matmul(query, key.transpose(-2, -1)) \
#              / math.sqrt(d_k)
#     if mask is not None:
#         scores = scores.masked_fill(mask == 0, -1e9)
#     p_attn = F.softmax(scores, dim = -1)
#     if dropout is not None:
#         p_attn = dropout(p_attn)
#     return torch.matmul(p_attn, value), p_attn

