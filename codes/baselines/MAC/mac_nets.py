"""
MAC-Network from `Compositional Attention Networks for Machine Reasoning`.
https://arxiv.org/pdf/1803.03067.pdf
Encoder:
    Input Unit: a SimpleEnocder (LSTM), paragraph -> lstm outputs cw_1,...,cw_m
                                        query: -> query_rep, query_rep_all(no concatenate)
    MACCell: reasoning iteratively, in each reasoning step
        Control Unit:   generate control state c_i given c_i-1 and time-step specific query_rep q_i
        Read Unit:      distill related info given c_i and knowledge base
        Write Unit:     generate new memory state m_i given m_i-1 and new info

    Output Unit: an MLP that maps the last memory state to an output distribution
"""
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

        # read unit: paragraph -> list of LSTM outputs
        self.paragraphReader = SimpleEncoder(model_config, shared_embeddings=self.embedding)

        # how many reasoning step to do
        self.iteration = model_config.mac.num_iteration

        # memory & control size
        base_mac_size = model_config.encoder.hidden_size
        bidirectional_mult = 1
        if model_config.encoder.bidirectional:
            bidirectional_mult = 2
        mac_size = base_mac_size * bidirectional_mult

        # MAC Cell
        self.MAC = MACCell(model_config, mac_size)


    def calculate_query(self, batch):
        """
        :param batch
                encoder_outputs: B x seq_len x dim
                ent_mask: B x max_abs x num_ents x seq_len
                outp_ents: B x num_ents (usually 2)
        :return:
                query_rep:      B x num_ents*dim
                query_rep_all:  B x num_ents x dim
        """
        encoder_outputs, query_mask = batch.encoder_outputs, batch.query_mask
        # expand
        num_ents = query_mask.size(2)
        seq_len = query_mask.size(1)
        # encoder_outputs # B x seq_len x dim
        query_mask = query_mask.transpose(1, 2)  # B x num_ents x seq_len

        query_rep_all = torch.bmm(query_mask.float(), encoder_outputs)  # B x num_ents x dim
        ents = query_rep_all.size(1)
        query_reps = []
        for i in range(ents):
            query_reps.append(query_rep_all[:, i, :].unsqueeze(1))

        query_rep = torch.cat(query_reps, -1)

        return query_rep, query_rep_all


    def forward(self, batch):
        # Input Unit
        knowledgeBase, _ = self.paragraphReader(batch)
        query_rep, query_rep_all = self.calculate_query(batch)
        batch.query_rep = query_rep

        # MAC Unit
        batch_size = knowledgeBase.size(0)
        memory, control = self.MAC.init_state(batch_size)
        for _ in range(self.iteration):
            memory, control = self.MAC(knowledgeBase, query_rep, query_rep_all, memory, control)

        return memory, None


class MACNetworkDecoder(Net):
    """
    Output Unit of MAC Network, which is an MLP.
    """
    def __init__(self, model_config, share_embeddings=None):
        super(MACNetworkDecoder, self).__init(model_config)

        base_enc_dim = model_config.encoder.hidden_size
        if model_config.encoder.bidirectional:
            base_enc_dim *= 2
        query_rep = base_enc_dim * model_config.decoder.query_ents

        self.output_unit = self.get_mlp(query_rep + base_enc_dim,
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


class MACCell(Net):

    # TODO
    def __init__(self, model_config, hidden_size):
        super(MACCell, self).__init__(model_config)

        self.model_config = model_config
        self.hidden_size = hidden_size


        base_enc_dim = model_config.embedding.dim
        if model_config.encoder.bidirectional:
            base_enc_dim *= 2
        query_rep_dim = base_enc_dim * model_config.decoder.query_ents

        # linear transformate query_rep into a position-aware vector
        # B, 2*dim -> B, dim
        self.transformQuestion = self.get_mlp(query_rep_dim,
                                               hidden_size, num_layers=2)

        # ---Control Unit
        self.contControl = nn.Linear(hidden_size*2, hidden_size)
        self.controlAttn = nn.Linear(hidden_size, 1)

        # Read Unit
        self.transformMemory = nn.Linear(hidden_size, hidden_size)
        self.transformKB = nn.Linear(hidden_size, hidden_size)
        self.combineInfo = nn.Linear(hidden_size*2, hidden_size)
        self.readAttn     = nn.Linear(hidden_size, hidden_size)

        # Write Unit
        self.writeNewMemory = self.get_mlp(hidden_size*2, hidden_size, num_layers=1)

    # TODO
    def init_state(self, batch_size):
        # self.c = nn.Parameter(torch.rand(hidden_size))
        # self.m = nn.Parameter(torch.rand(hidden_size))

        initMemory = torch.rand(batch_size, self.hidden_size)
        initCtrl = torch.rand(batch_size, self.hidden_size)


        return initMemory, initCtrl


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


    def read(self, knowledgeBase, prevMemory, curControl):
        """
        :param knowledgeBase: [B, T, dim], output of LSTM given a paragraph
        :param prevMemory:    [B, dim]
        :param curControl:    [B, dim]
        :return: r_i, [B, dim], retrieved info from knowledgeBase
        """
        mem = self.transformMemory(prevMemory)  # B x dim
        KB  = self.transformKB(knowledgeBase)   # B x T x dim
        mem_KB = mem.unsqueeze(1) * KB          # B x T x dim

        # 2. combine and linearly transform new and old knowledge
        combinedInfo = self.combineInfo(torch.cat([mem_KB, knowledgeBase], dim=-1))  # B x T x dim

        # 3. attn, query: ctrl, key: retrievedInfo, value: KB
        attnWeight = self.readAttn(curControl.unsqueeze(1) * combinedInfo)      # B x T x dim
        attnLogits = torch.softmax(attnWeight, dim=1)

        newInfo = (attnLogits * knowledgeBase).sum(dim=1).squeeze(1)

        return newInfo


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


    def forward(self, knowledgeBase, query_rep, query_rep_all, memory, control):

        newCtrl, newContCtrl = self.control(query_rep, query_rep_all, control)
        newInfo = self.read(knowledgeBase, memory, newCtrl)
        newMemory = self.write(memory, newInfo, newCtrl)

        return newMemory, newCtrl