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

        # memory & control size
        self.mac_size = model_config.embedding.dim
        if model_config.encoder.bidirectional:
            self.mac_size *= 2

        # how many reasoning step to do
        self.iteration = model_config.mac.num_iteration

        # 1. read unit: paragraph -> list of LSTM outputs
        self.reader = SimpleEncoder(model_config, shared_embeddings=self.embedding)
        # Optionally project query_rep, B x num_ents*dim -> B x dim
        self.projQuery = None
        if model_config.mac.projQuery:
            self.projQuery = nn.Linear(self.mac_size*model_config.decoder.query_ents, self.mac_size)

        # 2. MAC Cell
        self.MAC = MACCell(model_config, self.mac_size, self.iteration)


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
        encoder_outputs, query_mask = batch.knowledgeBase, batch.query_mask
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

        query_rep = torch.cat(query_reps, -1).squeeze(1)

        return query_rep, query_rep_all


    def forward(self, batch):
        # Input Unit
        knowledgeBase, _ = self.reader(batch)
        batch.knowledgeBase = knowledgeBase
        query_rep, query_rep_all = self.calculate_query(batch)

        if self.projQuery:
            query_rep = self.projQuery(query_rep)  # B x num_ents*dim -> B x dim

        batch.query_rep = query_rep

        # MAC Unit
        batch_size = knowledgeBase.size(0)
        memory, control = self.MAC.init_state(batch, batch_size)
        for i in range(self.iteration):
            memory, control = self.MAC(knowledgeBase, query_rep, query_rep_all, memory, control, i)
        # encoder return: encoder_output, encoder_hidden
        return memory, None


class MACNetworkDecoder(Net):
    """
    Output Unit of MAC Network, which is an MLP.
    """
    def __init__(self, model_config, share_embeddings=None):
        super(MACNetworkDecoder, self).__init__(model_config)

        base_enc_dim = model_config.embedding.dim
        if model_config.encoder.bidirectional:
            base_enc_dim *= 2

        query_rep = base_enc_dim
        if not model_config.projQuery:
            query_rep *= model_config.decoder.query_ents

        self.output_unit = self.get_mlp(query_rep + base_enc_dim,
                                        model_config.target_size, num_layers=2)

    def forward(self, batch, step_batch):
        query_rep = batch.query_rep
        enc_outputs = batch.encoder_outputs
        out = self.output_unit(torch.cat([query_rep, enc_outputs], -1))
        return out, None, None

    def init_hidden(self, encoder_outputs, batch_size):
        return None


    def calculate_query(self, batch):
        return None


class MACCell(Net):

    def __init__(self, model_config, hidden_size, iteration):
        super(MACCell, self).__init__(model_config)

        self.model_config = model_config
        self.hidden_size = hidden_size
        self.iteration = iteration

        # base_enc_dim = model_config.embedding.dim
        # if model_config.encoder.bidirectional:
        #     base_enc_dim *= 2
        query_rep_dim = hidden_size
        if not model_config.projQuery:
            query_rep_dim *= model_config.decoder.query_ents

        # linear transformate query_rep into a position-aware vector
        # B, 2*dim -> B, dim
        self.transformQuestion_1 = nn.Linear(query_rep_dim, hidden_size)
        if not model_config.mac.shareQuestion:
            self.transformQuestion_2 = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for i in range(iteration)])
        else:
            self.transformQuestion_2 = nn.Linear(hidden_size, hidden_size)

        # ---Control Unit
        self.contControl = nn.Linear(hidden_size*2, hidden_size)
        # optional, concate interaction and query_rep of each entities (2*dim), then project back to dim
        self.controlProj = nn.Linear(hidden_size*2, hidden_size)
        # control attention
        self.controlAttn = nn.Linear(hidden_size, 1)

        # Read Unit
        self.transformMemory = nn.Linear(hidden_size, hidden_size)
        self.transformKB = nn.Linear(hidden_size, hidden_size)
        self.combineInfo = nn.Linear(hidden_size*2, hidden_size)
        self.readAttn    = nn.Linear(hidden_size, 1)

        # Write Unit
        self.writeNewMemory = self.get_mlp(hidden_size*2, hidden_size, num_layers=1)

        # dropout
        self.memDrop = nn.Dropout(model_config.mac.dropout.memory)
        self.readDrop = nn.Dropout(model_config.mac.dropout.read)
        self.writeDrop = nn.Dropout(model_config.mac.dropout.write)


    def init_state(self, batch, batch_size):

        # self.c = nn.Parameter(torch.rand(hidden_size))
        # self.m = nn.Parameter(torch.rand(hidden_size))
        initMemory = torch.rand(batch_size, self.hidden_size).to(batch.inp.device)
        initCtrl = torch.rand(batch_size, self.hidden_size).to(batch.inp.device)
        return initMemory, initCtrl


    def control(self, query_rep, query_rep_all, ctrl, contCtrl=None):
        """
        :param query_rep:     [B, dim], concatenation of two query entities after a time-step specific linear transformation
        :param query_rep_all: [B, entity_num, dim] (typically entity_num = 2)
        :param ctrl:          [B, dim], previous control state
        :param contCtrl:      [B, dim], previous continuous control state (optional)
        :return ctrl:         [B, dim], new control state
        :return contCtrl:     [B, dim], new continuous control state
        """
        n_batch = query_rep.size(0)
        n_entity = query_rep_all.size(1)

        # 1: compute "continuous" control state given previous control and question.
        newContCtrl = self.contControl(torch.cat([ctrl, query_rep], -1)).tanh()  # B x dim

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
        attnLogit = self.controlAttn(interactions)                # B x entity_num x 1
        attnWeight = F.softmax(attnLogit, dim=1)                  # B x entity_num x 1
        newCtrl = (attnWeight * query_rep_all).sum(dim=1).squeeze(1)  # B x dim

        return newCtrl, newContCtrl


    def read(self, knowledgeBase, prevMemory, curControl):
        """
        :param knowledgeBase: [B, T, dim], output of LSTM given a paragraph
        :param prevMemory:    [B, dim]
        :param curControl:    [B, dim]
        :return: r_i, [B, dim], retrieved info from knowledgeBase
        """
        if True:
            prevMemory = self.memDrop(prevMemory)

        mem = self.transformMemory(prevMemory)  # B x dim
        KB  = self.transformKB(knowledgeBase)   # B x T x dim
        if True:
            mem = self.readDrop(mem)
            KB  = self.readDrop(KB)
        mem_KB = mem.unsqueeze(1) * KB          # B x T x dim

        if True:
            mem_KB = mem_KB.relu()

        # 2. combine and linearly transform new and old knowledge
        combinedInfo = self.combineInfo(torch.cat([mem_KB, knowledgeBase], dim=-1))  # B x T x dim
        interactions = torch.unsqueeze(curControl, 1) * combinedInfo

        if True:
            interactions = interactions.relu()

        # 3. attn, query: ctrl, key: retrievedInfo, value: KB
        attnLogit = self.readAttn(interactions)             # B x T x 1
        attnWeight = torch.softmax(attnLogit, dim=1)        # B x T x 1
        newInfo = (attnWeight * knowledgeBase).sum(dim=1).squeeze(1)  # B x dim

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
        if True:
            info = self.writeDrop(info)
        newMemory = torch.cat([info, memory], dim=1)    # B x 2*dim
        newMemory = self.writeNewMemory(newMemory)      # B x dim

        return newMemory

    # todo add dropout
    def forward(self, knowledgeBase, query_rep, query_rep_all, memory, control, i):

        # print('@@@@@knowledgeBase:\t', knowledgeBase.shape)
        # print('@@@@@query_rep:\t', query_rep.shape)
        # print('@@@@@query_rep_all:\t', query_rep_all.shape)
        # print('@@@@@memory:\t', memory.shape)
        # print('@@@@@control:\t', control.shape)

        # transform query_rep to point-wise question_rep
        transformedQuery = self.transformQuestion_1(query_rep).tanh()            # 2*hidden_size -> hidden_size
        if self.model_config.mac.shareQuestion:
            transformedQuery = self.transformQuestion_2(transformedQuery)
        else:
            transformedQuery = self.transformQuestion_2[i](transformedQuery)  # hidden_size -> hidden_size

        if True:
            transformedQuery = transformedQuery.tanh()
        # CONTROL unit
        newCtrl, newContCtrl = self.control(transformedQuery, query_rep_all, control)

        # READ unit
        newInfo = self.read(knowledgeBase, memory, newCtrl)

        # WRITE unit
        newMemory = self.write(memory, newInfo, newCtrl)

        return newMemory, newCtrl