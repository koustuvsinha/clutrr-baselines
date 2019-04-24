# placeholder class for one batch
import torch
import numpy as np
import itertools as it

class Batch:
    """
    Placeholder class for one batch
    """
    def __init__(self,
            inp=None,                   # story input, (B x s)
            inp_lengths=None,           # length of story, (B x 1)
            s_inp=None,  # story input in sentences (B x s x w) in sentence mode
            sent_lengths=None,          # B x s x 1
            target=None,                # target of the relation, (B x 1)
            text_target = None,         # target in text, (B x t)
            text_target_lengths = None, # target lengths, (B x 1)
            query = None,               # query relation pair, (B x 2)
            query_mask = None,          # query mask over input, (B x s x 2)
            query_text = None,          # query_text input, (B x q)
            query_text_lengths = None,  # query_text lengths, (B x 1)
            inp_ents = None,            # entities per story, (B x e)
            inp_ent_mask = None,        # mask over story which specifies entities, (B x s) / (B x s x w) in sentence mode
            inp_graphs = None,          # story graphs, (B x n x n), where n = max entity in dataset
            sentence_pointer = None,    # each pair of nodes point to a specific sentence by using a one-hot vector over the sentences (in batch mode), (B x n x n x w)
            config = None,
            orig_inp = None,            # Unmodified input
            orig_inp_sent = None,       # Unmodified input, sentence tokenized (list of list)
            inp_row_pos = None,         # position over input text (B x s x w)
            geo_batch = None,           # Pytorch Geometric Batch object (collection of Pytorch Data objects),
            geo_slices = None,          # Pytorch Geometric slices, to restore the original splits
            query_edge = None,          # tensor B x 2 of query edges
            bert_inp = None,            # tensor B x s, right now this contains the entity ids to be used with bert lstm
            bert_input_mask=None,       # input mask, 1 for words and 0 for padding
            bert_segment_ids=None,      # segment id, unique for each sentence
            ):

        """

        :param inp:                     story input, (B x s) / (B x s x w) in sentence mode
        :param inp_lengths:             length of story, (B x 1) / (B x s x 1) in sentence mode
        :param sent_lengths:            B x s x 1
        :param target:                  target of the relation, (B x 1)
        :param text_target:             target in text, (B x t)
        :param text_target_lengths:     target lengths, (B x 1)
        :param query:                   query relation pair, (B x 2)
        :param query_mask:              query mask over input, (B x s x 2)
        :param query_text:              query_text input, (B x q)
        :param query_text_lengths:      query_text lengths, (B x 1)
        :param inp_ents:                entities per story, (B x e)
        :param inp_ent_mask:            mask over story which specifies entities, (B x s) / (B x s x w) in sentence mode
        :param inp_graphs:              story graphs, (B x n x n), where n = max entity in dataset
        :param sentence_pointer:        each pair of nodes point to a specific sentence by using a one-hot vector over the sentences (in batch mode), (B x n x n x w)
        :param config:                  main config file
        :param orig_inp:                Unmodified input
        :param inp_row_pos:             position over input text (B x s x w)
        """

        self.inp = inp
        self.inp_lengths = inp_lengths
        self.s_inp = s_inp
        self.sent_lengths = sent_lengths
        self.target = target
        self.text_target = text_target
        self.text_target_lengths = text_target_lengths
        self.inp_ents = inp_ents
        self.query = query
        self.query_mask = query_mask
        self.query_text = query_text
        self.query_text_lengths = query_text_lengths
        self.inp_ent_mask = inp_ent_mask
        self.inp_graphs = inp_graphs
        self.config = config
        self.batch_size = inp.size(0)
        self.adj_mat = None
        self.encoder_outputs = None
        self.encoder_hidden = None
        # backpointer to encoder model for decoder
        self.encoder_model = None
        self.sentence_pointer = sentence_pointer
        self.orig_inp = orig_inp
        self.orig_inp_sent = orig_inp_sent
        self.inp_row_pos = inp_row_pos
        self.geo_batch = geo_batch
        self.geo_slices = geo_slices
        self.query_edge = query_edge
        self.bert_inp = bert_inp
        self.bert_input_mask = bert_input_mask
        self.bert_segment_ids = bert_segment_ids

    def to_device(self, device):
        self.inp = self.inp.to(device)
        self.s_inp = self.s_inp.to(device)
        self.target = self.target.to(device)
        self.text_target = self.text_target.to(device)
        self.text_target_lengths = self.text_target.to(device)
        self.query = self.query.to(device)
        self.query_mask = self.query_mask.to(device)
        # self.inp_graphs = self.inp_graphs.to(device)
        # self.adj_mat = self.adj_mat.to(device)
        self.inp_ent_mask = self.inp_ent_mask.to(device)
        if self.inp_row_pos is not None:
            self.inp_row_pos = self.inp_row_pos.to(device)
        if self.sentence_pointer is not None:
            self.sentence_pointer = self.sentence_pointer.to(device)
        if self.geo_batch is not None:
            self.geo_batch = self.geo_batch.to(device)
        if self.query_edge is not None:
            self.query_edge = self.query_edge.to(device)
        if self.bert_inp is not None:
            self.bert_inp = self.bert_inp.to(device)
        if self.bert_input_mask is not None:
            self.bert_input_mask = self.bert_input_mask.to(device)
        if self.bert_segment_ids is not None:
            self.bert_segment_ids = self.bert_segment_ids.to(device)

    def _process_adj_mat(self):
        """
        Deprecated.
        Get adjacency matrix of size B x n_e x n_e x n_s x n_dim
        """
        n_e = self.inp_graphs.size(1)
        n_s = self.config.model.graph.num_reads
        n_dim = self.config.model.graph.edge_dim
        self.adj_mat = torch.zeros((self.batch_size, n_e, n_e, n_s, n_dim))

    def clone(self):
        return Batch(inp=self.inp.clone().detach(),
                     inp_lengths=self.inp_lengths,
                     s_inp=self.s_inp.clone().detach(),
                     sent_lengths=self.sent_lengths,  # B x s x 1
                     target=self.target.clone().detach(),  # target of the relation, (B x 1)
                     text_target=self.text_target.clone().detach(),  # target in text, (B x t)
                     text_target_lengths=self.text_target.clone().detach(),  # target lengths, (B x 1)
                     query=self.query.clone().detach(),  # query relation pair, (B x 2)
                     query_mask=self.query_mask.clone().detach(),  # query mask over input, (B x s x 2)
                     query_text=self.query_text,  # query_text input, (B x q)
                     query_text_lengths=self.query_text_lengths,  # query_text lengths, (B x 1)
                     inp_ents=self.inp_ents,  # entities per story, (B x e)
                     inp_ent_mask=self.inp_ent_mask.clone().detach(),
                     # mask over story which specifies entities, (B x s) / (B x s x w) in sentence mode
                     inp_graphs=None,  # story graphs, (B x n x n), where n = max entity in dataset
                     sentence_pointer=None,
                     # each pair of nodes point to a specific sentence by using a one-hot vector over the sentences (in batch mode), (B x n x n x w)
                     config=self.config,
                     orig_inp=self.orig_inp,  # Unmodified input
                     orig_inp_sent=self.orig_inp_sent, # Unmodified input, sentence tokenized
                     inp_row_pos=None,  # position over input text (B x s x w)
                     geo_batch=self.geo_batch,  # Pytorch Geometric Batch object (collection of Pytorch Data objects),
                     geo_slices=self.geo_slices,  # Pytorch Geometric slices, to restore the original splits
                     query_edge=self.query_edge.clone().detach(),
                     bert_inp=self.bert_inp.clone().detach(), # right now this contains the entity ids to be used with bert lstm
                     bert_input_mask=self.bert_input_mask.clone().detach(),
                     bert_segment_ids=self.bert_segment_ids.clone().detach()
                     )




