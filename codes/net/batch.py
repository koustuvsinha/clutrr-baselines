# placeholder class for one batch
import torch
import networkx as nx
import numpy as np
import itertools as it

class Batch:
    """
    Placeholder class for one batch
    """
    def __init__(self,
            inp=None,
            inp_lengths=None,
            sent_lengths=None,
            outp = None,
            outp_lengths = None,
            ent_mask = None,
            inp_ents = None,
            inp_ent_mask = None,
            outp_ents = None,
            inp_graphs = None,
            sentence_pointer = None,
            config = None,
            orig_inp = None,
            inp_row_pos = None,
            ):

        self.inp = inp
        self.inp_lengths = inp_lengths
        self.sent_lengths = sent_lengths
        self.outp = outp
        self.outp_lengths = outp_lengths
        self.inp_ents = inp_ents
        self.outp_ents = outp_ents
        self.ent_mask = ent_mask
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
        self.inp_row_pos = inp_row_pos

    def to_device(self, device):
        self.inp = self.inp.to(device)
        self.outp = self.outp.to(device)
        self.outp_ents = self.outp_ents.to(device)
        self.inp_graphs = self.inp_graphs.to(device)
        self.adj_mat = self.adj_mat.to(device)
        self.ent_mask = self.ent_mask.to(device)
        self.inp_ent_mask = self.inp_ent_mask.to(device)
        if self.inp_row_pos is not None:
            self.inp_row_pos = self.inp_row_pos.to(device)
        if self.sentence_pointer is not None:
            self.sentence_pointer = self.sentence_pointer.to(device)

    def process_adj_mat(self):
        """
        Get adjacency matrix of size B x n_e x n_e x n_s x n_dim
        """
        n_e = self.inp_graphs.size(1)
        n_s = self.config.model.graph.num_reads
        n_dim = self.config.model.graph.edge_dim
        self.adj_mat = torch.zeros((self.batch_size, n_e, n_e, n_s, n_dim))

    def prune_paths(self, start_ids, end_ids):
        """
        Given a start node and a finish node, prune the adjacency graph so that only the
        paths between start and end are present and nothing else
        :param start: starting node
        :param end: end node
        :return: pruned adjacency matrix
        """
        pruned_gs = []
        ent_masks = []
        for b_indx in range(self.batch_size):
            max_nodes = self.inp_graphs[b_indx].size(0)
            ent_mask = np.zeros(max_nodes)
            g = nx.from_numpy_matrix(self.inp_graphs[b_indx].cpu().numpy())
            pruned_g = np.zeros((max_nodes, max_nodes))
            start = start_ids[b_indx]
            end = end_ids[b_indx]
            for path in nx.all_simple_paths(g, source=start, target=end):
                for idx in range(len(path) - 1):
                    node_a = path[idx]
                    node_b = path[idx+1]
                    pruned_g[node_a][node_b] = 1
                    pruned_g[node_b][node_a] = 1
                for node in path:
                    ent_mask[node] = 1
            pruned_gs.append(pruned_g)
            ent_masks.append(ent_mask)
        self.inp_graphs = torch.LongTensor(np.array(pruned_gs)).to(self.inp_graphs.device)
        self.inp_ent_mask = torch.LongTensor(np.array(ent_masks)).to(self.inp_graphs.device)





