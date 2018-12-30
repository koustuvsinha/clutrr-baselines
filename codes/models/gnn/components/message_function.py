import torch
import torch.nn as nn

from codes.models.gnn.components.base_component import BaseComponent
from codes.utils.util import aeq
from time import time

class EdgeMessageFunction(BaseComponent):
    """
    Uses Edge representation to calculate the message
    """
    def __init__(self, model_config):
        super().__init__(model_config)
        self.name = "edge_message_function"
        self.edge_network = self.edge_nop

    def edge_nop(self, data):
        return data

    def get_edge_network(self):
        pos_dim = self.model_config.max_nodes
        if self.model_config.graph.pos_rep == 'random':
            pos_dim = self.model_config.graph.pos_dim
        input_dim = self.model_config.graph.node_dim * 2 + pos_dim * 2 + self.model_config.graph.feature_dim * 2
        output_dim = self.model_config.graph.edge_dim
        return self.get_mlp(input_dim, output_dim,
                            self.config.graph.message_function.num_layers, dropout=self.config.graph.dropout)

    def forward(self, data):
        h_v, h_v_pos, h_w, h_w_pos, e_vw, g_vw, alpha = data
        batch_size = h_w.size(0)
        num_nodes = h_w.size(1)
        # message = torch.zeros(batch_size, num_nodes, self.model_config.graph.edge_dim, device=h_v.device)
        t0 = time()
        # true_edge_emb = torch.cat([e_vw, h_w_pos, h_v_pos.unsqueeze(1).expand(-1, num_nodes, -1)], dim=2)
        # comb_edge_rep = alpha * true_edge_emb
        # message = comb_edge_rep*g_vw.unsqueeze(2)

        a = torch.cat((h_w_pos, h_v_pos), dim=2).unsqueeze(1).expand(-1, num_nodes, -1, -1)
        message = torch.cat([e_vw, a], dim=3) * g_vw.unsqueeze(3)
        #

        # for w in range(num_nodes):
        #     if torch.nonzero(e_vw[:, w]).size()[0] > 0 and torch.nonzero(g_vw[:,w]).size()[0] > 0:
        #         hat_edge_emb = self.edge_network(torch.cat((h_w[:, w, :], h_v, h_w_pos[:,w,:], h_v_pos), dim=1))
        #         true_edge_emb = torch.cat([e_vw[:, w, :], h_w_pos[:,w,:], h_v_pos], dim=1) # B x (edge_dim-pos + pos)
        #         #aeq(hat_edge_emb, true_edge_emb)
        #         comb_edge_rep = alpha * true_edge_emb #+ (1 - alpha) * hat_edge_emb
        #         message[:, w, :] = comb_edge_rep * g_vw[:, w].unsqueeze(1)
        # print("Time for inner loop = {}".format(time() - t0))
        return message

class NodeMessageFunction(BaseComponent):
    """
    Vanilla message creation function with only node features
    """
    def __init__(self, model_config):
        super().__init__(model_config)
        self.name = "node_message_function"
        self.message_fn = self.get_msg_network()
        self.edge_network = self.edge_nop

    def edge_nop(self, data):
        return data

    def get_msg_network(self):
        pos_dim = self.model_config.max_nodes
        if self.model_config.graph.pos_rep == 'random':
            pos_dim = self.model_config.graph.pos_dim
        input_dim = self.model_config.graph.node_dim * 2 + pos_dim * 2 + self.model_config.graph.feature_dim * 2
        output_dim = self.model_config.graph.node_dim
        return self.get_mlp(input_dim, output_dim,
                            self.config.graph.message_function.num_layers, dropout=self.config.graph.dropout)

    def forward(self, data):
        h_v, h_v_pos, h_w, h_w_pos, e_vw, g_vw, alpha = data
        batch_size = h_w.size(0)
        num_nodes = h_w.size(1)
        message = torch.zeros(batch_size, num_nodes, self.model_config.graph.edge_dim, device=h_v.device)
        for w in range(num_nodes):
            if torch.nonzero(g_vw[:,w]).size()[0] > 0:
                message[:, w, :] = self.message_fn(torch.cat((h_w[:, w, :], h_v, h_w_pos[:,w,:], h_v_pos), dim=1))
        return message
