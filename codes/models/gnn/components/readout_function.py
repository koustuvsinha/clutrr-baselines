import torch
import torch.nn as nn

from codes.models.gnn.components.base_component import BaseComponent
from codes.net.attention import Attn


class MaskedReadoutFunction(BaseComponent):

    def __init__(self, model_config):
        super().__init__(model_config=model_config)
        self.name = "readout_function"
        self.input_dim = self.config.graph.node_dim * 2 * self.config.query_entities
        #self.readout_fn = self.get_readout_function()

    def get_readout_function(self):

        output_dim = self.config.decoder.hidden_dim
        return self.get_mlp(self.input_dim, output_dim,
                            self.config.graph.readout_function.num_layers)


    def forward(self, h_T, h_pos, v, edge_fn):
        """
        Get the relation between nodes in v
        :param h_T: B x max_nodes x node_dim
        :param v: B x max_nodes x num_ents - 1-0 mask
        :param edge_fn: edge_func pointer
        :param
        :return:
        """
        masked_h_t = torch.bmm(h_T.transpose(1,2), v.float()) # B x node_dim x num_ents
        masked_h_t = masked_h_t.view(masked_h_t.size(0), -1) # B x (node_dim x num_ents)
        masked_h_pos = torch.bmm(h_pos.transpose(1,2), v.float())
        masked_h_pos = masked_h_pos.view(masked_h_pos.size(0), -1) # B x (pos_dim x num_ents)
        assert masked_h_t.size(0) == h_T.size(0)
        masked_h_t_pos = torch.cat([masked_h_t, masked_h_pos],dim=-1)

        batch_size = masked_h_t.size(0)
        edge_rep = edge_fn(masked_h_t_pos).unsqueeze(1)
        #comb_rep = torch.cat([edge_rep, masked_h_t.unsqueeze(1)],dim=-1)
        return edge_rep

class AttentiveReadoutFunction(BaseComponent):
    """
    Readout function that uses attention to extract the relation
    For each timestep, calculate the attention as follows:

    a_j = [h_j; x_j]
        where h_j is the last state of node after message passing
        x_j : decoder state

        decoder_hidden_state + node hidden state
    """

    def __init__(self, model_config, concat_size=None):
        super().__init__(model_config=model_config)
        self.name = "attentive_readout_function"
        self.attn = Attn('concat', model_config.graph.node_dim, concat_size)


    def forward(self, h_T, last_hidden, mask):
        """
        :param h_T: B x max_nodes x node_dim
        :param last_hidden: B x hidden_dim
        :param mask: B x max_nodes
        :return:
        """
        attn_weights = self.attn(last_hidden, h_T.transpose(0,1), mask) # B x 1 x max_nodes
        context = attn_weights.bmm(h_T)

        return context

class AverageReadout(BaseComponent):
    """
    Readout function which just averages the last states of all nodes into one
    """
    def __init__(self, model_config):
        super().__init__(model_config=model_config)
        self.name = "averaged_readout_function"

    def forward(self, h_T, mask):
        """

        :param h_T: B x max_nodes x node_dim
        :param mask: B x max_nodes x 1
        :return: B x node_dim
        """
        node_count = torch.sum(mask, dim=1).float() # B x 1
        return torch.sum(h_T, dim=1) / node_count
