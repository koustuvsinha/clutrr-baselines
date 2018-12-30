import torch
from codes.models.gnn.components.component_registry import choose_message_function, \
    choose_readout_function, choose_update_function

from codes.net.base_net import Net
from codes.net.attention import Attn
from codes.utils.util import merge_first_two_dims_batch, unmerge_first_two_dims_batch

class MPNNModel(Net):
    '''
    Base class for all the message passing based models
    '''

    def __init__(self, model_config):
        super().__init__(model_config=model_config)
        self.name = "base_mpnn_model"
        self.description = "This is the base class for all the message passing based models. " \
                           "All the other models should extend this class. " \
                           "It is not to be used directly."

        self.message_function = self.get_message_function()
        self.update_function = self.get_update_function()
        self.readout_function = self.get_readout_function()
        if model_config.graph.message_function.use_attention:
            concat_size = model_config.graph.node_dim * 2 + model_config.graph.pos_dim + model_config.graph.feature_dim
            self.msg_attn = Attn('concat',model_config.graph.node_dim, concat_size=concat_size)

    def get_message_function(self):
        return choose_message_function(self.model_config)

    def get_update_function(self):
        return choose_update_function(self.model_config)

    def get_readout_function(self):
        return choose_readout_function(self.model_config)

    def forward(self, data):
        """
        data is a tuple of form (g, e, h_in, h_pos)
        g: graph of shape batch x n x n
        e: edge labels - batch x n x n x edge_dim
        h_in: input node embedding, randomly generate for now - batch x n x node_dim
        h_pos: one hot vector of the nodes
        """

        g, e, h_in, h_pos, mask = data
        # Could be skipped later if we ensure the types
        g = g.float()
        e = e.float()
        #h_in = self.embedding(h_in)
        _neighbor_count = torch.sum(g, dim=2).unsqueeze(2)
        neighbor_count = torch.max(torch.ones_like(_neighbor_count), _neighbor_count)
        # This is used for doing say averaging over neighbours

        h_t = h_in
        c_t = torch.zeros_like(h_t)
        # annealing strategy
        alpha = 1

        num_nodes = h_in.size(1)
        # h_t[:, v, :] = h_t_u.clone() * mask[:, v,


        for t in range(0, self.model_config.graph.num_message_rounds):

            message = self.message_function((h_t, h_pos, h_t, h_pos, e, g, alpha))
            if self.model_config.graph.message_function.use_attention:
                # calculate attention over the neighbors here (like GAT)
                node_rep = torch.cat([h_t, h_pos], dim=2)
                scores = self.msg_attn(node_rep, message, g)
                scores = merge_first_two_dims_batch(scores).unsqueeze(1)
                message = merge_first_two_dims_batch(message)
                message = scores.bmm(message).squeeze(1)
                message = unmerge_first_two_dims_batch(message,
                                                       second_dim=num_nodes)
            else:
                # just average the neighbors
                message = torch.sum(message, dim=2) / neighbor_count

            h_t_u, c_t_u = self.update_function((h_t.clone(), message.clone(), 1, c_t.clone()))

            h_t = h_t_u.clone() * mask
            c_t = c_t_u.clone() * mask

            # for v in range(0, num_nodes):
            #     message = self.message_function((h_t[:, v, :], h_pos[:, v, :],
            #                                      h_t, h_pos, e[:, v, :], g[:,v,:], alpha))
            #     message = g[:, v, :].unsqueeze(2).expand_as(message) * message # B x max_nodes x message_dim
            #     if self.model_config.message_function.use_attention:
            #         # calculate attention over the neighbors here (like GAT)
            #         node_rep = torch.cat([h_t[:, v, :], h_pos[:, v, :]],dim=1)
            #         scores = self.msg_attn(node_rep, message.transpose(0,1), g[:,v,:])
            #         message = scores.bmm(message).squeeze(1)
            #     else:
            #         # just average the neighbors
            #         message = torch.sum(message, dim=1) / neighbor_count[:, v, :]
            #
            #     h_t_u, c_t_u = self.update_function((h_t[:, v, :].clone(), message.clone(), t, c_t[:,v,:].clone()))
            #     h_t[:, v, :] = h_t_u.clone() * mask[:, v, :] # mask to make sure we are not updating the nodes who are not present in the current graph
            #     # message = self.message_function((h_t, h_pos,
            #     #                                  h_t, h_pos, e, g, alpha))
            #     c_t[:, v, :] = c_t_u.clone() * mask[:, v, :]

            alpha = 1  # decide an annealing strategy here
        return h_t
