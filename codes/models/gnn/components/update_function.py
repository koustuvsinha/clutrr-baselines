import torch
import torch.nn as nn

from codes.models.gnn.components.base_component import BaseComponent


class UpdateFunction(BaseComponent):

    def __init__(self, model_config):
        super().__init__(model_config=model_config)
        self.name = "update_function"
        self.update_function = self.get_update_function()

    def get_update_function(self):
        input_dim = self.config.graph.edge_dim + self.config.graph.node_dim
        output_dim = self.config.graph.node_dim
        return self.get_mlp(input_dim, output_dim,
                            self.config.graph.update_function.num_layers, dropout=self.config.graph.dropout)

    # Update node hv given message mv
    def forward(self, data):
        """
        mp_t : if index 0, then just copy the message directly into h_new
        :param data:
        :return:
        """
        h_v, m_v, mp_t, _ = data
        if mp_t == 0:
            h_new = m_v
        else:
            h_new = self.update_function(torch.cat((m_v, h_v), dim=1))

        return h_new, _

class LSTMUpdateFunction(BaseComponent):

    def __init__(self, model_config):
        super().__init__(model_config=model_config)
        self.name = "lstm_update_function"
        self.update_function = nn.LSTM(
            model_config.graph.edge_dim,
            model_config.graph.edge_dim,
            bidirectional=False
        )
        self.layer_norm = nn.LayerNorm((model_config.graph.edge_dim))

    # Update node hv given message mv
    def forward(self, data):
        """
        mp_t : if index 0, then just copy the message directly into h_new
        Note that this function will have to be changed if the lstm is bidirectional (which is not the case for now)
        :param data:
        :return:
        """
        h_v, m_v, mp_t, c_t = data
        shape = h_v.shape
        h_v = h_v.view(shape[0]*shape[1], shape[2])
        m_v = m_v.view(shape[0]*shape[1], shape[2])
        c_t = c_t.view(shape[0]*shape[1], shape[2])
        if self.model_config.graph.update_function.use_layer_norm:
            h_v = self.layer_norm(h_v)
            c_t = self.layer_norm(c_t)
        # h_v = h_v.transpose(1, 0).contiguous()
        # c_t = c_t.transpose(1, 0).contiguous()
        # m_v = m_v.transpose(1, 0).contiguous()
        h_v = h_v.unsqueeze(0).contiguous()
        c_t = c_t.unsqueeze(0).contiguous()
        m_v = m_v.unsqueeze(0).contiguous()
        _, (h_new, c_new) = self.update_function(m_v, (h_v, c_t))
        h_new = h_new.squeeze(0) # B x dim
        c_new = c_new.squeeze(0) # B x dim




        return h_new.view(*shape), c_new.view(*shape)