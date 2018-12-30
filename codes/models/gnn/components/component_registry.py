# Adding this a seperate file so that we can try different variants and combinations
from codes.models.gnn.components.message_function import EdgeMessageFunction, NodeMessageFunction
from codes.models.gnn.components.update_function import UpdateFunction, LSTMUpdateFunction
from codes.models.gnn.components.readout_function import MaskedReadoutFunction

def choose_message_function(model_config):
    if model_config.graph.message_function.fn_type == 'edge':
        return EdgeMessageFunction(model_config)
    elif model_config.graph.message_function.fn_type == 'node':
        return NodeMessageFunction(model_config)
    else:
        raise NotImplementedError("Update function {} not implemented".format(
            model_config.graph.message_function.fn_type))


def choose_update_function(model_config):
    if model_config.graph.update_function.fn_type == 'mlp':
        return UpdateFunction(model_config)
    elif model_config.graph.update_function.fn_type == 'lstm':
        return LSTMUpdateFunction(model_config)
    else:
        raise NotImplementedError("Update function {} not implemented".format(
            model_config.graph.update_function.fn_type))

def choose_readout_function(model_config):
    return MaskedReadoutFunction(model_config)
