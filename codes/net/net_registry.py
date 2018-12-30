import importlib
from copy import deepcopy

from codes.utils.config import get_config


def prepare_config_for_model(config, num_nodes=0):
    model_config = deepcopy(config.model)
    model_config.num_nodes = num_nodes
    return model_config


def choose_model(config):
    """
    Dynamically load both encoder and decoder
    :param config:
    :return:
    """
    model_config = prepare_config_for_model(config)
    encoder_model_name = model_config.encoder.name
    encoder_module = _import_module(encoder_model_name)
    decoder_model_name = model_config.decoder.name
    decoder_module = _import_module(decoder_model_name)
    return encoder_module(model_config), decoder_module(model_config)

def _import_module(full_module_name):
    """
    Import className from python file
    https://stackoverflow.com/a/8790232
    :param full_module_name: full resolvable module name
    :return: module
    """
    path, name = full_module_name.rsplit('.',1)
    base_module = importlib.import_module(path)
    module = getattr(base_module, name)
    return module


if __name__ == "__main__":
    config = get_config()
    model = choose_model(config, 10)
    print(model)
