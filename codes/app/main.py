import random

import numpy as np
import torch

from comet_ml import Experiment
from codes.experiment.experiment import run_experiment
from codes.utils.config import get_config
from codes.utils.util import set_seed
from codes.utils.argument_parser import argument_parser
from addict import Dict
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def start(config, experiment):
    config = Dict(config)
    set_seed(seed=config.general.seed)
    run_experiment(config, experiment)

if __name__ == '__main__':
    config_id = argument_parser()
    print(config_id)
    config = get_config(config_id=config_id)
    ex = Experiment(api_key=config.log.comet.api_key,
                    workspace=config.log.comet.workspace,
                    project_name=config.log.comet.project_name)
    name = 'exp_{}'.format(config_id)
    config.general.exp_name = name
    ex.log_parameters(config)
    ex.set_name(name)
    start(config, ex)
