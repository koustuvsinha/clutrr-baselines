from comet_ml import Experiment, ExistingExperiment
from codes.experiment.experiment import run_experiment
from codes.utils.config import get_config
from codes.utils.util import set_seed, flatten_dictionary
from codes.utils.argument_parser import argument_parser
from addict import Dict
import os
import logging
import sys
import signal
import socket

base_path = os.path.dirname(os.path.realpath(__file__)).split('codes')[0]

def start(config, experiment):
    config = Dict(config)
    set_seed(seed=config.general.seed)
    run_experiment(config, experiment)

def resume(config, experiment):
    config = Dict(config)
    set_seed(seed=config.general.seed)
    run_experiment(config, experiment, resume=True)

# SLURM REQUEUE LOGIC
def get_job_id():
    if 'SLURM_ARRAY_JOB_ID' in os.environ:
        return '%s_%s' % (os.environ['SLURM_ARRAY_JOB_ID'],
                          os.environ['SLURM_ARRAY_TASK_ID'])
    else:
        return os.environ['SLURM_JOB_ID']



def requeue_myself():
    job_id = get_job_id()
    logging.warning("Requeuing job %s", job_id)
    os.system('scontrol requeue %s' % job_id)



def sig_handler(signum, frame):
    logger.warning("Signal handler called with signal " + str(signum))
    prod_id = int(os.environ['SLURM_PROCID'])
    job_id = get_job_id()
    logging.warning(
        "Host: %s - Global rank: %i" % (socket.gethostname(), prod_id))
    if prod_id == 0:
        requeue_myself()
    else:
        logging.warning("Not the master process, no need to requeue.")
    sys.exit(-1)


def term_handler(signum, frame):
    logging.warning("Signal handler called with signal " + str(signum))
    logging.warning("Bypassing SIGTERM.")


def init_signal_handler():
    """
    Handle signals sent by SLURM for time limit / pre-emption.
    """
    signal.signal(signal.SIGUSR1, sig_handler)
    signal.signal(signal.SIGTERM, term_handler)
    logging.warning("Signal handler installed.")

if __name__ == '__main__':
    init_signal_handler()
    config_id, exp_id = argument_parser()
    print(config_id)
    if len(exp_id) == 0:
        config = get_config(config_id=config_id)
        log_base = config['general']['base_path']
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("{0}/{1}.log".format(log_base, config_id)),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger()
        logger.info("Running new experiment")
        ex = Experiment(api_key=config.log.comet.api_key,
                        workspace=config.log.comet.workspace,
                        project_name=config.log.comet.project_name,
                        disabled=True,
                        auto_output_logging=None,
                        log_code=False)
        name = 'exp_{}'.format(config_id)
        config.general.exp_name = name
        ex.log_parameters(flatten_dictionary(config))
        ex.set_name(name)
        start(config, ex)
    else:
        logging.info("Resuming old experiment with id {}".format(exp_id))
        config = get_config(config_id=config_id)
        logger = logging.getLogger()
        ex = ExistingExperiment(
            api_key=config.log.comet.api_key,
            previous_experiment=exp_id,
            workspace=config.log.comet.workspace,
            project_name=config.log.comet.project_name,
            disabled=config.log.comet.disabled,
            auto_output_logging=None,
            log_code=False,)
        name = 'exp_{}'.format(config_id)
        config.general.exp_name = name
        resume(config, ex)
