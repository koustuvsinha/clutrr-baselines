import itertools as it
from collections import namedtuple
from codes.utils.config import get_config
import os
from copy import deepcopy
import yaml
import json
import operator
from functools import reduce
from codes.utils.util import flatten_dictionary
import argparse
import glob
import numpy as np

def getFromDict(dataDict, mapList):
    return reduce(operator.getitem, mapList, dataDict)

def setInDict(dataDict, mapList, value):
    getFromDict(dataDict, mapList[:-1])[mapList[-1]] = value

def create_list_of_Hyperparams(hyperparams_dict):
    sep = "$$"
    Hyperparam = namedtuple("Hyperparam", ["key_list", "value"])
    flattend_dict = flatten_dictionary(hyperparams_dict)
    Hyperparam_list = []
    for key, val_list in flattend_dict.items():
        temp_list = []
        keylist = key.split(sep)
        for val in val_list:
            temp_list.append(Hyperparam(keylist, val))
        Hyperparam_list.append(temp_list)
    return it.product(*Hyperparam_list, repeat=1)

def create_configs(config_id):
    base_config = get_config(config_id=config_id)
    current_id = 0
    # for general
    hyperparams_dict = {
        "model": {
            "optimiser":{
                "learning_rate": [0.1, 0.01, 0.001, 0.0001]
            },
            "embedding": {
                    "dim": [50, 100, 150, 200, 250, 300]
            }
        }
    }

    if config_id == 'rn':
        # for bilstm
        hyperparams_dict.update({
            "model": {
                "rn": {
                    "g_theta_dim": [64, 128, 256],
                    "f_theta": {
                        "dim_1": [64, 128, 256, 512],
                        "dim_2": [64, 128, 256, 512]
                    }
                }
            }
        })

    if config_id == 'rn_tpr':
        hyperparams_dict.update({
            "model": {
                "rn": {
                    "g_theta_dim": [64, 128, 256],
                    "f_theta": {
                        "dim_1": [64, 128, 256, 512],
                        "dim_2": [64, 128, 256, 512]
                    }
                }
            }
        })

    if config_id == 'mac':
        hyperparams_dict.update({
            "model": {
                "rn": {
                    "g_theta_dim": [64, 128, 256],
                    "f_theta": {
                        "dim_1": [64, 128, 256, 512],
                        "dim_2": [64, 128, 256, 512]
                    }
                }
            }
        })

    if config_id == 'gat_clean':
        hyperparams_dict.update({
            "model": {
                "graph": {
                    "message_dim": [50, 100, 150, 200],
                    "num_message_rounds": [1,2,3,4,5]
                }
            }
        })


    path = os.path.dirname(os.path.realpath(__file__)).split('/codes')[0]
    target_dir = os.path.join(path, "config")

    for hyperparams in create_list_of_Hyperparams(hyperparams_dict):
        new_config = deepcopy(base_config)
        current_str_id = config_id + "_hp_" + str(current_id)
        new_config["general"]["id"] = current_str_id
        new_config["model"]["checkpoint"] = False
        # new_config["log"]["base_path"] = "/checkpoint/koustuvs/clutrr/"
        for hyperparam in hyperparams:
            setInDict(new_config, hyperparam.key_list, hyperparam.value)
        new_config_file = target_dir + "/{}.yaml".format(current_str_id)
        with open(new_config_file, "w") as f:
            f.write(yaml.dump(yaml.load(json.dumps(new_config)), default_flow_style=False))
        current_id += 1

def create_run_file(args):
    path = os.path.dirname(os.path.realpath(__file__)).split('/codes')[0]
    config_dir = os.path.join(path, "config")
    config_files = glob.glob(os.path.join(config_dir, args.model + '_hp*'))
    script_dir = os.path.join(path, "scripts")
    if not os.path.exists(script_dir):
        os.makedirs(script_dir)
    print("Found {} config files".format(len(config_files)))
    run_file = "#!/bin/sh\n"
    if not args.local:
        run_file += "#SBATCH --job-name=hyperparam_{}\n".format(args.model)
        run_file += "#SBATCH --output=/checkpoint/koustuvs/jobs/{}.out\n".format(args.model)
        run_file += "#SBATCH --error=/checkpoint/koustuvs/jobs/{}.err\n".format(args.model)
        run_file += "#SBATCH --partition=uninterrupted\n"
        run_file += "#SBATCH --nodes=1\n"
        run_file += "#SBATCH --ntasks-per-node=1\n"
        run_file += "#SBATCH --gres=gpu:1\n"
        run_file += "#SBATCH --cpus-per-task 24\n"
        run_file += "#SBATCH --time 01:00:00\n"
    run_file += "source activate gnnlogic\n"
    run_file += "cd {}\n".format(script_dir)
    run_file += "export COMET_API='{}'\n".format(args.comet_api)
    run_file += "export COMET_WORKSPACE='{}'\n".format(args.comet_workspace)
    run_file += "export COMET_PROJECT='{}'\n".format(args.comet_project)
    run_file += "export PYTHONPATH={}\n".format(path)
    gpus = args.gpu.split(',')
    print("Found {} gpus".format(len(gpus)))
    chunk_config_ids = np.arange(len(config_files))
    chunk_config_ids = list(np.array_split(chunk_config_ids, len(gpus)))
    print("Partitioning runs in the gpus : {}".format([len(t) for t in chunk_config_ids]))
    for gpu_id, chunk in enumerate(chunk_config_ids):
        mini_run = "#!/bin/sh\n"
        for config_file_id in chunk:
            config_file = config_files[config_file_id]
            cname = config_file.split('.yaml')[-2].split('/')[-1]
            pre = 'CUDA_VISIBLE_DEVICES={} '.format(gpu_id)
            mini_run += pre + "python {}/codes/app/main.py --config_id {} > /dev/null 2>&1\n".format(path, cname)
        mini_file = 'mini_{}_{}.sh'.format(args.model, gpu_id)
        with open(os.path.join(script_dir, mini_file), 'w') as mp:
            mp.write(mini_run)
        run_file += "sh {} > /dev/null 2>&1 &\n".format(mini_file)
    with open('{}/{}_hyp_run.sh'.format(script_dir, args.model), 'w') as fp:
        fp.write(run_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--comet_api", type=str, default='')
    parser.add_argument("--comet_workspace", type=str, default='koustuvs')
    parser.add_argument("--comet_project", type=str, default='compositionality-nli')
    parser.add_argument('--model', type=str, default='bilstm', help='either one in bilstm,gat_clean,lstm_atten,mac,rn,rn_tpr')
    parser.add_argument('--local', action='store_true', help="If true, run on machines not on slurm")
    parser.add_argument('--gpu', type=str, default='0', help='works in local, run jobs on this gpu')
    parser.add_argument('--stdout', type=str, default='std_outputs', help='folder to store std outputs')
    parser.add_argument('--script_dir', type=str, default='scripts', help='folder to dump all these scripts')
    args = parser.parse_args()

    create_configs(args.model)
    create_run_file(args)
