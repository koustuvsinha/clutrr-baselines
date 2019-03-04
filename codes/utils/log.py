import json

import numpy as np
from pathlib import Path
import pandas as pd
import os

from codes.utils.config import get_config
from contextlib import contextmanager

TIME = "time"
CONFIG = "config"
METRIC = "metric"
METADATA = "metadata"
TYPE = "type"
MESSAGE = "message"
PRINT = "print"
LOSS = "loss"
BATCH_SIZE = "batch_size"
TIME_TAKEN = "time_taken"
BATCH_INDEX = "batch_index"
MODE = "mode"
EPOCH_INDEX = "epoch_index"
BEST_EPOCH_INDEX = "best_epoch_index"
ITERATION_INDEX = "iteration_index"
TRAIN = "train"
VAL = "val"
TEST = "test"
LOG = "log"
BLEU = "bleu"
ENTOVERLAP = "entity_overlap"
RELOVERLAP = "rel_overlap"
FILE_PATH = "file_path"

log_types = [TIME, CONFIG, METRIC, METADATA]
ALL_KEYS = [LOSS, BLEU, ENTOVERLAP, RELOVERLAP, BATCH_SIZE,
                                    BATCH_INDEX, EPOCH_INDEX, TIME_TAKEN, MODE, ITERATION_INDEX]

def _format_log(log):
    return json.dumps(log)


def write_log(log):
    '''This is the default method to write a log. It is assumed that the log has already been processed
     before feeding to this method'''
    print(log)


def read_log(log):
    '''This is the single point to read any log message from the file since all the log messages are persisted as jsons'''
    try:
        data = json.loads(log)
    except json.JSONDecodeError as e:
        data = {
        }
    return data


def _format_custom_logs(keys=[], raw_log={}, _type=METRIC):
    log = {}
    if (keys):
        for key in keys:
            if key in raw_log:
                log[key] = raw_log[key]
    else:
        log = raw_log
    log[TYPE] = _type
    return _format_log(log), log


def write_message_logs(message):
    kwargs = {MESSAGE: message}
    log,_ = _format_custom_logs(keys=[], raw_log=kwargs, _type=PRINT)
    write_log(log)


def write_config_log(config):
    config[TYPE] = CONFIG
    log = _format_log(config)
    write_log(log)


def write_metric_logs(sacred_run=None, **kwargs):
    log, log_d = _format_custom_logs(keys=ALL_KEYS, raw_log=kwargs, _type=METRIC)
    write_log(log)
    sacred_run.log_scalar(log_d[MODE] + '.' + LOSS, log_d[LOSS],
                          log_d[ITERATION_INDEX])
    sacred_run.log_scalar(log_d[MODE] + '.' + RELOVERLAP, log_d[RELOVERLAP],
                          log_d[ITERATION_INDEX])

def write_metadata_logs(**kwargs):
    log,_ = _format_custom_logs(keys=[BEST_EPOCH_INDEX], raw_log=kwargs, _type=METADATA)
    write_log(log)


def pprint(config):
    print(json.dumps(config, indent=4))


def parse_log_file(log_file_path):
    logs = {}
    running_metrics = {}
    metric_keys = [LOSS, TIME_TAKEN]
    top_level_keys = [CONFIG, METADATA]
    for key in top_level_keys:
        logs[key] = []
    for mode in [TRAIN, VAL, TEST]:
        logs[mode] = {}
        running_metrics[mode] = {}
        for key in metric_keys:
            logs[mode][key] = []
            running_metrics[mode][key] = []

    with open(log_file_path, "r") as f:
        for line in f:
            data = read_log(line)
            if (data):
                _type = data[TYPE]
                if (_type == CONFIG):
                    logs[_type].append(data)
                elif(_type==METADATA):
                    logs[METADATA].append(data)
                else:
                    if(not(BATCH_INDEX in data)):
                        epoch_index = data[EPOCH_INDEX]
                        mode = data[MODE]
                        for key in metric_keys:
                            if key in data:
                                logs[mode][key].append(data[key])
                    # epoch_index = data[EPOCH_INDEX]
                    # batch_index = data[BATCH_INDEX]
                    # mode = data[MODE]
                    # if (batch_index == 0 and epoch_index > 0):
                        # new epoch
                        # for key in metric_keys:
                        #     if(key==TIME_TAKEN):
                        #         logs[mode][key].append(sum(running_metrics[mode][key]))
                        #     else:
                        #         logs[mode][key].append(np.mean(np.asarray(running_metrics[mode][key])))
                        #     running_metrics[mode][key] = []
                    # for key in metric_keys:
                    #     running_metrics[mode][key].append(data[key])
    logs = _transform_logs(logs)
    return logs

def _transform_logs(logs):
    keys_to_transform = set([TRAIN, VAL, TEST])
    for key in logs:
        if(key in keys_to_transform):
            metric_dict = {}
            for metric in logs[key]:
                if(logs[key][metric]):
                    metric_dict[metric] = np.asarray(logs[key][metric])
            logs[key] = metric_dict
    return logs

def get_config_from_appid(app_id):
    config = get_config(read_cmd_args=False)
    log_file_path = config[LOG][FILE_PATH]
    logs_dir = "/".join(log_file_path.split("log.txt")[0].split("/")[:-2])
    log_file_path = Path(logs_dir, app_id, "log.txt")
    logs = parse_log_file(log_file_path)
    return logs[CONFIG][0]

def write_sequences(true_inp, true_outp, pred_outp, mode, epoch=0, exp_name='', remove_prev=True, conf=None, classes=None, test_fl=''):
    """
    Write sequences to csv
    :param true_inp:
    :param true_outp:
    :param pred_outp:
    :param remove_prev: if True, remove previous epoch file and only keep the current one
    :return:
    """
    base_path = os.path.dirname(os.path.realpath(__file__)).split('/codes')[0]
    logs_folder = str(base_path) + '/logs/'
    test_fl = test_fl.split('/')[-1]
    filename = logs_folder + '{}_{}_{}_epoch_{}.csv'.format(exp_name, mode, test_fl, epoch)

    if remove_prev and epoch > 0:
        prev_filename = logs_folder + '{}_{}_{}_epoch_{}.csv'.format(exp_name, mode, test_fl, epoch - 1)
        try:
            os.remove(prev_filename)
        except OSError:
            pass
    confs = [','.join([str(ci) for ci in c]) for c in conf]
    classes = [classes for d in true_inp]
    df = pd.DataFrame({'true_inp':true_inp, 'true_outp':true_outp, 'pred_outp': pred_outp, 'conf': confs, 'target_class': classes})
    # Save in experiment folder
    df.to_csv(filename, index=False)

class FakeExperiment():
    """
    Fake logger to disable comet
    """
    def __init__(self, *args, **kwargs):
        pass

    def log_parameters(self, *args, **kwargs):
        pass

    def set_name(self, *args, **kwargs):
        pass

    @contextmanager
    def train(self, cr):
        pass

    @contextmanager
    def validate(self, cr):
        pass

    @contextmanager
    def test(self, *args, **kwargs):
        pass

    def log_metric(self, *args, **kwargs):
        pass

    def log_dataset_info(self, *args, **kwargs):
        pass





