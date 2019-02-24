## Experiment utility class to track, save and reproduce experiments
from addict import Dict
import os
import torch
import shutil
import logging

class Experiment:
    def __init__(self, config):
        self.dataloaders = Dict()
        self.data_util = None
        self.model = Dict()
        self.trainer = None
        self.optimizers = None
        self.schedulers = None
        # device: gpu or cpu
        self.device = None
        self.validation_metrics = None
        self.quality_metrics = None
        self.metric_to_perform_early_stopping = None
        self.generator = None
        self.epoch_index = 0
        self.iteration_index = Dict()
        self.config = config
        self.comet_exp = None
        self.model_save_path = os.path.join(config.general.base_path, 'model')
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

    def save_checkpoint(self, is_best=False):
        """
        Save all the stuff for rerunning again
        :return:
        """
        model_pre = self.config.general.id
        experiment_id = self.comet_ml.id
        filename = os.path.join(self.model_save_path,
                         '{}_{}_checkpoint.pt'.format(model_pre, experiment_id))
        best_filename = os.path.join(self.model_save_path,
                         '{}_{}_best.pt'.format(model_pre, experiment_id))
        # save the model
        torch.save({
            'epoch_index': self.epoch_index,
            'iteration_index': self.iteration_index,
            'config': self.config,
            'experiment_id': experiment_id,
            'model.encoder': self.model.encoder.state_dict(),
            'model.decoder': self.model.decoder.state_dict(),
            'optimizer': [opt.state_dict() for opt in self.optimizers]
        }, filename)

        if is_best:
            shutil.copyfile(filename, best_filename)

    def load_checkpoint(self, exp_id):
        model_pre = self.config.general.id
        checkpoint_name = '{}_{}_checkpoint.pt'.format(model_pre, exp_id)
        checkpoint_path = os.path.join(self.model_save_path,
                                       checkpoint_name)
        if os.path.isfile(checkpoint_path):
            logging.info("loading checkpoint {}".format(checkpoint_name))
            checkpoint = torch.load(checkpoint_path)
            self.model.encoder.load_state_dict(checkpoint['model.encoder'])
            self.model.decoder.load_state_dict(checkpoint['model.decoder'])
            for idx, opt in enumerate(checkpoint['optimizer']):
                self.optimizers[idx].load_state_dict(opt)
            self.epoch_index = checkpoint['epoch_index']
            self.iteration_index = checkpoint['iteration_index']
            self.config = checkpoint['config']
        else:
            raise FileNotFoundError("Checkpoint not found")





