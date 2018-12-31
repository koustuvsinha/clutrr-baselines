## Experiment utility class to track, save and reproduce experiments
from addict import Dict
import os
import torch
import shutil
import logging

class Experiment:
    def __init__(self):
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
        self.config = None
        self.comet_exp = None
        parent_dir = os.path.abspath(os.pardir).split('/codes')[0]
        self.model_save_path = os.path.join(parent_dir, 'model')
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

    def save_checkpoint(self, is_best=False):
        """
        Save all the stuff for rerunning again
        :return:
        """
        model_pre = self.config.general.id
        filename = os.path.join(self.model_save_path,
                         '{}_checkpoint.pt'.format(model_pre))
        best_filename = os.path.join(self.model_save_path,
                         '{}_best.pt'.format(model_pre))
        # save the model
        torch.save({
            'epoch': self.epoch_index,
            'config': self.config,
            'model.encoder': self.model.encoder.state_dict(),
            'model.decoder': self.model.decoder.state_dict(),
            'optimizer': [opt.state_dict() for opt in self.optimizers]
        }, filename)

        if is_best:
            shutil.copyfile(filename, best_filename)

    def load_checkpoint(self):
        model_pre = self.config.general.id
        checkpoint_name = '{}_checkpoint.pt'.format(model_pre)
        checkpoint_path = os.path.join(self.model_save_path,
                                       checkpoint_name)
        if os.path.isfile(checkpoint_path):
            logging.info("loading checkpoint {}".format(checkpoint_name))
            checkpoint = torch.load(checkpoint_path)
            self.model.encoder.load_state_dict(checkpoint['model.encoder'])
            self.model.decoder.load_state_dict(checkpoint['model.decoder'])
            for idx, opt in enumerate(checkpoint['optimizer']):
                self.optimizers[idx].load_state_dict(opt)





