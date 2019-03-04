import os
import random
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pdb

class Net(nn.Module):
    '''
    Base class for all models
    '''

    def __init__(self, model_config):
        super(Net, self).__init__()
        self.model_config = model_config
        self.name = "base_model"
        self.description = "This is the base class for all the models. All the other models should " \
                           "extend this class. It is not to be used directly."

        self.embedding = None
        #self.init_embeddings()

        self.criteria = nn.CrossEntropyLoss()

        self.dropout = nn.Dropout(self.model_config.dropout_probability)

        ## num directions is a flag to multiply with hidden dims
        self.num_directions = 2 if model_config.decoder.bidirectional else 1

        ## to be set by trainer
        self.max_entity_id = 0
        self.random_weights = None

        # flag for graph mode
        self.graph_mode = False
        self.one_hot = False

    def loss(self, outputs, labels):
        '''
        Method to perform the loss computation
        :param outputs:
        :param labels:
        :return:
        '''
        return self.criteria(outputs, labels)

    def track_loss(self, outputs, labels):
        # There are two different functions as we might be interested in tracking one loss and
        # optimising another
        return self.loss(outputs, labels)

    def save_model(self, epochs=-1, optimizers=None, is_best_model=False):
        '''
        Method to persist the net
        '''
        # return
        state = {
            "epochs": epochs + 1,
            "state_dict": self.state_dict(),
            "optimizers": [optimizer.state_dict() for optimizer in optimizers],
            "np_random_state": np.random.get_state(),
            "python_random_state": random.getstate(),
            "pytorch_random_state": torch.get_rng_state()
        }
        if is_best_model:
            path = os.path.join(self.model_config["save_dir"],
                                "best_model.tar")
        else:
            path = os.path.join(self.model_config["save_dir"],
                                "model_epoch_" + str(epochs + 1) + "_timestamp_" + str(int(time())) + ".tar")
        torch.save(state, path)
        print("saved net to path = {}".format(path))

    def load_model(self, optimizers):
        path = self.model_config.load_path
        print("Loading net from path {}".format(path))
        # checkpoint = torch.load(path)
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        epochs = checkpoint["epochs"]
        self._load_metadata(checkpoint)
        self._load_model_params(checkpoint["state_dict"])
        for optim_index, optimizer in enumerate(optimizers):
            # optimizer.load_state_dict(checkpoint[OPTIMIZERS][optim_index]())
            optimizer.load_state_dict(checkpoint["optimizers"][optim_index])
        return optimizers, epochs

    def _load_metadata(self, checkpoint):
        np.random.set_state(checkpoint["np_random_state"])
        random.setstate(checkpoint["python_random_state"])
        torch.set_rng_state(checkpoint["pytorch_random_state"])

    def _load_model_params(self, state_dict):
        self.load_state_dict(state_dict)

    def get_model_params(self):
        model_parameters = list(filter(lambda p: p.requires_grad, self.parameters()))
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("Total number of params = " + str(params))
        return model_parameters

    def get_optimizers(self):
        '''Method to return the list of optimizers for the net'''
        optimizers = []

        model_params = self.get_model_params()
        if (model_params):
            if (self.model_config.optimiser.name == "adam"):
                optimizers.append(optim.Adam(model_params,
                                             lr=self.model_config.optimiser.learning_rate,
                                             weight_decay=self.model_config.optimiser.l2_penalty
                                             ))
        if optimizers:
            if (self.model_config.optimiser.scheduler_type == "exp"):
                schedulers = list(map(lambda optimizer: optim.lr_scheduler.ExponentialLR(
                    optimizer=optimizer, gamma=self.model_config.optimiser.scheduler_gamma), optimizers))
            elif (self.model_config.optimiser.scheduler_type == "plateau"):
                schedulers = list(map(lambda optimizer: optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer=optimizer, mode="max", patience=self.model_config.optimiser.scheduler_patience,
                    factor=self.model_config.optimiser.scheduler_gamma, verbose=True), optimizers))

            return optimizers, schedulers
        return None

    def get_mlp(self, input_dim, output_dim, num_layers=2, dropout=0.0):
        network_list = []
        assert num_layers > 0
        if num_layers > 1:
            for _ in range(num_layers-1):
                network_list.append(nn.Linear(input_dim, input_dim))
                network_list.append(nn.ReLU())
                network_list.append(nn.Dropout(dropout))
        network_list.append(nn.Linear(input_dim, output_dim))
        return nn.Sequential(
            *network_list
        )

    def get_mlp_h(self, input_dim, output_dim, num_layers=2, dropout=0.0):
        network_list = []
        assert num_layers > 0
        network_list.append(nn.Linear(input_dim, output_dim))
        if num_layers > 1:
            for _ in range(num_layers - 1):
                network_list.append(nn.Linear(output_dim, output_dim))
                network_list.append(nn.ReLU())
                network_list.append(nn.BatchNorm1d(num_features=output_dim))
                network_list.append(nn.Dropout(dropout))
        else:
            network_list.append(nn.ReLU())
            network_list.append(nn.BatchNorm1d(num_features=output_dim))
            network_list.append(nn.Dropout(dropout))
        return nn.Sequential(
            *network_list
        )

    def forward(self, data):
        '''
        Forward pass of the network
        :param data: batch of the edges to train on
        :return:
        '''

        pass

    def get_param_count(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        return sum([np.prod(p.size()) for p in model_parameters])

    def init_embeddings(self):
        """
        Initialize embeddings
        :return:
        """
        self.embedding = nn.Embedding(num_embeddings=self.model_config.vocab_size,
                                      embedding_dim=self.model_config.embedding.dim,
                                      padding_idx=0,
                                      max_norm=1)
        if (self.model_config.embedding.should_use_pretrained_embedding):
            self.load_embeddings()
        else:
            torch.nn.init.xavier_uniform_(self.embedding.weight)
        self.embedding.weight.data[self.embedding.padding_idx].fill_(0)

    def load_embeddings(self):
        """
        Optional : Load pretrained embeddings for the words
        :return:
        """
        pre_emb = torch.load(self.model_config.model.embedding.pretrained_embedding_path)
        if pre_emb.size(1) != self.embedding.weight.size(1):
            raise RuntimeError('Embedding dimensions mismatch for pretrained embedding')
        self.embedding.weight.data = pre_emb

    def freeze_embeddings(self):
        """
        Only use this when loading embeddings
        :return:
        """
        self.embedding.requires_grad = False

    def get_entity_mask(self, ids, entity_id=0, mode='max'):
        """
        Get entity mask 1-0
        :param ids: 1D or 2D matrix of word_ids
        :param entity_id: int
        :param mode: if max, then mask till the id,
            if equal then only mask for the id
        :return: same shape as ids
        """
        if mode == 'max':
            mask = ids > entity_id
        else:
            mask = ids == entity_id
        mask = mask.to(self.embedding.weight.device)
        return mask.float()

    def randomize_entity_embeddings(self, fixed=False, padding=True):
        """
        Randomize the entity embeddings.
        At each epoch, randomize the entity embeddings
        :param fixed: if True, then re-use the old random weights
        :return:
        """
        if self.one_hot:
            if not self.graph_mode:
                raise NotImplementedError("one hot mode only for graph")
        with torch.no_grad():
            vocab_size = self.embedding.weight.size(0)
            if self.one_hot:
                random_weights = torch.eye(vocab_size).to(self.embedding.weight.device)
            elif fixed:
                random_weights = self.random_weights
            else:
                random_weights = torch.nn.init.xavier_uniform_(torch.zeros(
                    self.embedding.weight.size()).to(self.embedding.weight.device))
            if not self.graph_mode:
                ids = torch.arange(0, vocab_size)
                assert self.max_entity_id > 0
                mask = self.get_entity_mask(ids, self.max_entity_id)
                mask = mask.unsqueeze(1)
                entity_mask = (1 - mask)
                # check for padding
                if padding:
                    entity_mask[0].fill_(0.0)
                # randomly assign the weights
                idx = torch.randperm(random_weights.nelement()).to(random_weights.device)
                random_weights = random_weights.view(-1)[idx].view(random_weights.size())
                self.embedding.weight.mul_(mask)
                self.embedding.weight.add_(random_weights.mul_(entity_mask))
            else:
                idx = torch.randperm(random_weights.nelement()).to(random_weights.device)
                random_weights = random_weights.view(-1)[idx].view(random_weights.size())
                self.embedding.weight = nn.Parameter(random_weights)
            self.random_weights = random_weights.clone().detach()


