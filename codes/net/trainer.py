# Trainer class to get the loss and predictions
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from addict import Dict
from codes.net.batch import Batch
import pdb

class Trainer:
    def __init__(self, model_config, encoder_model, decoder_model,
                 max_entity_id=0):
        """
        Generic trainer for an encoder-decoder model
        Changes:
            - now bringing the embedding handling logic to trainer itself,
            - so we can have a clear demarcation of concerns
        :param encoder_model:
        :param decoder_model:
        :param max_entity_id: max entity id
        """
        self.model_config = model_config
        self.max_entity_id = max_entity_id
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model
        self.encoder_model.max_entity_id = max_entity_id
        self.decoder_model.max_entity_id = max_entity_id
        self.padding = True
        if model_config.name == 'graph':
            self.padding = False
        # initialize embeddings
        if self.model_config.embedding.entity_embedding_policy in ['random','fixed']:
            # randomize the entity embeddings first time
            self.encoder_model.randomize_entity_embeddings(padding=self.padding)
        else:
            print("Learning entity embeddings")
        loss_criteria = model_config.loss_criteria
        if loss_criteria == 'CE':
            self.criteria = nn.CrossEntropyLoss()
        elif loss_criteria == 'NLL':
            self.criteria = nn.NLLLoss()
        else:
            raise NotImplementedError("Provided loss criteria not implemented")
        self.tf_ratio = model_config.tf_ratio

    def get_optimizers(self):
        '''Method to return the list of optimizers for the trainer'''
        optimizers = []

        model_params_encoder = self.encoder_model.get_model_params()
        model_params_decoder = self.decoder_model.get_model_params()
        model_params = model_params_encoder + model_params_decoder

        if (model_params):
            if (self.model_config.optimiser.name == "adam"):
                optimizers.append(optim.Adam(model_params,
                                             lr=self.model_config.optimiser.learning_rate,
                                             weight_decay=self.model_config.optimiser.l2_penalty
                                             ))
            elif (self.model_config.optimiser.name == 'sgd'):
                optimizers.append(optim.SGD(model_params, lr=self.model_config.optimiser.learning_rate,
                                            weight_decay=self.model_config.optimiser.l2_penalty))
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

    def batchLoss(self, batch: Batch, mode='train'):
        """
        Run the Loss
        -> For classification experiments, use loss_type = `classify` in model.config
        -> For seq2seq experiments, use  loss_type = `seq2seq` in model.config
        :param batch - Batch class
        :param mode:
        :return:
        """
        # choose a policy of invalidating entity embeddings here
        if self.model_config.embedding.entity_embedding_policy == 'random':
            # randomize the entity embeddings at each epoch
            self.encoder_model.randomize_entity_embeddings(padding=self.padding)
        if self.model_config.embedding.entity_embedding_policy == 'fixed':
            # fix the random entity embeddings which were used before
            self.encoder_model.randomize_entity_embeddings(fixed=True, padding=self.padding)

        # run the encoder
        encoder_outputs, encoder_hidden = self.encoder_model(batch)
        batch.encoder_outputs = encoder_outputs
        batch.encoder_hidden = encoder_hidden
        batch.encoder_model = self.encoder_model

        query_rep = self.decoder_model.calculate_query(batch)  # query representation or question representation
        decoder_inp = None

        # batch.outp should be B x 1
        step_batch = Dict()
        step_batch.query_rep = query_rep
        logits, attn, hidden_rep = self.decoder_model(batch, step_batch)
        if logits.dim() > 2:
            logits = logits.squeeze(1)
        loss = self.criteria(logits, batch.target.squeeze(1))
        # generate prediction
        topv, topi = logits.data.topk(1)
        next_words = topi.squeeze(1)
        decoder_outp = next_words
        # confidence of classes
        conf = torch.exp(F.log_softmax(logits,dim=1))


        return decoder_outp, loss, conf

    def train(self):
        self.encoder_model.train()
        self.decoder_model.train()

    def eval(self):
        self.encoder_model.eval()
        self.decoder_model.eval()

    def expand_hidden(self, hidden_rep, max_abs=5):
        return hidden_rep.unsqueeze(2).expand(-1, -1, max_abs, -1).contiguous()\
            .view(hidden_rep.size(0), -1, hidden_rep.size(-1))

















