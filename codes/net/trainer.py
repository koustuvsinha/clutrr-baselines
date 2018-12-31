# Trainer class to get the loss and predictions
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from addict import Dict
from codes.net.batch import Batch

class Trainer:
    def __init__(self, model_config, encoder_model, decoder_model,
                 max_entity_id=0):
        """
        Generic trainer for an encoder-decoder model
        :param encoder_model:
        :param decoder_model:
        :param max_entity_id: max entity id
        """
        self.model_config = model_config
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model
        loss_criteria = model_config.loss_criteria
        if loss_criteria == 'CE':
            self.criteria = nn.CrossEntropyLoss()
        elif loss_criteria == 'NLL':
            self.criteria = nn.NLLLoss()
        else:
            raise NotImplementedError("Provided loss criteria not implemented")
        self.tf_ratio = model_config.tf_ratio
        self.max_entity_id = max_entity_id
        self.encoder_model.max_entity_id = max_entity_id
        self.decoder_model.max_entity_id = max_entity_id

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
        batch_size  = batch.batch_size
        loss = 0
        if self.model_config.encoder.invalidate_embeddings:
            # invalidate the entity embeddings
            self.encoder_model.invalidate_embeddings()
        if self.model_config.decoder.invalidate_embeddings:
            self.decoder_model.invalidate_embeddings()
        # run the encoder
        encoder_outputs, encoder_hidden = self.encoder_model(batch)
        batch.encoder_outputs = encoder_outputs
        batch.encoder_hidden = encoder_hidden
        batch.encoder_model = self.encoder_model
        # calculate the hidden state of decoder
        hidden_rep = self.decoder_model.init_hidden(encoder_outputs, batch_size)
        #hidden_rep[0] = self.expand_hidden(hidden_rep[0])
        #hidden_rep[1] = self.expand_hidden(hidden_rep[1])
        query_rep = self.decoder_model.calculate_query(batch)  # query representation or question representation
        decoder_inp = None
        decoder_outp = None

        if self.model_config.loss_type == 'seq2seq':
            teacher_forcing = False

            max_outp_length = max([w for bt in batch.outp_lengths for w in bt])
            batch.outp = batch.outp.view(-1, max_outp_length)

            for step in range(max_outp_length - 1):
                if mode == 'train':
                    teacher_forcing = np.random.choice([True, False],p=[self.tf_ratio, 1-self.tf_ratio])
                if teacher_forcing or step == 0:
                    decoder_inp = batch.outp[:,step].unsqueeze(1)
                else:
                    decoder_inp = decoder_inp

                step_batch = Dict()
                step_batch.decoder_inp = decoder_inp
                step_batch.hidden_rep = hidden_rep
                step_batch.query_rep = query_rep

                logits, attn, hidden_rep = self.decoder_model(batch, step_batch)
                topv, topi = logits.data.topk(1)
                next_words = topi.squeeze(1)
                loss += self.criteria(logits.squeeze(1), batch.outp[:, step+1])
                if step == 0:
                    decoder_outp = next_words
                else:
                    decoder_outp = torch.cat((decoder_outp, next_words), 1)
                decoder_inp = next_words

        elif self.model_config.loss_type == 'classify':
            # batch.outp should be B x 1
            step_batch = Dict()
            step_batch.decoder_inp = decoder_inp
            step_batch.hidden_rep = hidden_rep
            step_batch.query_rep = query_rep
            logits, attn, hidden_rep = self.decoder_model(batch, step_batch)
            if logits.dim() > 2:
                logits = logits.squeeze(1)
            loss = self.criteria(logits, batch.target.squeeze(1))
            # generate prediction
            topv, topi = logits.data.topk(1)
            next_words = topi.squeeze(1)
            decoder_outp = next_words
        else:
            raise NotImplementedError("Value in config.model.loss_type not implemented.")


        return decoder_outp, loss

    def train(self):
        self.encoder_model.train()
        self.decoder_model.train()

    def eval(self):
        self.encoder_model.eval()
        self.decoder_model.eval()

    def expand_hidden(self, hidden_rep, max_abs=5):
        return hidden_rep.unsqueeze(2).expand(-1, -1, max_abs, -1).contiguous()\
            .view(hidden_rep.size(0), -1, hidden_rep.size(-1))

















