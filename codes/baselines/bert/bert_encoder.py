# BERT Encoder
# Devlin, Jacob, et al. "Bert: Pre-training of deep bidirectional transformers for language understanding." arXiv preprint arXiv:1810.04805 (2018).
# Use HuggingFace pytorch


import torch
import torch.nn as nn
from codes.net.base_net import Net
import pdb
import numpy as np
from pytorch_pretrained_bert import BertModel
from codes.baselines.lstm.basic import SimpleEncoder


class BERTEncoder(Net):
    """
    Bert with fixed encoding scheme
    """
    def __init__(self, model_config, shared_embeddings=None):
        super().__init__(model_config)

        if not shared_embeddings:
            self.init_embeddings()
        else:
            self.embedding = shared_embeddings

        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.model.eval()

    def forward(self, batch):
        out = batch.inp
        with torch.no_grad():
            out, _ = self.model(out, batch.bert_segment_ids, batch.bert_input_mask, output_all_encoded_layers=False)
        return out, None

class BERTLSTMEncoder(Net):
    """
    BERT with LSTM on top of it.
    Get the output from BERT, replace the embedding with our embedding value, and run lstm on top of it.
    Essentially, treat BERT as an embedding layer.
    """

    def __init__(self, model_config, shared_embeddings=None):
        super().__init__(model_config)

        if not shared_embeddings:
            self.init_embeddings()
        else:
            self.embedding = shared_embeddings

        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.model.eval()
        self.lstm_encoder = SimpleEncoder(model_config, use_embedding=False, shared_embeddings=self.embedding)

    def forward(self, batch):
        out = batch.inp
        # pdb.set_trace()
        with torch.no_grad():
            out, _ = self.model(out, batch.bert_segment_ids, batch.bert_input_mask, output_all_encoded_layers=False)
        entity_mask = batch.inp_ent_mask.byte()
        entity_emb = self.embedding(batch.bert_inp)
        # replace entity_emb with out in entity mask positions
        entity_emb = entity_emb * entity_mask.unsqueeze(2).float()
        out = out * (~entity_mask).unsqueeze(2).float()
        out = out + entity_emb
        mbatch = batch.clone()
        mbatch.inp = out
        out, _ = self.lstm_encoder(mbatch)

        return out, None
