# BERT Encoder
# Devlin, Jacob, et al. "Bert: Pre-training of deep bidirectional transformers for language understanding." arXiv preprint arXiv:1810.04805 (2018).
# Use HuggingFace pytorch


import torch
import torch.nn as nn
from codes.net.base_net import Net
import pdb
import numpy as np
from pytorch_pretrained_bert.modeling import BertForSequenceClassification


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

    def forward(self, batch):
        out = batch.bert_inp
        # fix for special tokens
        return out, None
