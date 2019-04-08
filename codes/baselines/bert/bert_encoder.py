# BERT Encoder
# Devlin, Jacob, et al. "Bert: Pre-training of deep bidirectional transformers for language understanding." arXiv preprint arXiv:1810.04805 (2018).
# Use HuggingFace pytorch


import torch
import torch.nn as nn
from codes.net.base_net import Net
import pdb
import numpy as np
from pytorch_pretrained_bert import BertModel


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
            layer_outs, _ = self.model(out, output_all_encoded_layers=True)
        out = layer_outs[-1]
        return out, None
