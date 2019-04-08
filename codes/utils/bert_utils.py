# Run BERT-as-a-service on all data, and store them for later processing

import hashlib
import torch
from bert_serving.client import BertClient
import os
import logging
from sacremoses import MosesDetokenizer
import pdb

class BertLocalCache():
    '''Class to provide a local client for interfacing with BERT'''
    def __init__(self, config):
        self.hash_to_idx_map = {}
        self.embeddings = []
        self.model_config = config.model
        self.bert_client = BertClient(
            ip=self.model_config.bert.ip,
            port=self.model_config.bert.port,
            port_out=self.model_config.bert.port_out)
        self.sentences = []

    def hash_fn(self, string_to_hash):
        '''Simple hash function'''
        md = MosesDetokenizer()
        if type(string_to_hash) == list:
            string_to_hash = md.detokenize(string_to_hash)
        return hashlib.sha224(string_to_hash.encode('utf-8')).hexdigest()

    def update_cache(self, list_of_sentences):
        for idx, sentence in enumerate(list_of_sentences):
            sentence_hash = self.hash_fn(sentence)
            self.hash_to_idx_map[sentence_hash] = len(self.sentences)
            self.sentences.append(sentence)

    def run_bert(self):
        pdb.set_trace()
        embedding = self.query_bert(self.sentences)
        self.embeddings = embedding

    def save_cache(self, path):
        self.embeddings = torch.tensor(self.embeddings)
        torch.save({'embeddings':self.embeddings, 'hashmap':self.hash_to_idx_map},
                   os.path.join(path, self.model_config.bert.embedding_file))

    def is_cache_present(self, path):
        return os.path.exists(os.path.join(path, self.model_config.bert.embedding_file))

    def load_cache(self, path):
        cache = torch.load(os.path.join(path, self.model_config.bert.embedding_file))
        self.embeddings = cache['embeddings']
        self.hash_to_idx_map = cache['hashmap']


    def query_bert(self, sentences):
            bert_out = self.bert_client.encode(sentences, is_tokenized=True)
            # removing CLS in the beginning and SEP at the end
            bert_out = bert_out[:, 1:-1, ]
            return bert_out

    def query(self, list_of_sentences):
        with torch.no_grad():
            embeddings = [None for _ in range(len(list_of_sentences))]
            for idx, sentence in enumerate(list_of_sentences):
                sentence_hash = self.hash_fn(sentence)
                embedding_idx = self.hash_to_idx_map[sentence_hash]
                embeddings[idx] = self.embeddings[embedding_idx].unsqueeze(0)
            return torch.cat(embeddings, dim=0)
