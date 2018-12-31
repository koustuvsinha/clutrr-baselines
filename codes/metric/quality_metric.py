from nltk.translate.bleu_score import sentence_bleu
import numpy as np
import os
import json

base_path = os.path.dirname(os.path.realpath(__file__)).split('codes')[0]

class QualityMetric:
    def __init__(self, data):
        self.data = data
        self.entity_ids = data.entity_ids
        self.entity_words = [data.id2word[e] for e in self.entity_ids]

    def entity_overlap(self, prediction, hypothesis):
        """
        Return a score based on how many entities we got correct
        :param prediction:
        :param hypothesis:
        :return: score between 0 and 1
        """
        com_n = []
        for idx, pred in enumerate(prediction):
            hyp_ents = set(self.entity_words).intersection(set(hypothesis[idx]))
            pred_ents = set(self.entity_words).intersection(set(pred))
            com_ent = hyp_ents.intersection(pred_ents)
            score = 0
            if len(hyp_ents) > 0:
                score = len(com_ent) / len(hyp_ents)
            com_n.append(score)
        return np.mean(com_n)

    def relation_overlap(self, prediction, hypothesis):
        """
        Calculate how many relations are correctly predicted
        :param prediction:
        :param hypothesis:
        :return: accuracy score
        """
        corr = []
        for idx, pred in enumerate(prediction):
            hyp_relation = set(hypothesis[idx])
            pred_relation = set(pred)
            corr_rel = hyp_relation.intersection(pred_relation)
            corr.append(len(corr_rel)/ len(hyp_relation))
        return np.mean(corr)

    def batch_bleu(self, prediction, hypothesis):
        """
        Return a bleu score for the entire batch
        :param prediction:
        :param hypothesis:
        :return:
        """
        b_n = []
        for idx, pred in enumerate(prediction):
            b_n.append(sentence_bleu([hypothesis[idx]], pred))
        return np.mean(b_n)


