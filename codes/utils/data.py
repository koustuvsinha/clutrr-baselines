# Data preparation and manipulation codes
import torch
import torch.utils.data as data
import re
import pandas as pd
import json
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
import pickle as pkl
import itertools as it
from addict import Dict
from codes.net.batch import Batch
from codes.utils.config import get_config
import os
import json
from ast import literal_eval as make_tuple
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

base_path = os.path.dirname(os.path.realpath(__file__)).split('codes')[0]
UNK_WORD = '<unk>'
PAD_TOKEN = '<pad>'
START_TOKEN = '<s>'
END_TOKEN = '</s>'

class DataRow():
    """
    Defines a single instance of data row
    """
    def __init__(self):
        self.id = None
        self.story = None
        self.story_sents = None # same story, but sentence tokenized
        self.query = None
        self.text_query = None
        self.target = None
        self.text_target = None
        self.story_graph = None


class DataUtility():
    """
    Data preparation and utility class
    """
    def __init__(self,
                 config,
                 num_workers=1,
                 common_dict=True):
        """

        :param main_file: file where the summarization resides
        :param train_test_split: default 0.8
        :param sentence_mode: if sentence_mode == True, then split story into sentence
        :param single_abs_line: if True, then output pair is single sentences of abs
        :param num_reads: number of reads for a sentence
        :param dim: dimension of edges
        """
        self.config = config
        # derive configurations
        self.train_test_split = config.dataset.train_test_split
        self.max_vocab = config.dataset.max_vocab
        self.tokenization = config.dataset.tokenization
        self.common_dict = config.dataset.common_dict
        self.batch_size = config.model.batch_size
        self.num_reads = config.model.graph.num_reads
        self.dim = config.model.graph.edge_dim
        self.sentence_mode = config.dataset.sentence_mode
        self.single_abs_line = config.dataset.single_abs_line

        self.word2id = {}
        self.id2word = {}
        self.target_word2id = {}
        self.target_id2word = {}
        # dict of dataRows
        # all rows are indexed by their key `id`
        self.dataRows = {'train':{}, 'test':{}}

        self.train_indices = []
        self.test_indices = []
        self.val_indices = []
        self.special_tokens = [PAD_TOKEN, UNK_WORD, START_TOKEN, END_TOKEN]
        self.main_file = ''
        self.common_dict = common_dict
        self.num_workers = num_workers
        # keep some part of the vocab fixed for the entities
        # for that we need to first calculate the max number of unique entities *per row*
        self.train_data = None
        self.test_data = None
        self.train_file = ''
        self.test_file = ''
        self.max_ents = 0
        self.entity_ids = []
        self.entity_map = {} # map entity for each puzzle
        self.max_entity_id = 0
        self.adj_graph = []
        self.dummy_entity = '' # return this entity when UNK entity
        self.load_dictionary = config.dataset.load_dictionary
        self.max_sent_length = 0
        # check_data flags
        self.data_has_query = False
        self.data_has_text_query = False
        self.data_has_target = False
        self.data_has_text_target = False
        self.preprocessed = set() # set of puzzle ids which has been preprocessed

    def process_data(self, base_path, train_file, load_dictionary=True, preprocess=True):
        """
        Load data and run preprocessing scripts
        :param main_file .csv file of the data
        :return:
        """
        self.train_file = train_file
        train_data = pd.read_csv(self.train_file, comment='#')
        train_data = self._check_data(train_data)
        logging.info("Start preprocessing data")
        if load_dictionary:
            dictionary_file = os.path.join(base_path, 'dict.json')
            logging.info("Loading dictionary from {}".format(dictionary_file))
            dictionary = json.load(open(dictionary_file))
            # fix id2word keys
            dictionary['id2word'] = {int(k):v for k,v in dictionary['id2word'].items()}
            dictionary['target_id2word'] = {int(k): v for k, v in dictionary['target_id2word'].items()}
            for key, value in dictionary.items():
                setattr(self, key, value)
        train_data, max_ents_train, = self.process_entities(train_data)
        if preprocess:
            if not load_dictionary:
                self.max_ents = max_ents_train + 1 # keep an extra dummy entity for unknown entities
            self.preprocess(train_data, mode='train')
            self.train_data = train_data
            self.split_indices()
        else:
            return train_data, max_ents_train

    def process_test_data(self, base_path, test_files):
        """
        Load testing data
        :param test_files: array of file names
        :return:
        """
        self.test_files = test_files #[os.path.join(base_path, t) + '_test.csv' for t in test_files]
        test_datas = [pd.read_csv(tf, comment='#') for tf in self.test_files]
        for test_data in test_datas:
            self._check_data(test_data)
        logging.info("Loaded test data, starting preprocessing")
        p_tests = []
        for ti, test_data in enumerate(test_datas):
            test_data, max_ents_test, = self.process_entities(test_data)
            self.preprocess(test_data, mode='test',
                            test_file=test_files[ti])
            p_tests.append(test_data)
        self.test_data = p_tests
        logging.info("Done preprocessing test data")


    def _check_data(self, data):
        """
        Check if the file has correct headers.
        For all the subsequent experiments, make sure that the dataset generated
        or curated has the following fields:
        - id : unique uuid for each puzzle          : required
        - story : input text                        : required
        - query : query entities                    : optional
        - text_query : the question for QA models   : optional
        - target : classification target            : required if config.model.loss_type set to classify
        - text_target : seq2seq target              : required if config.model.loss_type set to seq2seq
        :param data:
        :return: data
        """
        # check for required stuff
        assert "id" in list(data.columns)
        assert "story" in list(data.columns)
        if self.config.model.loss_type == 'classify':
            assert "target" in list(data.columns)
        if self.config.model.loss_type == 'seq2seq':
            assert "text_target" in list(data.columns)
        # turn on flag if present
        if "target" in list(data.columns):
            self.data_has_target = True
        if "text_target" in list(data.columns):
            self.data_has_text_target = True
        if "query" in list(data.columns) and len(data['query'].value_counts()) > 0:
            self.data_has_query = True
        else:
            data['query'] = ''
        if "text_query" in list(data.columns) and len(data['text_query'].value_counts()) > 0:
            self.data_has_text_query = True
        else:
            data['text_query'] = ''
        return data

    def process_entities(self, data, placeholder='[]'):
        """
        extract entities and replace them with placeholders.
        Also maintain a per-puzzle mapping of entities
        :param placeholder: if [] then simply use regex to extract entities as they are already in
        a placeholder. If None, then use Spacy EntityTokenizer
        :return: max number of entities in dataset
        """
        max_ents = 0
        if placeholder == '[]':
            for i,row in data.iterrows():
                story = row['story']
                ents = re.findall('\[(.*?)\]', story)
                uniq_ents = set(ents)
                if len(uniq_ents) > max_ents:
                    max_ents = len(uniq_ents)
                pid = row['id']
                query = row['query'] if self.data_has_query else ''
                query = list(make_tuple(query))
                text_query = row['text_query'] if self.data_has_text_query else ''
                text_target = row['text_target'] if self.data_has_text_target else ''
                entity_map = {}
                for idx, ent in enumerate(uniq_ents):
                    entity_map[ent] = '@ent{}'.format(idx)
                    story = story.replace('[{}]'.format(ent), entity_map[ent])
                    text_target = text_target.replace('[{}]'.format(ent), entity_map[ent])
                    text_query = text_query.replace('[{}]'.format(ent), entity_map[ent])
                    try:
                        ent_index = query.index(ent)
                        query[ent_index] = entity_map[ent]
                    except ValueError:
                        pass
                data.at[i, 'story'] = story
                data.at[i, 'text_target'] = text_target
                data.at[i, 'text_query'] = text_query
                data.at[i, 'query'] = tuple(query)
                data.at[i, 'entities'] = json.dumps(list(uniq_ents))
                self.entity_map[pid] = entity_map
        else:
            raise NotImplementedError("Not implemented, should replace with a tokenization policy")
        return data, max_ents

    def preprocess(self, data, mode='train', single_abs_line=True, test_file=''):
        """
        Usual preprocessing: tokenization, lowercase, and create word dictionaries
        Also, split stories into sentences
        :param single_abs_line: if True, separate the abstracts into its corresponding lines
        and add each story-abstract pairs
        N.B. change: dropping `common_dict=True` as I am assuming I will always use a common
        dictionary for reasoning and QA. Separate dictionary makes sense for translation which
        I am not working at the moment.
        :return:
        """

        words = Counter()
        max_sent_length = 0
        if self.data_has_target:
            # assign target ids
            self.assign_target_id(list(data['target']))

        for i,row in data.iterrows():
            dataRow = DataRow()
            dataRow.id = row['id']
            story_sents = sent_tokenize(row['story'])
            story_sents = [self.tokenize(sent) for sent in story_sents]
            words.update([word for sent in story_sents for word in sent])
            dataRow.story_sents = story_sents
            dataRow.story = [word for sent in story_sents for word in sent] # flatten
            if self.data_has_text_target:
                # preprocess text_target
                text_target = self.tokenize(row['text_target'])
                dataRow.text_target = text_target
                words.update([word for word in text_target])
            if self.data_has_text_query:
                # preprocess text_query
                text_query = self.tokenize(row['text_query'])
                dataRow.text_query = text_query
                words.update([word for word in text_query])
            max_sl = max([len(s) for s in story_sents])
            if max_sl > max_sent_length:
                max_sent_length = max_sl
            if self.data_has_query:
                dataRow.query = row['query']
            if self.data_has_target:
                dataRow.target = self.target_word2id[row['target']]
            if mode == 'train':
                self.dataRows[mode][dataRow.id] = dataRow
            else:
                if test_file not in self.dataRows[mode]:
                    self.dataRows[mode][test_file] = {}
                self.dataRows[mode][test_file][dataRow.id] = dataRow
            self.preprocessed.add(dataRow.id)

        # only assign word-ids in train data
        if mode == 'train' and not self.load_dictionary:
            self.assign_wordids(words)

        # get adj graph
        ct = 0
        if mode == 'train':
            for i, row in data.iterrows():
                dR = self.dataRows[mode][row['id']]
                dR.story_graph = self.prepare_ent_graph(dR.story_sents)
                ct += 1
            logging.info("Processed {} stories in mode {}".format(ct,
                                                                  mode))
            self.max_sent_length = max_sent_length
        else:
            for i,row in data.iterrows():
                dR = self.dataRows[mode][test_file][row['id']]
                dR.story_graph = self.prepare_ent_graph(dR.story_sents)
                ct +=1
            logging.info("Processed {} stories in mode {} and file: {}".format(
                ct, mode, test_file))



    def tokenize(self, sent):
        """
        tokenize sentence based on mode
        :sent - sentence
        :param mode: word/char
        :return: splitted array
        """
        words = []
        if self.tokenization == 'word':
            words = word_tokenize(sent)
        if self.tokenization == 'char':
            words = sent.split('')
        # correct for tokenizing @entity
        corr_w = []
        tmp_w = ''
        for i,w in enumerate(words):
            if w == '@':
                tmp_w = w
            else:
                tmp_w += w
                corr_w.append(tmp_w)
                tmp_w = ''
        return corr_w

    def _insert_wordid(self, token, id):
        if token not in self.word2id:
            assert id not in set([v for k,v in self.word2id.items()])
            self.word2id[token] = id
            self.id2word[id] = token

    def assign_wordids(self, words, special_tokens=None):
        """
        Given a set of words, create word2id and id2word
        :param words: set of words
        :param special_tokens: set of special tokens to add into dictionary
        :return:
        """
        count = 0
        if not special_tokens:
            special_tokens = self.special_tokens
        ## if max_vocab is not -1, then shrink the word size
        if self.max_vocab >= 0:
            words = [tup[0] for tup in words.most_common(self.max_vocab)]
        else:
            words = list(words.keys())
        # add pad token
        self._insert_wordid(PAD_TOKEN, count)
        count +=1
        # reserve a block for entities. Record this block for future use.
        start_ent_num = count
        for idx in range(self.max_ents - 1):
            self._insert_wordid('@ent{}'.format(idx), count)
            count +=1
        # reserve a dummy entity
        self.dummy_entity = '@ent{}'.format(self.max_ents - 1)
        self._insert_wordid(self.dummy_entity, count)
        count += 1
        end_ent_num = count
        self.max_entity_id = end_ent_num - 1
        self.entity_ids = list(range(start_ent_num, end_ent_num))
        # add other special tokens
        if special_tokens:
            for tok in special_tokens:
                if tok == PAD_TOKEN:
                    continue
                else:
                    self._insert_wordid(tok, count)
                    count += 1
        # finally add the words
        for word in words:
            if word not in self.word2id:
                self._insert_wordid(word, count)
                count += 1

        logging.info("Modified dictionary. Words : {}, Entities : {}".format(
            len(self.word2id), len(self.entity_ids)))

    def assign_target_id(self, targets):
        """
        Assign IDS to targets
        :param targets:
        :return:
        """
        for target in set(targets):
            if target not in self.target_word2id:
                last_id = len(self.target_word2id)
                self.target_word2id[target] = last_id
        self.target_id2word = {v: k for k, v in self.target_word2id.items()}
        logging.info("Target Entities : {}".format(len(self.target_word2id)))

    def split_indices(self):
        """
        Split training file indices into training and validation
        Now we use separate testing file
        :return:
        """
        indices = list(self.dataRows['train'].keys())
        mask_i = np.random.choice(indices, int(len(indices) * self.train_test_split), replace=False)
        self.val_indices = [self.dataRows['train'][i].id for i in indices if i not in set(mask_i)]
        self.train_indices = [self.dataRows['train'][i].id for i in indices if i in set(mask_i)]


    def prepare_ent_graph(self, sents, max_nodes=0):
        """
        Given a list of sentences, return an adjacency matrix between entities
        Assumes entities have the format @ent{num}
        We can use OpenIE in later editions to automatically detect entities
        :param sents: list(list(str))
        :param max_nodes: max number of nodes in the adjacency matrix, int
        :return: list(list(int))
        """
        if max_nodes == 0:
            max_nodes = len(self.entity_ids)
        adj_mat = np.zeros((max_nodes, max_nodes))
        for sent in sents:
            ents = list(set([w for w in sent if '@ent' in w]))
            if len(ents) > 1:
                for ent1, ent2 in it.combinations(ents, 2):
                    ent1_id = self.get_entity_id(ent1) - 1
                    ent2_id = self.get_entity_id(ent2) - 1
                    adj_mat[ent1_id][ent2_id] = 1
                    adj_mat[ent2_id][ent1_id] = 1
        return adj_mat

    def get_dataloader(self, mode='train', test_file=''):
        """
        Return a new SequenceDataLoader instance with appropriate rows
        :param mode: train/val/test
        :return: SequenceDataLoader object
        """
        if mode != 'test':
            if mode == 'train':
                indices = self.train_indices
            else:
                indices = self.val_indices
            dataRows = self._select(self.dataRows['train'], indices)
        else:
            dataRows = [v for k,v in self.dataRows['test'][test_file].items()]

        logging.info("Total rows : {}, batches : {}"
                     .format(len(dataRows),
                             len(dataRows) // self.batch_size))

        collate_FN = collate_fn
        if self.sentence_mode:
            collate_FN = sent_collate_fn

        return data.DataLoader(SequenceDataLoader(dataRows, self),
                               batch_size=self.batch_size,
                               num_workers=self.num_workers,
                               collate_fn=collate_FN)

    def map_text_to_id(self, text):
        if isinstance(text, list):
            return list(map(self.get_token, text))
        else:
            return self.get_token(text)

    def get_token(self, word, target=False):
        if target and word in self.target_word2id:
            return self.target_word2id[word]
        elif word in self.word2id:
            return self.word2id[word]
        else:
            return self.word2id[UNK_WORD]

    def get_entity_id(self, entity):
        if entity in self.word2id:
            return self.word2id[entity]
        else:
            return self.word2id[self.dummy_entity]

    def _filter(self, array, mask):
        """
        filter array based on boolean mask
        :param array: any array
        :param mask: boolean mask
        :return: filtered
        """
        return [array[i] for i,p in enumerate(mask) if p]

    def _select(self, array, indices):
        """
        Select based on ids
        :param array:
        :param indices:
        :return:
        """
        return [array[i] for i in indices]

    def save(self, filename='data_files.pkl'):
        """
        Save the current data utility into pickle file
        :param filename: location
        :return: None
        """
        pkl.dump(self.__dict__, open(filename, 'wb'))
        logging.info("Saved data in {}".format(filename))

    def load(self, filename='data_files.pkl'):
        """
        Load previously saved data utility
        :param filename: location
        :return:
        """
        logging.info("Loading data from {}".format(filename))
        self.__dict__.update(pkl.load(open(filename,'rb')))
        logging.info("Loaded")


class SequenceDataLoader(data.Dataset):
    """
    Separate dataloader instance
    """

    def __init__(self, dataRows, data):
        """
        :param dataRows: training / validation / test data rows
        :param data: pointer to DataUtility class
        """
        self.dataRows = dataRows
        self.data = data

    def __getitem__(self, index):
        """
        Return single training row for dataloader
        :param item:
        :return:
        """
        orig_inp = self.dataRows[index].story
        inp_row_graph = self.dataRows[index].story_graph
        inp_row_pos = []
        if self.data.sentence_mode:
            sent_lengths = [len(sent) for sent in self.dataRows[index].story_sents]
            inp_row = [[self.data.get_token(word) for word in sent] for sent in self.dataRows[index].story_sents]
            inp_ents = [[id for id in sent if id in self.data.entity_ids] for sent in inp_row]
            inp_row_pos = [[widx + 1 for widx, word in enumerate(sent)] for sent in inp_row]
        else:
            sent_lengths = [len(self.dataRows[index].story)]
            inp_row = [self.data.get_token(word) for word in self.dataRows[index].story]
            inp_ents = list(set([id for id in inp_row if id in self.data.entity_ids]))

        ## calculate one-hot mask for entities which are used in this row
        flat_inp_ents = inp_ents
        if self.data.sentence_mode:
            flat_inp_ents = [p for x in inp_ents for p in x]
        inp_ent_mask = [1 if idx+1 in flat_inp_ents else 0 for idx in range(len(self.data.entity_ids))]


        # calculate for each entity pair which sentences contain them
        # output should be a max_entity x max_entity x num_sentences --> which should be later padded
        # if not sentence mode, then just output max_entity x max_entity x 1
        num_sents = len(inp_row) # 8, say
        if self.data.sentence_mode:
            assert len(inp_row) == len(inp_ents)
            sentence_pointer = np.zeros((len(self.data.entity_ids), len(self.data.entity_ids),
                                         num_sents))
            for sent_idx, inp_ent in enumerate(inp_ents):
                if len(inp_ent) > 1:
                    for ent1, ent2 in it.combinations(inp_ent, 2):
                        # check if two same entities are not appearing
                        if ent1 == ent2:
                            raise NotImplementedError("For now two same entities cannot appear in the same sentence")
                        assert ent1 != ent2
                        # remember we are shifting one bit here
                        sentence_pointer[ent1-1][ent2-1][sent_idx] = 1

        else:
            sentence_pointer = np.ones((len(self.data.entity_ids), len(self.data.entity_ids), 1))


        # calculate the output
        target = [self.dataRows[index].target]
        query = [self.data.get_token(tp) for tp in self.dataRows[index].query] # tuple
        # debugging
        if self.data.get_token('UNKUNK') in query:
            print("shit")
            raise AssertionError("Unknown element cannot be in the query. Check the data.")
        # one hot integer mask over the input text which specifies the query strings
        query_mask = [[1 if w == ent else 0 for w in self.__flatten__(inp_row)] for ent in query]
        # TODO: use query_text and query_text length and pass it back
        # text_query = [self.data.get_token(tp) for tp in self.dataRows[index].text_query]
        text_query = []
        text_target = [START_TOKEN] + self.dataRows[index].text_target + [END_TOKEN]
        text_target = [self.data.get_token(tp) for tp in text_target]

        return inp_row, inp_ents, query, text_query, query_mask, target, text_target, inp_row_graph, \
               sent_lengths, inp_ent_mask, sentence_pointer, orig_inp, inp_row_pos

    def __flatten__(self, arr):
        if any(isinstance(el, list) for el in arr):
            return [a for b in arr for a in b]
        else:
            return arr

    def __len__(self):
        return len(self.dataRows)


## Helper functions
def simple_merge(rows):
    lengths = [len(row) for row in rows]
    padded_rows = pad_rows(rows, lengths)
    return padded_rows, lengths

def nested_merge(rows):
    lengths = []
    for row in rows:
        row_length = [len(current_row) for current_row in row]
        lengths.append(row_length)

    # lengths = [len(row) for row in rows]
    padded_rows = pad_nested_row(rows, lengths)
    return padded_rows, lengths

def simple_np_merge(rows):
    lengths = [len(row) for row in rows]
    padded_rows = pad_rows(rows, lengths)
    return padded_rows, lengths

def collate_fn(data):
    """
    helper function for torch.DataLoader
    :param data: list of tuples (inp, outp)
    :return:
    """
    ## sort dataset by inp sentences
    data.sort(key=lambda x: len(x[0]), reverse=True)
    inp_data, inp_ents, query, text_query, query_mask, target, text_target, inp_graphs, sent_lengths, inp_ent_mask, *_ = zip(*data)
    inp_data, inp_lengths = simple_merge(inp_data)
    # outp_data, outp_lengths = simple_merge(outp_data)
    text_target, text_target_lengths = simple_merge(text_target)

    # outp_data = outp_data.view(-1, outp_data.shape[2]) no need to reshape now, will do it later
    query = torch.LongTensor(query)
    query_mask = pad_ents(query_mask, inp_lengths)
    target = torch.LongTensor(target)

    # prepare batch
    batch = Batch(
        inp=inp_data,
        inp_lengths=inp_lengths,
        sent_lengths=sent_lengths,
        target=target,
        text_target=text_target,
        text_target_lengths=text_target_lengths,
        inp_ents=inp_ents,
        query=query,
        query_mask=query_mask,
        inp_graphs=torch.LongTensor(inp_graphs),
        inp_ent_mask = torch.LongTensor(inp_ent_mask)
    )

    return batch

def sent_merge(rows, sent_lengths):
    lengths = [len(row) for row in rows]
    max_sent_l = max([n for sentl in sent_lengths for n in sentl])
    padded_rows = torch.zeros(len(rows), max(lengths), max_sent_l).long()
    for i,row in enumerate(rows):
        end = lengths[i]
        for j,sent_row in enumerate(row):
            padded_rows[i, j, :sent_lengths[i][j]] = torch.LongTensor(sent_row)
    return padded_rows, lengths

def sent_collate_fn(data):
    """
    helper function for torch.DataLoader
    modified to handle sentences
    :param data: list of tuples (inp, outp)
    :return:
    """

    ## sort dataset by number of sentences
    data.sort(key=lambda x: len(x[0]), reverse=True)
    inp_data, inp_ents, query, text_query, query_mask, target, text_target, inp_graphs\
        , sent_lengths, inp_ent_mask, sentence_pointer\
        , orig_inp, inp_row_pos = zip(*data)

    inp_data, inp_lengths = sent_merge(inp_data, sent_lengths)
    inp_row_pos, _ = sent_merge(inp_row_pos, sent_lengths)
    max_node, _, _ = sentence_pointer[0].shape
    sentence_pointer = [sp.reshape(-1, sp.shape[2]) for sp in sentence_pointer]
    sentence_pointer = [sp.tolist() for sp in sentence_pointer]
    sentence_pointer = [s for sp in sentence_pointer for s in sp] # flatten
    sentence_pointer, sent_lens = simple_merge(sentence_pointer)
    sentence_pointer = sentence_pointer.view(inp_data.size(0), max_node, max_node, -1)

    sent_lengths = pad_sent_lengths(sent_lengths)

    text_target, text_target_lengths = simple_merge(text_target)
    query = torch.LongTensor(query)
    query_mask = pad_ents(query_mask, inp_lengths)
    target = torch.LongTensor(target)

    # prepare batch
    batch = Batch(
        inp=inp_data,
        inp_lengths=inp_lengths,
        sent_lengths=sent_lengths,
        target=target,
        text_target=text_target,
        text_target_lengths=text_target_lengths,
        inp_ents=inp_ents,
        query=query,
        query_mask=query_mask,
        inp_graphs=torch.LongTensor(inp_graphs),
        sentence_pointer=sentence_pointer,
        orig_inp = orig_inp,
        inp_ent_mask = torch.LongTensor(inp_ent_mask),
        inp_row_pos = inp_row_pos
    )


    return batch

def pad_rows(rows, lengths):
    padded_rows = torch.zeros(len(rows), max(lengths)).long()
    for i, row in enumerate(rows):
        end = lengths[i]
        padded_rows[i, :end] = torch.LongTensor(row[:end])
    return padded_rows

def pad_nested_row(rows, lengths):
    max_abstract_length = max([l for ln in lengths for l in ln])
    max_num_abstracts = max(list(map(len, rows)))
    padded_rows = torch.zeros(len(rows), max_num_abstracts, max_abstract_length).long()
    for i, row in enumerate(rows):
        for j, abstract in enumerate(row):
            end = lengths[i][j]
            padded_rows[i, j, :end] = torch.LongTensor(row[j][:end])
    return padded_rows


def pad_ents(ents, lengths):
    padded_ents = torch.zeros((len(ents), max(lengths), 2)).long()
    for i, row in enumerate(ents):
        end = lengths[i]
        for ent_n in range(len(row)):
            padded_ents[i, :end, ent_n] = torch.LongTensor(row[ent_n][:end])
    return padded_ents

def pad_nested_ents(ents, lengths):
    abstract_lengths = []
    batch_size = len(ents)
    abstracts_per_batch = len(ents[0])
    num_entities = len(ents[0][0])
    abstract_lengths = []
    for row in ents:
        row_length = [len(abstract_line[0]) for abstract_line in row]
        abstract_lengths.append(row_length)
    abstract_lengths = [a for c in abstract_lengths for a in c]
    max_abstract_length = max(abstract_lengths)
    padded_ents = torch.zeros(batch_size, abstracts_per_batch, num_entities, max_abstract_length).long()
    for i, batch_row in enumerate(ents):
        for j, abstract in enumerate(batch_row):
            for ent_n in range(len(abstract)):
                end = lengths[i]
                padded_ents[i, j, ent_n, :end] = torch.LongTensor(batch_row[j][ent_n][:end])
    return padded_ents

def pad_sent_lengths(sent_lens):
    """
    given sentence lengths, pad them so that the total batch length is equal
    :return:
    """
    max_len = max([len(sent) for sent in sent_lens])
    pad_lens = []
    for sent in sent_lens:
        pad_lens.append(sent + [0]*(max_len - len(sent)))
    return pad_lens

def generate_dictionary(config):
    """
    Before running an experiment, make sure that a dictionary
    is generated
    Check if the dictionary is present, if so then return
    :return:
    """
    parent_dir = os.path.abspath(os.pardir).split('/codes')[0]
    dictionary_file = os.path.join(parent_dir, 'data', config.dataset.data_path, 'dict.json')
    if os.path.isfile(dictionary_file):
        logging.info("Dictionary present at {}".format(dictionary_file))
        return
    logging.info("Creating dictionary with all test files")
    ds = DataUtility(config)
    datas = []
    train_data, max_ents = ds.process_data(os.path.join(parent_dir, 'data', config.dataset.data_path),
                    config.dataset.train_file, load_dictionary=False, preprocess=False)
    datas.append(train_data)
    for test_file in config.dataset.test_files:
        test_data, max_e = ds.process_data(os.path.join(parent_dir, 'data', config.dataset.data_path),
                        test_file, load_dictionary=False, preprocess=False)
        datas.append(test_data)
        if max_e > max_ents:
            max_ents = max_e
    ds.max_ents = max_ents
    for data in datas:
        ds.preprocess(data)

    # save dictionary
    dictionary = {
        'word2id': ds.word2id,
        'id2word': ds.id2word,
        'target_word2id': ds.target_word2id,
        'target_id2word': ds.target_id2word,
        'max_ents': ds.max_ents,
        'max_vocab': ds.max_vocab,
        'max_entity_id': ds.max_entity_id,
        'entity_ids': ds.entity_ids,
        'dummy_entitiy': ds.dummy_entity,
        'entity_map': ds.entity_map
    }
    json.dump(dictionary, open(dictionary_file,'w'))
    logging.info("Saved dictionary at {}".format(dictionary_file))

if __name__ == '__main__':
    # Generate a dictionary once and re-use it over again
    # We do this to resolve the issue of unknown elements in generalizability
    # experiments
    # Take the last training file which has the longest path and make a dictionary
    parent_dir = os.path.abspath(os.pardir).split('/codes')[0]
    config = get_config(config_id='lstm')
    generate_dictionary(config)



