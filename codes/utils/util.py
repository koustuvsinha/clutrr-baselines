import collections
import pathlib
import random
from random import shuffle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import subprocess

NP_INT_DATATYPE = np.int

def flatten(d, parent_key='', sep='_'):
    # Logic for flatten taken from https://stackoverflow.com/a/6027615/1353861
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def grouped(iterable, n):
    # Modified from https://stackoverflow.com/questions/5389507/iterating-over-every-two-elements-in-a-list/39038787
    return zip(*[iter(iterable)] * n)


# Taken from https://stackoverflow.com/questions/38191855/zero-pad-numpy-array/38192105
def padarray(A, size, const=1):
    t = size - len(A)
    return np.pad(A, pad_width=(0, t), mode='constant', constant_values=const)


def parse_file(file_name):
    '''Method to read the given input file and return an iterable for the lines'''
    with open(file_name, encoding='utf-8') as f:
        for line in f:
            yield line


def get_device_id(device):
    if (device == "cpu"):
        return -1
    elif (device == "gpu"):
        return None
    else:
        return None


def shuffle_list(*ls):
    """Taken from https://stackoverflow.com/questions/23289547/shuffle-two-list-at-once-with-same-order"""
    l = list(zip(*ls))
    shuffle(l)
    return zip(*l)


def chunks(l, n):
    """
    Taken from https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def reverse_dict(_dict):
    return {v: k for k, v in _dict.items()}


def padarray(A, size, const=0):
    # Taken from https://stackoverflow.com/questions/38191855/zero-pad-numpy-array/38192105
    t = size - len(A)
    return np.pad(A, pad_width=(0, t), mode='constant', constant_values=const).astype(NP_INT_DATATYPE)


def make_dir(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def get_device_name(device_type):
    if torch.cuda.is_available() and "cuda" in device_type:
        return device_type
    return "cpu"

def get_current_commit_id():
    command = "git rev-parse HEAD"
    commit_id = subprocess.check_output(command.split()).strip().decode("utf-8")
    return commit_id

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def merge_first_two_dims_batch(batch):
    shape = batch.shape
    return batch.view(shape[0]*shape[1], *shape[2:])

def unmerge_first_two_dims_batch(batch, first_dim=None, second_dim = None):
    shape = batch.shape
    if(first_dim):
        second_dim = int(shape[0]/first_dim)
    elif(second_dim):
        first_dim = int(shape[0] / second_dim)
    return batch.view(first_dim, second_dim, *shape[1:])



def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    Source: https://github.com/OpenNMT/OpenNMT-py/blob/2e6935f738b5c2be26d51e3ba35c9453c77e0566/onmt/utils/misc.py#L29
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
         .transpose(0, 1) \
         .repeat(count, 1) \
         .transpose(0, 1) \
         .contiguous() \
         .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x

def check_id_emb(tensor, max_id, min_id=0):
    """
    check if the ids are within max id before running through Embedding layer
    :param tensor:
    :param max_id:
    :return:
    """
    assert tensor.lt(max_id).all()
    assert tensor.ge(min_id).all()

def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels.long()]

def aeq(tensor_a, tensor_b):
    """
    Assert given two tensors are equal in dimensions
    :param tensor_a:
    :param tensor_b:
    :return:
    """
    assert tensor_a.dim() == tensor_b.dim()
    for d in range(tensor_a.dim()):
        assert tensor_a.size(d) == tensor_b.size(d)

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table
      https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L14'''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

def flatten_dictionary(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_hid, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        #output = self.layer_norm(output + residual)
        return output

