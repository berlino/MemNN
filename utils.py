import numpy as np
import cPickle
import torch
from config import config
import os

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    #Generates a batch iterator for a dataset.
    data = torch.Tensor(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    print "{} batches expected".format(num_batches_per_epoch * num_epochs)
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[torch.LongTensor(shuffle_indices)]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def batch_sort_iter(data, batch_size, num_epochs, padding=False, sort=True):
    #variable length but sorted
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    print "{} batches expected".format(num_batches_per_epoch * num_epochs)
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield to_tensor(data[start_index:end_index], padding = padding, sort = sort)

# padding and to tensor
def to_tensor(data, padding = False, sort=True):
    if padding:
        if not sort:  return padding_list(data, config.win_len) #TODO: Fix
        else:  return padding_list(data, len(data[-1]))
    else: return torch.LongTensor(data)

# 2 for UNK
# return LongTensor
def padding_list(l, length, pad=2):
    ret_l = []
    for item in l:
        if len(item) < length:
            pad_item = [pad for _ in range(length - len(item))]
            item = list(item) + list(pad_item)
        assert len(item) == length
        ret_l.append(item)
    return torch.LongTensor(ret_l)

def dump_to_file(obj, filename):
    path = os.path.dirname(filename)
    if not os.path.exists(path):
        print "Warning: file path not exisits, writed in the cur dir"
        filename = "./" + os.path.basename(filename)
    with open(filename, "w") as f:
        print "Dumping to:", filename
        cPickle.dump(obj, f)

def load_from_file(filename):
    print "Loading: ", filename
    with open(filename) as f:
        return cPickle.load(f)

