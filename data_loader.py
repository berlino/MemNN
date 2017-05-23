#!/usr/bin/python
from __future__ import division
from config import config
import torchfile
from utils import *
import pdb

data_dir = config.data_dir
train_file = data_dir + "/torch/train_1.txt"
dev_file = data_dir + "/torch/dev_1.txt"
test_file = data_dir + "/torch/test_1.txt"
wiki_file = data_dir + "/torch/wiki-w=0-d=3-i-m.txt"

train_x = torchfile.load(train_file + ".vecarray.x")
train_y = torchfile.load(train_file + ".vecarray.y")
dev_x = torchfile.load(dev_file + ".vecarray.x")
dev_y = torchfile.load(dev_file + ".vecarray.y")
test_x = torchfile.load(test_file + ".vecarray.x")
test_y = torchfile.load(test_file + ".vecarray.y")
wiki_x = torchfile.load(wiki_file + ".hash.facts1_va")
wiki_y = torchfile.load(wiki_file + ".hash.facts2_va")
wiki_ind_ = torchfile.load(wiki_file + ".hash.facts_ind")
wiki_hash_ = torchfile.load(wiki_file + ".hash.facts_hash_va")

#pdb.set_trace()

def get(x, i):
    start = int(x["idx"][i]) - 1
    length = int(x["len"][i])
    ret = x["data"][start : start + length]
    return [ int(x) -1 for x in ret]

def extract(x,y):
    q = []
    a = []
    l = int(x["cnt"][0])
    for i in range(l):
        q_ = get(x,i)
        a_ = get(y,i)
        q.append(q_)
        a.append(a_)
    return q, a

train_q, train_a = extract(train_x, train_y)
dev_q, dev_a = extract(dev_x, dev_y)
test_q, test_a = extract(test_x, test_y)
wiki_q, wiki_a = extract(wiki_x, wiki_y)
# -1 represents none !
wiki_hash, _ = extract(wiki_hash_, wiki_hash_)
#wiki_ind = extract(wiki_ind_, wiki_ind_)

assert len(train_q) == 96185
assert len(dev_q) == 10000
