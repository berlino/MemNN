#!/usr/local/bin/python
from __future__ import division
import numpy as np
import torchfile
from config import config
from utils import *
from data_loader import *
import pdb

# generating title and entity_dic
def make_dict(x,y):
    title_dict = {}
    entity_dict = {}
    cnt = 0
    for sent, a in zip(x,y):
        if sent[0] == 6:
            #movie
            sent = sent[1:]
            title_dict[tuple(sent)] = a[0]
        else:
            #window
            assert sent[0] == 7
            sent = sent[2:]
            entity_dict[tuple(sent)] = a[0]
            if a[0] not in sent:
                cnt += 1
    print "Pos bug", cnt / len(y)
    for sent, a in zip(x,y):
        if sent[0] == 6:
            #movie
            continue
        else:
            #window
            assert sent[0] == 7
            sent_ = sent[2:]
            k_ = tuple(sent_)
            if k_ not in title_dict:
                title_dict[k_] = sent[1]
    return title_dict, entity_dict

d1, d2 = make_dict(wiki_q, wiki_a)
#pdb.set_trace()
dump_to_file(d1, config.title_dict)
dump_to_file(d2, config.entity_dict)
