#!/usr/local/bin/python
from __future__ import division
import numpy as np
from data_loader import *
from config import config
from utils import *
import pdb

# parameters
train_file_path =  "./pkl/sim/train_pair.pkl"
dev_file_path =  "./pkl/sim/dev_pair.pkl"
test_file_path =  "./pkl/sim/test_pair.pkl"

title_dict = load_from_file(config.title_dict)
entity_dict= load_from_file(config.entity_dict)

# evaluate coverage
def evaluate(x, y, notitle=False):
    cnt = 0
    for q, a in zip(x,y):
        for token in q:
            flag = False
            if token >= len(wiki_hash): continue
            cand_inds = wiki_hash[token]
            if len(cand_inds) == 0: continue
            for cand in cand_inds:
                if cand == -1: continue
                sig = False
                sent = wiki_q[cand]
                if sent[0] == 6:
                    #movie
                    if not notitle:
                        sig = max([i_ in a for i_ in wiki_a[cand]])
                else:
                    #window
                    assert sent[0] == 7
                    sig = -1
                    if title_dict:
                        # change 1: to original
                        k_ = tuple(sent[2:])
                        try:
                            sent[1] = title_dict[tuple(sent[2:])] 
                        except:
                            pass
                        sig = max([i_ in a for i_ in sent])
                    else:
                        sig = max([i_ in a for i_ in wiki_a[cand]])
                if sig == True:
                    cnt += 1
                    flag = True
                    break
            if flag: break
    return cnt / len(x)

'''
1. raw sent with sim
2. form a special sent with title
'''
def make_pairs(x,y):
    pairs = {}
    for q, a in zip(x,y):
        cnt = 0
        for token in q:
            if token >= len(wiki_hash): continue
            cand_inds = wiki_hash[token]
            if len(cand_inds) == 0: continue 
            for cand in cand_inds:
                if cand == -1: continue  # none
                sent = wiki_q[cand]
                sim = -1
                if sent[0] == 6:
                    #movie
                    #sent = sent[1:]
                    #sig = max([i_ in a for i_ in wiki_a[cand]])
                    #if sig: sim = 1 #TODO: fix the loss function
                    continue
                else:
                    #window
                    assert sent[0] == 7
                    sent = sent[2:]
                    sig = max([i_ in a for i_ in wiki_a[cand]])
                    if sig: sim = 1
                pair = (tuple(q), tuple(sent))
                if pair not in pairs:
                    pairs[pair] = sim
                    cnt += 1
                elif pairs[pair] != sim:
                    # it's related to the notitle
                    pairs[pair] = max(sim, pairs[pair])
        if cnt > config.ir_size: print "Too many windows:{}".format(cnt)
    ret_pair = []
    for k in pairs:
        #pdb.set_trace()
        q_,a_ = k
        try:
            a_ = [6, title_dict[a_], 6] + list(a_)
        except:
            a_ = [6, entity_dict[a_], 6] + list(a_)
        ret_pair.append((q_,a_,pairs[k]))
    # for batch
    ret_pair.sort(key=lambda x:len(x[0]))
    ret_q = []
    ret_a = []
    ret_sim = []
    for q_, a_, s_ in ret_pair:
        ret_q.append(q_)
        # TODO: why extra length
        ret_a.append(a_[:config.win_len])
        ret_sim.append(s_)
    return (ret_q, ret_a, ret_sim)

if __name__ == "__main__":
    print "Evaluating coverage"
    print evaluate(test_q, test_a, notitle = False)
    print evaluate(test_q, test_a, notitle = True)
    
    train_pairs = make_pairs(train_q, train_a)
    dump_to_file(train_pairs, train_file_path)
    dev_pairs = make_pairs(dev_q, dev_a)
    dump_to_file(dev_pairs, dev_file_path)
    #test_pairs = make_pairs(test_q, test_a)
    #dump_to_file(test_pairs, dev_file_path)
