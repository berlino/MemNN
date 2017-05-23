#!/usr/bin/python
from __future__ import division
from Model import KVMemoryReader
from config import config
import operator
from utils import *
import pdb
from debug import *
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
from torch.legacy.nn import CosineEmbeddingCriterion
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

train_q, train_w, train_e_p, train_a = load_from_file("./pkl/reader/{}/train_pair.pkl".format(config.d_embed))
dev_q, dev_w, dev_e_p, dev_a = load_from_file("./pkl/reader/{}/dev_pair.pkl".format(config.n_embed))
#train_q, train_w, train_e_p, train_a = load_from_file("./pkl/toy/reader/train_pair.pkl")
#dev_q, dev_w, dev_e_p, dev_a = load_from_file("./pkl/toy/reader/dev_pair.pkl")

def modify(q, wiki, pos, ans):
    tL = torch.LongTensor
    ret_q = []
    ret_key = []
    ret_value = []
    ret_cand = []
    ret_a = []
    for qu,w,p,a_ in zip(q,wiki,pos,ans):
        # encoding the candidate
        can_dict = {}
        qu = qu.numpy()
        w = w.numpy()
        p = p.numpy()
        a_ = a_.numpy()

        # generate local candidate
        len_w = len(w)
        cand_ind = []
        for i in range(len_w):
            if w[i][1] not in can_dict:
                can_dict[w[i][1]] = len(can_dict)
            if w[i][p[i]] not in can_dict:
                can_dict[w[i][p[i]]] = len(can_dict)
        if a_[0] not in can_dict:
            continue
        else:
            sort_l = sorted(can_dict.items(), key=operator.itemgetter(1))
            cand_l = [x[0] for x in sort_l]

            # split into key value format
            #pdb.set_trace()
            key_m, val_m = transKV(w,p)
            ret_q.append(tL(qu))
            ret_key.append(tL(key_m))
            ret_value.append(tL(val_m))
            ret_cand.append(tL(cand_l))
            ret_a.append(tL([can_dict[a_[0]]]))
    print len(ret_q) / len(q)
    return ret_q, ret_key,ret_value,ret_cand, ret_a 

def transKV(sents, pos):
    unk = 2
    ret_k = []
    ret_v = []
    for sent, p in zip(sents,pos):
        k_ = sent[3:].tolist() + [unk]
        v_ = sent[1]
        #pdb.set_trace()
        ret_k.append(k_)
        ret_v.append(v_)
        #print toSent(k_),toSent([v_])

        k_ = [sent[1]] + sent[3:].tolist()
        v_ = sent[p] 
        ret_k.append(k_)
        ret_v.append(v_)
        #print toSent(k_),toSent([v_])
    return np.array(ret_k), np.array(ret_v)

def train(epoch): 
    for e_ in range(epoch):
	if (e_ + 1) % 10 == 0:
            adjust_learning_rate(optimizer, e_)
        cnt = 0
        loss = Variable(torch.Tensor([0]))
        for i_q, i_k, i_v, i_cand, i_a in zip(train_q, train_key,train_value, train_cand, train_a):
            cnt += 1
            i_q = i_q.unsqueeze(0) # add dimension
            probs = model.forward(i_q, i_k, i_v,i_cand)
            i_a = Variable(i_a)
            curr_loss = loss_function(probs, i_a)
            loss = torch.add(loss, torch.div(curr_loss, config.batch_size)) 
            
            # naive batch implemetation, the lr is divided by batch size
            if cnt % config.batch_size == 0:
                print "Training loss", loss.data.sum()
                loss.backward()
                optimizer.step()
                loss = Variable(torch.Tensor([0]))
                model.zero_grad()
            if cnt % config.valid_every == 0:
                print "Accuracy:",eval()

def adjust_learning_rate(optimizer, epoch):
    lr = config.lr / (2 ** (epoch // 10))
    print "Adjust lr to ", lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def eval():
    cnt = 0
    for i_q, i_k, i_v, i_cand, i_a in zip(dev_q, dev_key, dev_value, dev_cand, dev_a):
        i_q = i_q.unsqueeze(0) # add dimension
        try:
            ind = model.predict(i_q, i_k, i_v, i_cand)
        except:
            continue
        if ind == i_a[0]:
            cnt += 1
    return cnt / len(dev_q)

model = KVMemoryReader(config)
model.load_embed(config.pre_embed_file)
# here lr is divide by batch size since loss is accumulated 
optimizer = optim.SGD(model.parameters(), lr=config.lr)
print "Training setting: lr {0}, batch size {1}".format(config.lr, config.batch_size)

loss_function = nn.NLLLoss()

print "{} batch expected".format(len(train_q) * config.epoch / config.batch_size)
train_q, train_key, train_value, train_cand,train_a = modify(train_q, train_w, train_e_p, train_a)
dev_q, dev_key, dev_value, dev_cand, dev_a = modify(dev_q, dev_w, dev_e_p, dev_a)
train(config.epoch)
dump_to_file(model, config.reader_model)
