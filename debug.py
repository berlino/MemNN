#!/usr/bin/python
import operator
import cPickle
import pdb

dict_file = "../WikiMovies/data/torch/dict.txt"
pkl_file = "./pkl/dic.pkl"
word2id = {}
id2word = []

def gen_dict():
    with open(dict_file) as f:
        for line in f:
            line = line.strip()
            ind = line.rfind("\t")
            w = line[:ind]
            word2id[w.lower()] = len(word2id)
    sort_l = sorted(word2id.items(), key=operator.itemgetter(1))
    id2word = [x[0] for x in sort_l]
    print "Voc len,", len(word2id)
    with open(pkl_file, 'w') as f:
        cPickle.dump((word2id, id2word), f)
    
def load_dict():
    with open(pkl_file) as f:
        word2id, id2word = cPickle.load(f)
    print "Loading dict:", len(id2word)
    return word2id, id2word

def toSent(sent):
    return " ".join([id2word[i] for i in sent])

#gen_dict()
word2id, id2word = load_dict()
#pdb.set_trace()