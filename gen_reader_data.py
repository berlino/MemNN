#!/usr/bin/python
from __future__ import division
from config import config
from utils import *
from data_loader import *
import pdb
from Model import MLP

# dump path
train_file_path =  "./pkl/reader/300/train_pair.pkl"
dev_file_path =  "./pkl/reader/300/dev_pair.pkl"
test_file_path =  "./pkl/reader/300/test_pair.pkl"

model = MLP(config)
model.load(config.pre_embed_file)

title_dict = load_from_file(config.title_dict)
entity_dict= load_from_file(config.entity_dict)

def predict_sim(x,y):
    x = torch.LongTensor([x])
    y = torch.LongTensor([y])
    sim = torch.LongTensor([1])
    #pdb.set_trace()
    loss = model.forward(x,y,sim)
    return loss.data.numpy()[0]

def extract_ans_pair(golds, wiki_ans):
    s1 = set(golds)
    s2 = set(wiki_ans)
    s_ = s1.intersection(s2)
    return list(s_)[0]

# the answer of test is list
def make_pairs(x,y,test=False):
    eos = 1 # index
    unk = 2
    ret_pair = []
    for q, a in zip(x,y):
        cand_facts = []
        for token in q:
            if token >= len(wiki_hash): continue
            cand_inds = wiki_hash[token]
            if len(cand_inds) == 0: continue 

            # for every cand in pre-select candidates
            for cand in cand_inds:
                if cand == -1: continue  # none
                s_sim_pair = None # store the result

                # reform the sentence and extract the entity position
                sent = wiki_q[cand]
                ent_pos = 0
                if sent[0] == 6:
                    #movie
                    continue
                else:
                    #window
                    assert sent[0] == 7
                    raw_sent = sent[2:]
                    try:
                        sent = [6, title_dict[tuple(raw_sent)],6] +  sent[2:]
                    except:
                        sent = [6, entity_dict[tuple(raw_sent)], 6] +  sent[2:]
                    try:
                        ent_pos = sent.index(entity_dict[tuple(raw_sent)])
                    except:
                        # bug of preprocesing
                        e_tmp_p = (len(sent) -3) // 2 + 3
                        sent = sent[:e_tmp_p] + [entity_dict[tuple(raw_sent)]] + sent[e_tmp_p:]
                        ent_pos = e_tmp_p
                        #pdb.set_trace()
                        #print "Pos bug"

                # sentence too long or too short
                if len(sent) < config.win_len:
                    sent = sent + [unk for _ in range(config.win_len - len(sent))]
                else:
                    # change based on the end_pos
                    if ent_pos <= 6:
                        sent = sent[:config.win_len]
                    else:
                        pre_len = len(sent)
                        sent = sent[:3] + sent[pre_len-config.win_len+3:]
                        ent_pos = ent_pos + config.win_len - pre_len
                assert len(sent) == config.win_len and ent_pos < config.win_len

                # add the pair to the list
                s_sim_pair = (predict_sim(q,sent), ent_pos)
                cand_facts.append((s_sim_pair, sent, ent_pos))

        # filter out the top K facts
        cand_facts.sort(key=lambda x:x[0]) 
        filter_size = min(len(cand_facts), config.filter_size)
        cand_qs = [x[1] for x in cand_facts[:filter_size]]
        cand_ent_pos = [x[2] for x in cand_facts[:filter_size]]
        if test:
            ret_pair.append((q, cand_qs, cand_ent_pos, a))
        else:
            ret_pair.append((q, cand_qs, cand_ent_pos, a[0]))

    # convert to tensor for batch
    ret_pair.sort(key=lambda x:len(x[0]))
    ret_q = []
    ret_w = []
    ret_a = []
    ret_ent_p = []
    for q_, w_q_, w_e_p_, a_ in ret_pair:
        ret_q.append(torch.LongTensor(q_))
        ret_w.append(torch.LongTensor(w_q_))
        ret_ent_p.append(torch.LongTensor(w_e_p_))
        if not test:
            ret_a.append(torch.LongTensor([a_])) # single value must have []
        else:
            ret_a.append(a_) # test no need to be tensor
    return (ret_q, ret_w, ret_ent_p, ret_a)

def evaluate(data):
    cnt1 = 0
    cnt2 = 0
    cnt3 = 0
    q, w, e_p, a = data
    for q_, w_, a_ in zip(q, w, a):
        #pdb.set_trace()
        w_np = w_.numpy()
        cnt3 += len(w_np)
        if len(w_np) == 0: continue


        # top 1
        best_w = w_np[0]
        for tmp_a in a_:
            sig2 = tmp_a in best_w
            if sig2:
                cnt2 += 1
                break

        # coverage
        for tmp_a in a_:
            sig1 = [tmp_a in tmp_w for tmp_w in w_np]
            if max(sig1): 
                cnt1 += 1
                break
    return cnt1 / len(a), cnt2 / len(a), cnt3 / len(a)

if __name__ == "__main__":
    train_pairs = make_pairs(train_q, train_a)
    dump_to_file(train_pairs, train_file_path)
    dev_pairs = make_pairs(dev_q, dev_a)
    dump_to_file(dev_pairs, dev_file_path)
    test_pairs = make_pairs(test_q, test_a, test=True)
    dump_to_file(test_pairs, test_file_path)
    
    #test_pairs = load_from_file(test_file_path)
    print evaluate(test_pairs)
