#/usr/bin/python
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.legacy.nn import CosineEmbeddingCriterion
from torch.legacy.nn import Sum
import pdb

class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.embed = nn.Embedding(config.n_embed, config.d_embed)
        self.cosine = nn.CosineEmbeddingLoss()
        #self.mean = Sum(0,True)

    def forward(self, x1, x2, y):
        #pdb.set_trace()
        v1 = Variable(x1)
        v2 = Variable(x2)
        y = Variable(y)
        v1 = self.embed(v1)
        v1 = v1.mean(1).squeeze(1)
        v2 = self.embed(v2)
        v2 = v2.mean(1).squeeze(1)
        #pdb.set_trace()
        loss = self.cosine(v1,v2,y)
        return loss

    def save(self, filename):
        tmp = [x for x in self.parameters()]
        with open(filename, "w") as f:
            torch.save(tmp[0], f) 

    def load(self, filename):
        embed_t = None
        with open(filename) as f:
            embed_t = torch.load(f)
        self.embed.weight = embed_t


class RnnReader(nn.Module):
    def __init__(self, config):
        super(RnnReader, self).__init__()
        self.config = config
        self.embed = nn.Embedding(config.n_embed, config.d_embed)
        self.rnn_doc = nn.GRU(config.d_embed, config.rnn_fea_size, batch_first=True, bidirectional=True)
        self.rnn_qus = nn.GRU(config.d_embed, config.rnn_fea_size, batch_first=True, bidirectional=True)
        self.h0_doc = torch.rand(2,1, self.config.rnn_fea_size)
        self.h0_q = Variable(torch.rand(2, 1, self.config.rnn_fea_size))

    def forward(self, qu, w, e_p):
        qu = Variable(qu)
        w = Variable(w)
        embed_q = self.embed(qu)
        embed_w = self.embed(w)
        s_ = embed_w.size()
        b_size = s_[0]

        #pdb.set_trace()
        h0_doc = Variable(torch.cat([self.h0_doc for _ in range(b_size)], 1))
        out_qus, h_qus = self.rnn_qus(embed_q, self.h0_q)
        out_doc, h_doc = self.rnn_doc(embed_w, h0_doc)

        q_state = torch.cat([out_qus[0,-1,:self.config.rnn_fea_size], out_qus[0,0,self.config.rnn_fea_size:]],0)
        
        # token attention
        doc_tit_ent_dot = []
        doc_tit_ent = []
        doc_states = []
        for i,k in enumerate(e_p):
            # memory
            t_e_v = self.cat(out_doc[i,1], out_doc[i,k])
            # dot product
            title = torch.dot(out_doc[i,1], q_state)
            entity = torch.dot(out_doc[i,k], q_state)
            token_att = torch.cat([title, entity],0).unsqueeze(0)
            s_m = F.softmax(token_att)
            att_v = torch.mm(s_m, t_e_v)
            doc_tit_ent.append(att_v)
            # concate start and end
            state_ = torch.cat([out_doc[i,-1,:self.config.rnn_fea_size], out_doc[i,0,self.config.rnn_fea_size:]],0)
            doc_states.append(state_.unsqueeze(0))
        #pdb.set_trace()
        t_e_vecs = torch.cat(doc_tit_ent,0)

        # sentence attention
        doc_states_v = torch.cat(doc_states, 0)
        doc_dot = torch.mm(doc_states_v, q_state.unsqueeze(1))
        doc_sm = F.softmax(doc_dot)
        t_doc_feat = torch.add(doc_states_v, t_e_vecs)
        doc_feat = torch.mm(doc_sm.view(1,-1), t_doc_feat)

        score = torch.mm(self.embed.weight, doc_feat.view(-1,1)).view(1,-1)
        score_n = F.log_softmax(score)

        return score_n

    def predict(self, q, w, e_p):
        score = self.forward(q, w, e_p)
        _, index = torch.max(score.squeeze(0), 0)
        return index.data.numpy()[0]

    # concat 1-D tensor
    def cat(self, t1, t2):
        return torch.cat([t1.unsqueeze(0), t2.unsqueeze(0)],0)

    def load_embed(self, filename):
        embed_t = None
        with open(filename) as f:
            embed_t = torch.load(f)
        self.embed.weight = embed_t

class LocalReader(nn.Module):
    def __init__(self, config):
        super(LocalReader, self).__init__()
        self.config = config
        self.embed = nn.Embedding(config.n_embed, config.d_embed)
        self.rnn = nn.GRU(config.d_embed, config.rnn_fea_size, batch_first=True, bidirectional=True)
        #self.rnn_doc = nn.GRU(config.d_embed, config.rnn_fea_size, batch_first=True, bidirectional=True)
        #self.rnn_qus = nn.GRU(config.d_embed, config.rnn_fea_size, batch_first=True, bidirectional=True)
        self.h0_doc = torch.rand(2,1, self.config.rnn_fea_size)
        self.h0_q = Variable(torch.rand(2, 1, self.config.rnn_fea_size))

    def forward(self, qu, w, e_p):
        qu = Variable(qu)
        w = Variable(w)
        embed_q = self.embed(qu)
        embed_w = self.embed(w)
        s_ = embed_w.size()
        b_size = s_[0]

        #pdb.set_trace()
        h0_doc = Variable(torch.cat([self.h0_doc for _ in range(b_size)], 1))
        out_qus, h_qus = self.rnn_qus(embed_q, self.h0_q)
        out_doc, h_doc = self.rnn_doc(embed_w, h0_doc)

        q_state = torch.cat([out_qus[0,-1,:self.config.rnn_fea_size], out_qus[0,0,self.config.rnn_fea_size:]],0)
        #q_state = out_qus[:,-1,:]

        # token attention
        title_states = []
        entity_states = []
        candidate_states = []
        doc_states = []
        for i,bu in enumerate(e_p):
            k, t_i, e_i = bu
            # memory
            #title_states.append(out_doc[i,1].unsqueeze(0))
            #entity_states.append(out_doc[i,k].unsqueeze(0))
            #state_ = torch.cat([out_doc[i,-1,:self.config.rnn_fea_size], out_doc[i,0,self.config.rnn_fea_size:]],0)
            #doc_states.append(state_.unsqueeze(0))

            if t_i < len(candidate_states):
                torch.add(candidate_states[t_i], out_doc[i,1].unsqueeze(0))
            else:
                candidate_states.append(out_doc[i,1].unsqueeze(0))
            if e_i < len(candidate_states):
                torch.add(candidate_states[e_i], out_doc[i,k].unsqueeze(0))
            else:
                candidate_states.append(out_doc[i,k].unsqueeze(0))
        #doc_states_v = torch.cat(doc_states, 0)
        #title_states_v = torch.cat(title_states, 0)
        #entity_states_v = torch.cat(entity_states, 0)
        cand_states_v = torch.cat(candidate_states, 0)

        # add
        #title_states_v = torch.add(doc_states_v, title_states_v)
        #entity_states_v = torch.add(doc_states_v, entity_states_v)
        # final feature
        #f_fea_v = torch.cat([title_states_v, entity_states_v], 0)
        #f_fea_v = torch.mm(q_state.unsqueeze(0), torch.transpose(f_fea_v,0,1))
        f_fea_v = torch.mm(q_state.unsqueeze(0), torch.transpose(cand_states_v,0,1))

        score_n = F.log_softmax(f_fea_v)
        return score_n

    def predict(self, q, w, e_p):
        score = self.forward(q, w, e_p)
        _, index = torch.max(score.squeeze(0), 0)
        return index.data.numpy()[0]

    # concat 1-D tensor
    def cat(self, t1, t2):
        return torch.cat([t1.unsqueeze(0), t2.unsqueeze(0)],0)

    def load_embed(self, filename):
        embed_t = None
        with open(filename) as f:
            embed_t = torch.load(f)
        self.embed.weight = embed_t

class MemoryReader(nn.Module):
    def __init__(self, config):
        super(MemoryReader, self).__init__()
        self.config = config
        self.embed_A = nn.Embedding(config.n_embed, config.d_embed)
        self.embed_B = nn.Embedding(config.n_embed, config.d_embed)
        self.embed_C = nn.Embedding(config.n_embed, config.d_embed)
        self.H = nn.Linear(config.d_embed, config.d_embed)

    def forward(self, qu, w, cand):
        qu = Variable(qu)
        w = Variable(w)
        cand = Variable(cand)
        embed_q = self.embed_B(qu)
        embed_w1 = self.embed_A(w)
        embed_w2 = self.embed_C(w)
        embed_c = self.embed_C(cand)

        #pdb.set_trace()
        q_state = torch.sum(embed_q, 1).squeeze(1)
        w1_state = torch.sum(embed_w1, 1).squeeze(1)
        w2_state = torch.sum(embed_w2, 1).squeeze(1)

        for _ in range(self.config.hop):
            sent_dot = torch.mm(q_state, torch.transpose(w1_state, 0, 1))
            sent_att = F.softmax(sent_dot)

            a_dot = torch.mm(sent_att, w2_state)
            a_dot = self.H(a_dot)
            q_state = torch.add(a_dot, q_state)

        f_feat = torch.mm(q_state, torch.transpose(embed_c, 0, 1))
        score = F.log_softmax(f_feat)
        return score

    def predict(self, q, w, e_p):
        score = self.forward(q, w, e_p)
        _, index = torch.max(score.squeeze(0), 0)
        return index.data.numpy()[0]

    def load_embed(self, filename):
        embed_t = None
        with open(filename) as f:
            embed_t = torch.load(f)
        self.embed_A.weight = embed_t
        self.embed_B.weight = embed_t
        self.embed_C.weight = embed_t

class RLReader(nn.Module):
    def __init__(self, config):
        super(RLReader, self).__init__()
        self.config = config
        self.embed_A = nn.Embedding(config.n_embed, config.d_embed)
        self.embed_B = nn.Embedding(config.n_embed, config.d_embed)
        self.embed_C = nn.Embedding(config.n_embed, config.d_embed)
        self.H = nn.Linear(config.d_embed, config.d_embed)
        self.rnn_doc = nn.GRU(config.d_embed, config.rnn_fea_size, batch_first=True, bidirectional=False)
        self.rnn_qus = nn.GRU(config.d_embed, config.rnn_fea_size, batch_first=True, bidirectional=False)
        self.h0_doc = Variable(torch.rand(2, 1, self.config.rnn_fea_size))
        self.h0_q = Variable(torch.rand(2, 1, self.config.rnn_fea_size))


    def forward(self, qu, w, cand):
        qu = Variable(qu)
        w = Variable(w)
        cand = Variable(cand)
        embed_q = self.embed_B(qu)
        embed_w1 = self.embed_A(w)
        embed_c = self.embed_C(cand)

        #pdb.set_trace()
        q_state = torch.sum(embed_q, 1).squeeze(1)
        w1_state = torch.sum(embed_w1, 1).squeeze(1)

        sent_dot = torch.mm(q_state, torch.transpose(w1_state, 0, 1))
        sent_att = F.softmax(sent_dot)

        q_rnn_state = self.rnn_qus(embed_q, self.h0_q)[-1].squeeze(0)
        #pdb.set_trace()

        action = sent_att.multinomial()

        sent = embed_w1[action.data[0]]
        sent_state = self.rnn_doc(sent, self.h0_doc)[-1].squeeze(0)
        q_state = torch.add(q_state, sent_state)

        f_feat = torch.mm(q_state, torch.transpose(embed_c, 0, 1))
        reward_prob = F.log_softmax(f_feat).squeeze(0)

        return action, reward_prob

    def predict(self, q, w, e_p):
        _, score = self.forward(q, w, e_p)
        _, index = torch.max(score.squeeze(0), 0)
        return index.data.numpy()[0]

    def load_embed(self, filename):
        embed_t = None
        with open(filename) as f:
            embed_t = torch.load(f)
        self.embed_A.weight = embed_t
        self.embed_B.weight = embed_t
        self.embed_C.weight = embed_t


class PairReader(nn.Module):
    def __init__(self, config):
        super(PairReader, self).__init__()
        self.config = config
        self.embed = nn.Embedding(config.n_embed, config.d_embed)
        self.rnn = nn.LSTM(config.d_embed, config.rnn_fea_size, batch_first=True, bidirectional=False, dropout=0.1)
        self.h0 = Variable(torch.FloatTensor(1, 1, self.config.rnn_fea_size).zero_())
        self.c0 = Variable(torch.FloatTensor(1, 1, self.config.rnn_fea_size).zero_())

    def forward(self, qu, w, cand):
        qu = Variable(qu)
        cand = Variable(cand)
        embed_q = self.embed(qu)
        embed_cand = self.embed(cand)

        out, (self.h0, self.c0) = self.rnn(embed_q, (self.h0, self.c0))
        self.h0.detach_()
        self.c0.detach_()
        q_state = out[:,-1,:]

        f_fea_v = torch.mm(q_state, torch.transpose(embed_cand,0,1))

        score_n = F.log_softmax(f_fea_v)
        return score_n

    def predict(self, q, w, e_p):
        score = self.forward(q, w, e_p)
        _, index = torch.max(score.squeeze(0), 0)
        return index.data.numpy()[0]

    # concat 1-D tensor
    def cat(self, t1, t2):
        return torch.cat([t1.unsqueeze(0), t2.unsqueeze(0)],0)

    def load_embed(self, filename):
        embed_t = None
        with open(filename) as f:
            embed_t = torch.load(f)
        self.embed.weight = embed_t

class KVMemoryReader(nn.Module):
    def __init__(self, config):
        super(KVMemoryReader, self).__init__()
        self.config = config
        self.embed_A = nn.Embedding(config.n_embed, config.d_embed)
        self.embed_B = nn.Embedding(config.n_embed, config.d_embed)
        self.embed_C = nn.Embedding(config.n_embed, config.d_embed)
        self.H = nn.Linear(config.d_embed, config.d_embed)

    def forward(self, qu, key, value, cand):
        qu = Variable(qu)
        key = Variable(key)
        value = Variable(value)
        cand = Variable(cand)
        embed_q = self.embed_B(qu)
        embed_w1 = self.embed_A(key)
        embed_w2 = self.embed_C(value)
        embed_c = self.embed_C(cand)

        #pdb.set_trace()
        q_state = torch.sum(embed_q, 1).squeeze(1)
        w1_state = torch.sum(embed_w1, 1).squeeze(1)
        w2_state = embed_w2

        for _ in range(self.config.hop):
            sent_dot = torch.mm(q_state, torch.transpose(w1_state, 0, 1))
            sent_att = F.softmax(sent_dot)

            a_dot = torch.mm(sent_att, w2_state)
            a_dot = self.H(a_dot)
            q_state = torch.add(a_dot, q_state)

        f_feat = torch.mm(q_state, torch.transpose(embed_c, 0, 1))
        score = F.log_softmax(f_feat)
        return score

    def predict(self, q, key, value, cand):
        score = self.forward(q, key, value, cand)
        _, index = torch.max(score.squeeze(0), 0)
        return index.data.numpy()[0]

    def load_embed(self, filename):
        embed_t = None
        with open(filename) as f:
            embed_t = torch.load(f)
        self.embed_A.weight = embed_t
        self.embed_B.weight = embed_t
        self.embed_C.weight = embed_t
