#!/usr/bin/python
from utils import *

# toy for sim
#train_q, train_a, train_sim = load_from_file("./pkl/train_pair.pkl")
#train_pairs = (train_q[:100], train_a[:100], train_sim[:100])
#dump_to_file(train_pairs, "./pkl/train_toy_pair.pkl")

# toy for reader
train_q, train_w, train_e_p, train_a = load_from_file("./pkl/reader/100_raw_clear/train_pair.pkl")
dev_q, dev_w, dev_e_p, dev_a = load_from_file("./pkl/reader/100_raw_clear/dev_pair.pkl")
train_pairs = (train_q[:100], train_w[:100] , train_e_p[:100], train_a[:100])
dev_pairs = (dev_q[:100], dev_w[:100], dev_e_p[:100], dev_a[:100])
dump_to_file(train_pairs, "./pkl/toy/reader/train_pair.pkl")
dump_to_file(dev_pairs, "./pkl/toy/reader/dev_pair.pkl")
