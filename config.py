from __future__ import division

class Config():
    def __init__(self):
        self.n_embed = 186841
        self.d_embed = 300
        self.sent_len = 20
        self.win_len = 7 + 3

        self.ir_size = 1000
        self.filter_size = 100

        self.batch_size = 128
        self.epoch = 100
        self.lr = 0.005
        self.l2 = 0.00001
        self.valid_every = self.batch_size * 100

        self.margin = 0.2

        self.rnn_fea_size = self.d_embed

        #model file
        self.pre_embed_file = "./model/{}/embedding.pre".format(self.d_embed)
        self.reader_model = "./model/reader_{}.torch".format(self.d_embed)

        self.title_dict = "./pkl/dict/title.dict"
        self.entity_dict = "./pkl/dict/entity.dict"

        #data dir
        self.data_dir = "./data"

        #memory network
        self.hop = 2

        # RL
        self.K = 1


config = Config()
