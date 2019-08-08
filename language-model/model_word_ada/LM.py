import torch
import torch.nn as nn
import torch.nn.functional as F
import model_word_ada.utils as utils

class LM(nn.Module):

    def __init__(self, rnn, soft_max, w_num, w_dim, droprate, label_dim = -1, add_relu=False):
        super(LM, self).__init__()

        self.rnn = rnn
        self.soft_max = soft_max

        if soft_max:
            self.forward = self.softmax_forward
        else:
            self.forward = self.embed_forward

        self.w_num = w_num
        self.w_dim = w_dim
        self.word_embed = nn.Embedding(w_num, w_dim)

        self.rnn_output = self.rnn.output_dim

        self.add_proj = label_dim > 0
        if self.add_proj:
            self.project = nn.Linear(self.rnn_output, label_dim)
            if add_relu:
                self.relu = nn.ReLU()
            else:
                self.relu = lambda x: x

        self.drop = nn.Dropout(p=droprate)

    def load_embed(self, origin_lm):
        self.word_embed = origin_lm.word_embed
        self.soft_max = origin_lm.soft_max

    def rand_ini(self):
        
        self.rnn.rand_ini()
        # utils.init_linear(self.project)
        self.soft_max.rand_ini()
        # if not self.tied_weight:
        utils.init_embedding(self.word_embed.weight)

        if self.add_proj:
            utils.init_linear(self.project)

    def init_hidden(self):
        self.rnn.init_hidden()

    def softmax_forward(self, w_in, target):

        w_emb = self.word_embed(w_in)
        
        w_emb = self.drop(w_emb)

        out = self.rnn(w_emb).contiguous().view(-1, self.rnn_output)

        if self.add_proj:
            out = self.drop(self.relu(self.project(out)))
            # out = self.drop(self.project(out))

        out = self.soft_max(out, target)

        return out

    def embed_forward(self, w_in, target):

        w_emb = self.word_embed(w_in)
        
        w_emb = self.drop(w_emb)

        out = self.rnn(w_emb).contiguous().view(-1, self.rnn_output)

        if self.add_proj:
            out = self.drop(self.relu(self.project(out)))
            # out = self.drop(self.project(out))

        out = self.soft_max(out, target)

        return out

    def log_prob(self, w_in):

        w_emb = self.word_embed(w_in)
        
        out = self.rnn(w_emb).contiguous().view(-1, self.rnn_output)

        if self.add_proj:
            out = self.relu(self.project(out))

        out = self.soft_max.log_prob(out)

        return out