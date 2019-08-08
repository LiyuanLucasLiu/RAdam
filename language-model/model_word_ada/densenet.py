"""
.. module:: densenet
    :synopsis: vanilla dense RNN
 
.. moduleauthor:: Liyuan Liu
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import model_word_ada.utils as utils

class BasicUnit(nn.Module):
    def __init__(self, unit, input_dim, increase_rate, droprate):
        super(BasicUnit, self).__init__()

        rnnunit_map = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}

        self.unit = unit

        self.layer = rnnunit_map[unit](input_dim, increase_rate, 1)

        if 'lstm' == self.unit:
            utils.init_lstm(self.layer)

        self.droprate = droprate

        self.input_dim = input_dim
        self.increase_rate = increase_rate
        self.output_dim = input_dim + increase_rate

        self.init_hidden()

    def init_hidden(self):

        self.hidden_state = None

    def rand_ini(self):
        return

    def forward(self, x):

        if self.droprate > 0:
            new_x = F.dropout(x, p=self.droprate, training=self.training)
        else:
            new_x = x

        out, new_hidden = self.layer(new_x, self.hidden_state)

        self.hidden_state = utils.repackage_hidden(new_hidden)

        out = out.contiguous()

        return torch.cat([x, out], 2)

class DenseRNN(nn.Module):
    def __init__(self, layer_num, unit, emb_dim, hid_dim, droprate):
        super(DenseRNN, self).__init__()

        self.layer_list = [BasicUnit(unit, emb_dim + i * hid_dim, hid_dim, droprate) for i in range(layer_num)]
        self.layer = nn.Sequential(*self.layer_list)
        self.output_dim = self.layer_list[-1].output_dim

        self.init_hidden()

    def init_hidden(self):

        for tup in self.layer_list:
            tup.init_hidden()

    def rand_ini(self):

        for tup in self.layer_list:
            tup.rand_ini()

    def forward(self, x):
        return self.layer(x)