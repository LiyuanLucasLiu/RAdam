import torch
import torch.nn as nn
import torch.nn.functional as F
import model_word_ada.utils as utils
from model_word_ada.bnlstm import BNLSTM

class BasicUnit(nn.Module):
    def __init__(self, unit, unit_number, emb_dim, hid_dim, droprate):
        super(BasicUnit, self).__init__()

        rnnunit_map = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU, 'bnlstm': BNLSTM}

        self.batch_norm = (unit == 'bnlstm')

        self.unit_number = unit_number
        # self.unit_weight = nn.Parameter(torch.FloatTensor([1] * unit_number))

        self.unit_list = nn.ModuleList()
        self.unit_list.append(rnnunit_map[unit](emb_dim, hid_dim, 1))
        if unit_number > 1:
            self.unit_list.extend([rnnunit_map[unit](hid_dim, hid_dim, 1) for ind in range(unit_number - 1)])

        self.droprate = droprate

        self.output_dim = emb_dim + hid_dim * unit_number

        self.init_hidden()

    def init_hidden(self):

        self.hidden_list = [None for i in range(self.unit_number)]

    def rand_ini(self):

        if not self.batch_norm:
            for cur_lstm in self.unit_list:
                utils.init_lstm(cur_lstm)

    def forward(self, x):

        out = 0
        # n_w = F.softmax(self.unit_weight, dim=0)
        for ind in range(self.unit_number):
            nout, new_hidden = self.unit_list[ind](x[ind], self.hidden_list[ind])
            self.hidden_list[ind] = utils.repackage_hidden(new_hidden)
            out = out + nout
            # out = out + n_w[ind] * self.unit_number * nout

        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        
        x.append(out)

        return x

class DDRNN(nn.Module):
    def __init__(self, layer_num, unit, emb_dim, hid_dim, droprate):
        super(DDRNN, self).__init__()

        layer_list = [BasicUnit(unit, i + 1, emb_dim, hid_dim, droprate) for i in range(layer_num)]
        self.layer = nn.Sequential(*layer_list)
        self.output_dim = layer_list[-1].output_dim

        self.init_hidden()

    def init_hidden(self):

        for tup in self.layer.children():
            tup.init_hidden()

    def rand_ini(self):

        for tup in self.layer.children():
            tup.rand_ini()

    def forward(self, x):
        out = self.layer([x])
        return torch.cat(out, 2)