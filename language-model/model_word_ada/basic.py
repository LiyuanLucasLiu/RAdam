import torch
import torch.nn as nn
import torch.nn.functional as F
import model_word_ada.utils as utils
from model_word_ada.bnlstm import BNLSTM

class BasicUnit(nn.Module):
    def __init__(self, unit, input_dim, hid_dim, droprate):
        super(BasicUnit, self).__init__()

        rnnunit_map = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU, 'bnlstm': BNLSTM}

        self.batch_norm = (unit == 'bnlstm')

        self.layer = rnnunit_map[unit](input_dim, hid_dim, 1)
        self.droprate = droprate

        self.output_dim = hid_dim

        self.init_hidden()

    def init_hidden(self):

        self.hidden_state = None

    def rand_ini(self):

        if not self.batch_norm:
            utils.init_lstm(self.layer)

    def forward(self, x):
        # set_trace()
        out, new_hidden = self.layer(x, self.hidden_state)

        self.hidden_state = utils.repackage_hidden(new_hidden)
        
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)

        return out

class BasicRNN(nn.Module):
    def __init__(self, layer_num, unit, emb_dim, hid_dim, droprate):
        super(BasicRNN, self).__init__()

        layer_list = [BasicUnit(unit, emb_dim, hid_dim, droprate)] + [BasicUnit(unit, hid_dim, hid_dim, droprate) for i in range(layer_num - 1)]
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
        return self.layer(x)