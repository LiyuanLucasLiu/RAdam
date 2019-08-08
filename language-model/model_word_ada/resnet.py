# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import model.utils as utils

# class BasicUnit(nn.Module):
#     def __init__(self, unit, input_dim, hid_dim, droprate):
#         super(BasicUnit, self).__init__()

#         rnnunit_map = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}
#         self.unit_number = unit_number

#         self.layer = rnnunit_map[unit](input_dim, hid_dim, 1)

#         self.droprate = droprate

#         self.output_dim = input_dim + hid_dim

#         self.init_hidden()

#     def init_hidden(self):

#         self.hidden_list = [None for i in range(unit_number)]

#     def rand_ini(self):

#         for cur_lstm in self.unit_list:
#             utils.init_lstm(cur_lstm)

#     def forward(self, x):

#         out, _ = self.layer(x)
        
#         if self.droprate > 0:
#             out = F.dropout(out, p=self.droprate, training=self.training)
        
#         return toch.cat([x, out], 2)

# class DenseRNN(nn.Module):
#     def __init__(self, layer_num, unit, emb_dim, hid_dim, droprate):
#         super(DenseRNN, self).__init__()

#         self.layer = nn.Sequential([BasicUnit(unit, emb_dim + i * hid_dim, hid_dim, droprate) for i in range(layer_num) ])

#         self.output_dim = self.layer[-1].output_dim

#         self.init_hidden()

#     def init_hidden(self):
#         self.layer.apply(lambda t: t.init_hidden())

#     def rand_ini(self):
#         self.layer.apply(lambda t: t.rand_ini())

#     def forward(self, x):
#         return self.layer(x)