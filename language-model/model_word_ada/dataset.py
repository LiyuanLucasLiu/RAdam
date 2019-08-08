import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import sys
import pickle
import random
from tqdm import tqdm

from torch.utils.data import Dataset

class EvalDataset(object):

    def __init__(self, dataset, sequence_length):
        super(EvalDataset, self).__init__()
        self.dataset = dataset

        self.sequence_length = sequence_length

        self.construct_index()

    def get_tqdm(self):
        return tqdm(self, mininterval=2, total=self.index_length, leave=False, file=sys.stdout, ncols=80)

    def construct_index(self):
        token_per_batch = self.sequence_length
        tot_num = len(self.dataset) - 1
        res_num = tot_num - tot_num % token_per_batch

        self.x = list(torch.unbind(torch.LongTensor(self.dataset[0:res_num]).view(-1, self.sequence_length), 0))
        self.y = list(torch.unbind(torch.LongTensor(self.dataset[1:res_num+1]).view(-1, self.sequence_length), 0))

        self.x.append(torch.LongTensor(self.dataset[res_num:tot_num]))
        self.y.append(torch.LongTensor(self.dataset[res_num+1:tot_num+1]))

        self.index_length = len(self.x)
        self.cur_idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.cur_idx == self.index_length:
            self.cur_idx = 0
            raise StopIteration

        word_t = autograd.Variable(self.x[self.cur_idx]).cuda().view(-1, 1)
        label_t = autograd.Variable(self.y[self.cur_idx]).cuda().view(-1, 1)

        self.cur_idx += 1
        
        return word_t, label_t

# class SmallDataset(object):

#     def __init__(self, dataset, batch_size, sequence_length):
#         super(SmallDataset, self).__init__()
#         self.dataset = dataset

#         self.batch_size = batch_size
#         self.sequence_length = sequence_length

#         self.construct_index()

#     def get_tqdm(self):
#         return tqdm(self, mininterval=2, total=self.index_length, leave=False, file=sys.stdout)

#     def construct_index(self):
#         token_per_batch = self.batch_size * self.sequence_length
#         res_num = len(self.dataset) - 1
#         res_num = res_num - res_num % token_per_batch

#         self.x = torch.LongTensor(self.dataset[0:res_num]).view(self.batch_size, -1, self.sequence_length).transpose_(0, 1).contiguous()
#         self.y = torch.LongTensor(self.dataset[1:res_num+1]).view(self.batch_size, -1, self.sequence_length).transpose_(0, 1).contiguous()

#         self.index_length = self.x.size(0)
#         self.cur_idx = 0

#     def __iter__(self):
#         return self

#     def __next__(self):
#         if self.cur_idx == self.index_length:
#             self.cur_idx = 0
#             raise StopIteration

#         word_t = autograd.Variable(self.x[self.cur_idx]).cuda()
#         label_t = autograd.Variable(self.y[self.cur_idx]).cuda()

#         self.cur_idx += 1
        
#         return word_t, label_t

class LargeDataset(object):

    def __init__(self, root, range_idx, batch_size, sequence_length):
        super(LargeDataset, self).__init__()
        self.root = root
        self.range_idx = range_idx
        self.shuffle_list = list(range(0, range_idx))
        self.shuffle()

        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.token_per_batch = self.batch_size * self.sequence_length

        self.total_batch_num = -1

    def shuffle(self):
        random.shuffle(self.shuffle_list)

    def get_tqdm(self):
        self.batch_count = 0

        if self.total_batch_num <= 0:
            return tqdm(self, mininterval=2, leave=False, file=sys.stdout).__iter__()
        else:
            return tqdm(self, mininterval=2, total=self.total_batch_num, leave=False, file=sys.stdout, ncols=80).__iter__()

    def __iter__(self):
        self.cur_idx = 0
        self.file_idx = 0
        self.index_length = 0
        return self

    def __next__(self):
        if self.cur_idx >= self.index_length:
            self.open_next()

        word_t = autograd.Variable(self.x[self.cur_idx]).cuda()
        # label_t = autograd.Variable(self.y[self.cur_idx]).cuda()
        label_t = self.y[self.cur_idx].cuda()

        self.cur_idx += 1

        return word_t, label_t

    def open_next(self):
        if self.file_idx >= self.range_idx:
            self.total_batch_num = self.batch_count
            self.shuffle()
            raise StopIteration

        self.dataset = pickle.load(open(self.root + 'train_' + str( self.shuffle_list[self.file_idx])+'.pk', 'rb'))

        res_num = len(self.dataset) - 1
        res_num = res_num - res_num % self.token_per_batch

        self.x = torch.LongTensor(self.dataset[0:res_num]).view(self.batch_size, -1, self.sequence_length).transpose_(0, 1).transpose_(1, 2).contiguous()
        self.y = torch.LongTensor(self.dataset[1:res_num+1]).view(self.batch_size, -1, self.sequence_length).transpose_(0, 1).transpose_(1, 2).contiguous()

        self.index_length = self.x.size(0)
        self.cur_idx = 0

        self.batch_count += self.index_length
        self.file_idx += 1