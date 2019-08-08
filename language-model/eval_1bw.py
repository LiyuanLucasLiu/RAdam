from __future__ import print_function
import datetime
import time
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import codecs
import pickle
import math

from model_word_ada.LM import LM
from model_word_ada.basic import BasicRNN
from model_word_ada.ddnet import DDRNN
from model_word_ada.densenet import DenseRNN
from model_word_ada.ldnet import LDRNN
from model_word_ada.dataset import LargeDataset, EvalDataset
from model_word_ada.adaptive import AdaptiveSoftmax
import model_word_ada.utils as utils

# from tensorboardX import SummaryWriter

import argparse
import json
import os
import sys
import itertools
import functools

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_folder', default='/data/billionwords/one_billion/')
    parser.add_argument('--load_checkpoint', default='./checkpoint/basic_1.model')
    parser.add_argument('--sequence_length', type=int, default=600)
    parser.add_argument('--hid_dim', type=int, default=2048)
    parser.add_argument('--word_dim', type=int, default=300)
    parser.add_argument('--label_dim', type=int, default=-1)
    parser.add_argument('--layer_num', type=int, default=2)
    parser.add_argument('--droprate', type=float, default=0.1)
    parser.add_argument('--add_relu', action='store_true')
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--rnn_layer', choices=['Basic', 'DDNet', 'DenseNet', 'LDNet'], default='Basic')
    parser.add_argument('--rnn_unit', choices=['gru', 'lstm', 'rnn'], default='lstm')
    parser.add_argument('--cut_off', nargs='+', default=[4000,40000,200000])
    parser.add_argument('--limit', type=int, default=76800)
    
    args = parser.parse_args()

    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)

    print('loading dataset')
    dataset = pickle.load(open(args.dataset_folder + 'test.pk', 'rb'))
    w_map, test_data = dataset['w_map'], dataset['test_data']

    cut_off = args.cut_off + [len(w_map) + 1]

    test_loader = EvalDataset(test_data, args.sequence_length)

    print('building model')

    rnn_map = {'Basic': BasicRNN, 'DDNet': DDRNN, 'DenseNet': DenseRNN, 'LDNet': functools.partial(LDRNN, layer_drop = 0)}
    rnn_layer = rnn_map[args.rnn_layer](args.layer_num, args.rnn_unit, args.word_dim, args.hid_dim, args.droprate)

    if args.label_dim > 0:
        soft_max = AdaptiveSoftmax(args.label_dim, cut_off)
    else:
        soft_max = AdaptiveSoftmax(rnn_layer.output_dim, cut_off)

    lm_model = LM(rnn_layer, soft_max, len(w_map), args.word_dim, args.droprate, label_dim = args.label_dim, add_relu = args.add_relu)
    lm_model.cuda()

    if os.path.isfile(args.load_checkpoint):
        print("loading checkpoint: '{}'".format(args.load_checkpoint))

        checkpoint_file = torch.load(args.load_checkpoint, map_location=lambda storage, loc: storage)
        lm_model.load_state_dict(checkpoint_file['lm_model'])
    else:
        print("no checkpoint found at: '{}'".format(args.load_checkpoint))

    test_lm = nn.NLLLoss()
    
    test_lm.cuda()
    lm_model.cuda()

    print('evaluating')
    lm_model.eval()

    iterator = test_loader.get_tqdm()

    lm_model.init_hidden()
    total_loss = 0
    total_len = 0
    for word_t, label_t in iterator:
        label_t = label_t.view(-1)
        tmp_len = label_t.size(0)
        output = lm_model.log_prob(word_t)
        total_loss += tmp_len * utils.to_scalar(test_lm(autograd.Variable(output), label_t))
        total_len += tmp_len

        if args.limit > 0 and total_len > args.limit:
            break

    print(str(total_loss / total_len))
    ppl = math.exp(total_loss / total_len)
    print('PPL: ' + str(ppl))
