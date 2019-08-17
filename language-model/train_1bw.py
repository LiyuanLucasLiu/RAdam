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
from model_word_ada.radam import RAdam
from model_word_ada.ldnet import LDRNN
from model_word_ada.densenet import DenseRNN
from model_word_ada.dataset import LargeDataset, EvalDataset
from model_word_ada.adaptive import AdaptiveSoftmax
import model_word_ada.utils as utils

# from tensorboardX import SummaryWriter
# writer = SummaryWriter(logdir='./cps/gadam/log_1bw_full/')

import argparse
import json
import os
import sys
import itertools
import functools

def evaluate(data_loader, lm_model, criterion, limited = 76800):
    print('evaluating')
    lm_model.eval()

    iterator = data_loader.get_tqdm()

    lm_model.init_hidden()
    total_loss = 0
    total_len = 0
    for word_t, label_t in iterator:
        label_t = label_t.view(-1)
        tmp_len = label_t.size(0)
        output = lm_model.log_prob(word_t)
        total_loss += tmp_len * utils.to_scalar(criterion(autograd.Variable(output), label_t))
        total_len += tmp_len

        if limited >=0 and total_len > limited:
            break

    ppl = math.exp(total_loss / total_len)
    print('PPL: ' + str(ppl))

    return ppl

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_folder', default='/data/billionwords/one_billion/')
    parser.add_argument('--load_checkpoint', default='')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--sequence_length', type=int, default=20)
    parser.add_argument('--hid_dim', type=int, default=2048)
    parser.add_argument('--word_dim', type=int, default=300)
    parser.add_argument('--label_dim', type=int, default=-1)
    parser.add_argument('--layer_num', type=int, default=2)
    parser.add_argument('--droprate', type=float, default=0.1)
    parser.add_argument('--add_relu', action='store_true')
    parser.add_argument('--layer_drop', type=float, default=0.5)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=14)
    parser.add_argument('--clip', type=float, default=5)
    parser.add_argument('--update', choices=['Adam', 'Adagrad', 'Adadelta', 'RAdam', 'SGD'], default='Adam')
    parser.add_argument('--rnn_layer', choices=['Basic', 'DDNet', 'DenseNet', 'LDNet'], default='Basic')
    parser.add_argument('--rnn_unit', choices=['gru', 'lstm', 'rnn', 'bnlstm'], default='lstm')
    parser.add_argument('--lr', type=float, default=-1)
    parser.add_argument('--lr_decay', type=lambda t: [int(tup) for tup in t.split(',')], default=[8])
    parser.add_argument('--cut_off', nargs='+', default=[4000,40000,200000])
    parser.add_argument('--interval', type=int, default=100)
    parser.add_argument('--check_interval', type=int, default=4000)
    parser.add_argument('--checkpath', default='./cps/gadam/')
    parser.add_argument('--model_name', default='adam')
    args = parser.parse_args()

    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)

    print('loading dataset')
    dataset = pickle.load(open(args.dataset_folder + 'test.pk', 'rb'))
    w_map, test_data, range_idx = dataset['w_map'], dataset['test_data'], dataset['range']

    cut_off = args.cut_off + [len(w_map) + 1]

    train_loader = LargeDataset(args.dataset_folder, range_idx, args.batch_size, args.sequence_length)
    test_loader = EvalDataset(test_data, args.batch_size)

    print('building model')

    rnn_map = {'Basic': BasicRNN, 'DDNet': DDRNN, 'DenseNet': DenseRNN, 'LDNet': functools.partial(LDRNN, layer_drop = args.layer_drop)}
    rnn_layer = rnn_map[args.rnn_layer](args.layer_num, args.rnn_unit, args.word_dim, args.hid_dim, args.droprate)

    if args.label_dim > 0:
        soft_max = AdaptiveSoftmax(args.label_dim, cut_off)
    else:
        soft_max = AdaptiveSoftmax(rnn_layer.output_dim, cut_off)

    lm_model = LM(rnn_layer, soft_max, len(w_map), args.word_dim, args.droprate, label_dim = args.label_dim, add_relu=args.add_relu)
    lm_model.rand_ini()
    # lm_model.cuda()

    optim_map = {'Adam' : optim.Adam, 'Adagrad': optim.Adagrad, 'Adadelta': optim.Adadelta, 'RAdam': RAdam, 'SGD': functools.partial(optim.SGD, momentum=0.9)}
    if args.lr > 0:
        optimizer=optim_map[args.update](lm_model.parameters(), lr=args.lr)
    else:
        optimizer=optim_map[args.update](lm_model.parameters())

    if args.load_checkpoint:
        if os.path.isfile(args.load_checkpoint):
            print("loading checkpoint: '{}'".format(args.load_checkpoint))
            checkpoint_file = torch.load(args.load_checkpoint, map_location=lambda storage, loc: storage)
            lm_model.load_state_dict(checkpoint_file['lm_model'], False)
            optimizer.load_state_dict(checkpoint_file['opt'], False)
        else:
            print("no checkpoint found at: '{}'".format(args.load_checkpoint))

    test_lm = nn.NLLLoss()
    
    test_lm.cuda()
    lm_model.cuda()
    
    batch_index = 0
    epoch_loss = 0
    full_epoch_loss = 0
    best_train_ppl = float('inf')
    cur_lr = args.lr

    try:
        for indexs in range(args.epoch):

            print('#' * 89)
            print('Start: {}'.format(indexs))
            
            iterator = train_loader.get_tqdm()
            full_epoch_loss = 0

            lm_model.train()

            for word_t, label_t in iterator:

                if 1 == train_loader.cur_idx:
                    lm_model.init_hidden()

                label_t = label_t.view(-1)

                lm_model.zero_grad()
                loss = lm_model(word_t, label_t)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm(lm_model.parameters(), args.clip)
                optimizer.step()

                batch_index += 1 

                if 0 == batch_index % args.interval:
                    s_loss = utils.to_scalar(loss)
                    # writer.add_scalars('loss_tracking/train_loss', {args.model_name:s_loss}, batch_index)
                
                epoch_loss += utils.to_scalar(loss)
                full_epoch_loss += utils.to_scalar(loss)
                if 0 == batch_index % args.check_interval:
                    epoch_ppl = math.exp(epoch_loss / args.check_interval)
                    # writer.add_scalars('loss_tracking/train_ppl', {args.model_name: epoch_ppl}, batch_index)
                    print('epoch_ppl: {} lr: {} @ batch_index: {}'.format(epoch_ppl, cur_lr, batch_index))
                    epoch_loss = 0

            if indexs in args.lr_decay and cur_lr > 0:
                cur_lr *= 0.1
                print('adjust_learning_rate...')
                utils.adjust_learning_rate(optimizer, cur_lr)
    
            test_ppl = evaluate(test_loader, lm_model, test_lm, -1)

            # writer.add_scalars('loss_tracking/test_ppl', {args.model_name: test_ppl}, indexs)
            print('test_ppl: {} @ index: {}'.format(test_ppl, indexs))
        
            torch.save({'lm_model': lm_model.state_dict(), 'opt':optimizer.state_dict()}, args.checkpath+args.model_name+'.model')

    except KeyboardInterrupt:

        print('Exiting from training early')
        test_ppl = evaluate(test_loader, lm_model, test_lm, -1)
        # writer.add_scalars('loss_tracking/test_ppl', {args.model_name: test_ppl}, args.epoch)

        torch.save({'lm_model': lm_model.state_dict(), 'opt':optimizer.state_dict()}, args.checkpath+args.model_name+'.model')

    # writer.close()
