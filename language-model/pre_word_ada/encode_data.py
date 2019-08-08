import pickle
import argparse
import os
import random
import numpy as np

from tqdm import tqdm

import itertools
import functools

def encode_dataset(input_file, w_map):

    w_eof = w_map['\n']

    with open(input_file, 'r') as fin:
        lines = list(filter(lambda t: t and not t.isspace(), fin.readlines()))
    
    lines = [tup.split() for tup in lines]

    dataset = list()

    for line in lines:

        dataset += list(map(lambda t: w_map[t], line)) + [w_eof]

    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', default="./data/ptb_train/ptb.train.txt")
    parser.add_argument('--test_file', default="./data/ptb_test/ptb.test.txt")
    parser.add_argument('--dev_file', default="./data/ptb_valid/ptb.valid.txt")
    parser.add_argument('--input_map', default="./data/ptb_map.pk")
    parser.add_argument('--output_file', default="./data/ptb_dataset.pk")
    args = parser.parse_args()

    with open(args.input_map, 'rb') as f:
        w_count = pickle.load(f)

    w_list = list(w_count.items())
    w_list.sort(key=lambda t: t[1], reverse=True)
    w_map = {kv[0]:v for v, kv in enumerate(w_list)}

    train_dataset = encode_dataset(args.train_file, w_map)
    test_dataset = encode_dataset(args.test_file, w_map)
    dev_dataset = encode_dataset(args.dev_file, w_map)

    with open(args.output_file, 'wb') as f:
        pickle.dump({'w_map': w_map, 'train_data': train_dataset, 'test_data':test_dataset, 'dev_data': dev_dataset}, f)