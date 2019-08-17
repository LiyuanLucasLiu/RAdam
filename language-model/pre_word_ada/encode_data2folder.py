import pickle
import argparse
import os
import codecs
import random
import numpy as np

from tqdm import tqdm

import itertools
import functools

def encode_dataset(input_folder, w_map, reverse):

    w_eof = w_map['\n']
    w_unk = w_map['<unk>']

    list_dirs = os.walk(input_folder)

    lines = list()

    for root, dirs, files in list_dirs:
        for file in tqdm(files):
            with codecs.open(os.path.join(root, file), 'r', 'utf-8') as fin:
                lines = lines + list(filter(lambda t: t and not t.isspace(), fin.readlines()))

    dataset = list()
    for line in lines:
        dataset += list(map(lambda t: w_map.get(t, w_unk), line.split())) + [w_eof]

    if reverse:
        dataset = dataset[::-1]

    return dataset

def encode_dataset2file(input_folder, output_folder, w_map, reverse):

    w_eof = w_map['\n']
    w_unk = w_map['<unk>']

    list_dirs = os.walk(input_folder)

    range_ind = 0

    for root, dirs, files in list_dirs:
        for file in tqdm(files):
            with codecs.open(os.path.join(root, file), 'r', 'utf-8') as fin:
                lines = list(filter(lambda t: t and not t.isspace(), fin.readlines()))
            
            dataset = list()
            for line in lines:
                dataset += list(map(lambda t: w_map.get(t, w_unk), line.split())) + [w_eof]

            if reverse:
                dataset = dataset[::-1]

            with open(output_folder+'train_'+ str(range_ind) + '.pk', 'wb') as f:
                pickle.dump(dataset, f)

            range_ind += 1

    return range_ind

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_folder', default="/data/billionwords/1-billion-word-language-modeling-benchmark/training-monolingual.tokenized.shuffled")
    parser.add_argument('--test_folder', default="/data/billionwords/1-billion-word-language-modeling-benchmark/heldout-monolingual.tokenized.shuffled")
    parser.add_argument('--input_map', default="/data/billionwords/1b_map.pk")
    parser.add_argument('--output_folder', default="/data/billionwords/one_billion/")
    parser.add_argument('--threshold', type=int, default=3)
    parser.add_argument('--unk', default='<unk>')
    parser.add_argument('--reverse', action='store_true')
    args = parser.parse_args()

    with open(args.input_map, 'rb') as f:
        w_count = pickle.load(f)

    unk_count = sum([v for k, v in w_count.items() if v <= args.threshold])
    w_list = [(k, v) for k, v in w_count.items() if v > args.threshold]
    w_list.append(('<unk>', unk_count))
    w_list.sort(key=lambda t: t[1], reverse=True)
    w_map = {kv[0]:v for v, kv in enumerate(w_list)}

    range_ind = encode_dataset2file(args.train_folder, args.output_folder, w_map, args.reverse)

    test_dataset = encode_dataset(args.test_folder, w_map, args.reverse)

    with open(args.output_folder+'test.pk', 'wb') as f:
        pickle.dump({'w_map': w_map, 'test_data':test_dataset, 'range' : range_ind}, f)
