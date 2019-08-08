import pickle
import argparse
import os
import random
import numpy as np
from tqdm import tqdm

import itertools
import functools

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', default="/data/billionwords/1-billion-word-language-modeling-benchmark/training-monolingual.tokenized.shuffled")
    parser.add_argument('--output_map', default="/data/billionwords/1b_map.pk")
    args = parser.parse_args()

    w_count = {'\n':0}

    list_dirs = os.walk(args.input_folder)
    
    for root, dirs, files in list_dirs:
        for file in tqdm(files):
            with open(os.path.join(root, file)) as fin:
                for line in fin:
                    if not line or line.isspace():
                        continue
                    line = line.split()
                    for tup in line:
                        w_count[tup] = w_count.get(tup, 0) + 1
                    w_count['\n'] += 1

    with open(args.output_map, 'wb') as f:
        pickle.dump(w_count, f)