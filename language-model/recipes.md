# Pre-process

```
python pre_word_ada/gene_map.py --input_folder /data/billionwords/1-billion-word-language-modeling-benchmark/training-monolingual.tokenized.shuffled --output_map /data/billionwords/1b_map.pk

python pre_word_ada/encode_data2folder.py --train_folder /data/billionwords/1-billion-word-language-modeling-benchmark/training-monolingual.tokenized.shuffled --test_folder /data/billionwords/1-billion-word-language-modeling-benchmark/heldout-monolingual.tokenized.shuffled --input_map /data/billionwords/1b_map.pk --output_folder /data/billionwords/one_billion/
```

# Training

## Adam
```
python train_1bw.py --dataset_folder /data/billionwords/one_billion/ --lr 0.001 --checkpath ./cps/gadam/ --model_name adam --update Adam
```

## RAdam
```
python train_1bw.py --dataset_folder /data/billionwords/one_billion/ --lr 0.001 --checkpath ./cps/gadam/ --model_name radam --update RAdam
```
