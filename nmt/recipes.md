
# Adam with warmup

```
CUDA_VISIBLE_DEVICES=0 fairseq-train data-bin/iwslt14.tokenized.de-en -a transformer_iwslt_de_en --optimizer adam2 --lr 0.0003 -s de -t en --label-smoothing 0.1 --dropout 0.3 --max-tokens 4000 --warmup-init-lr 1e-8 --min-lr '1e-09' --lr-scheduler linear --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --max-update 70000 --warmup-updates 4000 --adam-betas '(0.9, 0.999)' --save-dir /cps/gadam/nmt/adam_warmup_f_0 --tb-tag adam_warmup_f_0 --user-dir ./my_module --restore-file x.pt

bash eval.sh /cps/gadam/nmt/adam_warmup_f_0 0 >> results_f_5.txt

for SEED in 1111 2222 3333 4444
do

    CUDA_VISIBLE_DEVICES=0 fairseq-train data-bin/iwslt14.tokenized.de-en -a transformer_iwslt_de_en --optimizer adam2 --lr 0.0003 -s de -t en --label-smoothing 0.1 --dropout 0.3 --max-tokens 4000 --warmup-init-lr 1e-8 --min-lr '1e-09' --lr-scheduler linear --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --max-update 70000 --warmup-updates 4000 --adam-betas '(0.9, 0.999)' --save-dir /cps/gadam/nmt/adam_warmup_f_$SEED --tb-tag adam_warmup_f_$SEED --user-dir ./my_module --restore-file x.pt --seed $SEED

    bash eval.sh /cps/gadam/nmt/adam_warmup_f_$SEED 0 >> results_f_5.txt
done
```

# Adam-2k

```
CUDA_VISIBLE_DEVICES=0 fairseq-train data-bin/iwslt14.tokenized.de-en -a transformer_iwslt_de_en --optimizer adam2 --lr 0.0003086 -s de -t en --label-smoothing 0.1 --dropout 0.3 --max-tokens 4000 --min-lr '1e-09' --lr-scheduler linear --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --max-update 72000 --warmup-updates 1 --adam-betas '(0.9, 0.999)' --save-dir /cps/gadam/nmt/adam_1k --tb-tag adam_1k --user-dir ./my_module --fp16 --restore-file x.pt --adam-freeze 2000
```

# Adam-eps

```
CUDA_VISIBLE_DEVICES=0 fairseq-train data-bin/iwslt14.tokenized.de-en -a transformer_iwslt_de_en --optimizer adam2 --lr 0.0003 -s de -t en --label-smoothing 0.1 --dropout 0.3 --max-tokens 4000 --min-lr '1e-09' --lr-scheduler linear --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --max-update 70000 --warmup-updates 1 --adam-betas '(0.9, 0.999)' --save-dir /cps/gadam/nmt/adam_eps --tb-tag adam_eps --user-dir ./my_module --fp16 --adam-eps 1e-4 --restore-file x.pt

```

# RAdam

```
CUDA_VISIBLE_DEVICES=0 fairseq-train data-bin/iwslt14.tokenized.de-en -a transformer_iwslt_de_en --optimizer radam --lr 0.0003 -s de -t en --label-smoothing 0.1 --dropout 0.3 --max-tokens 4000 --min-lr '1e-09' --lr-scheduler linear --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --max-update 70000 --warmup-updates 1 --adam-betas '(0.9, 0.999)' --save-dir /cps/gadam/nmt/radam_0 --tb-tag radam_0 --user-dir ./my_module --fp16 --restore-file x.pt

bash eval.sh /cps/gadam/nmt/radam_0 0 >> results_f_5.txt

for SEED in 1111 2222 3333 4444
do
    CUDA_VISIBLE_DEVICES=0 fairseq-train data-bin/iwslt14.tokenized.de-en -a transformer_iwslt_de_en --optimizer radam --lr 0.0003 -s de -t en --label-smoothing 0.1 --dropout 0.3 --max-tokens 4000 --min-lr '1e-09' --lr-scheduler linear --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --max-update 70000 --warmup-updates 1 --adam-betas '(0.9, 0.9995)' --save-dir /cps/gadam/nmt/radam_$SEED --tb-tag radam_$SEED --user-dir ./my_module --fp16 --restore-file x.pt --seed $SEED

    bash eval.sh /cps/gadam/nmt/radam_$SEED 0 >> results_f_5.txt
done
```

# Novograd
We also implemented [novograd](https://arxiv.org/pdf/1905.11286.pdf), which claims no warmup is requried. 
We tried the following settings with with lr=0.03, 0.0003, 0.00003, 0.00001, none of these works without warmup. 

```
CUDA_VISIBLE_DEVICES=0 fairseq-train ./data-bin/iwslt14.tokenized.de-en -a transformer_iwslt_de_en --optimizer novograd --lr 0.0003 -s en -t de --label-smoothing 0.1 --dropout 0.3 --max-tokens 4000 --min-lr '1e-09' --lr-scheduler poly --weight-decay 5e-5 --criterion label_smoothed_cross_entropy --max-update 70000  --adam-betas '(0.9, 0.999)' --save-dir /ckp/nmt/novograd --tb-tag novograd --user-dir ./my_module --fp16 --restore-file x.pt
```
