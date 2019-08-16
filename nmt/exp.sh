CUDA_VISIBLE_DEVICES=0 fairseq-train data-bin/iwslt14.tokenized.en-de -a transformer_iwslt_de_en --optimizer adam2 --lr 0.0003 -s en -t de --label-smoothing 0.1 --dropout 0.3 --max-tokens 4000 --warmup-init-lr 1e-8 --min-lr '1e-09' --lr-scheduler linear --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --max-update 70000 --warmup-updates 4000 --adam-betas '(0.9, 0.999)' --save-dir /cps/gadam/nmt/adam_warmup_f_0 --tb-tag adam_warmup_f_0 --user-dir ./my_module --restore-file x.pt

bash eval.sh /cps/gadam/nmt/adam_warmup_f_0 0 >> results_f_5.txt

CUDA_VISIBLE_DEVICES=0 fairseq-train data-bin/iwslt14.tokenized.en-de -a transformer_iwslt_de_en --optimizer adam2 --lr 0.0003 -s en -t de --label-smoothing 0.1 --dropout 0.3 --max-tokens 4000 --warmup-init-lr 1e-8 --min-lr '1e-09' --lr-scheduler linear --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --max-update 70000 --warmup-updates 4000 --adam-betas '(0.9, 0.999)' --save-dir /cps/gadam/nmt/adam_warmup_f_1 --tb-tag adam_warmup_f_1 --user-dir ./my_module --restore-file x.pt --seed 1111

bash eval.sh /cps/gadam/nmt/adam_warmup_f_1 0 >> results_f_5.txt

CUDA_VISIBLE_DEVICES=1 fairseq-train data-bin/iwslt14.tokenized.en-de -a transformer_iwslt_de_en --optimizer adam2 --lr 0.0003 -s en -t de --label-smoothing 0.1 --dropout 0.3 --max-tokens 4000 --warmup-init-lr 1e-8 --min-lr '1e-09' --lr-scheduler linear --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --max-update 70000 --warmup-updates 4000 --adam-betas '(0.9, 0.999)' --save-dir /cps/gadam/nmt/adam_warmup_f_2 --tb-tag adam_warmup_f_2 --user-dir ./my_module --restore-file x.pt --seed 2222

bash eval.sh /cps/gadam/nmt/adam_warmup_f_2 1 >> results_f_5.txt

CUDA_VISIBLE_DEVICES=1 fairseq-train data-bin/iwslt14.tokenized.en-de -a transformer_iwslt_de_en --optimizer adam2 --lr 0.0003 -s en -t de --label-smoothing 0.1 --dropout 0.3 --max-tokens 4000 --warmup-init-lr 1e-8 --min-lr '1e-09' --lr-scheduler linear --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --max-update 70000 --warmup-updates 4000 --adam-betas '(0.9, 0.999)' --save-dir /cps/gadam/nmt/adam_warmup_f_3 --tb-tag adam_warmup_f_3 --user-dir ./my_module --restore-file x.pt --seed 3333

bash eval.sh /cps/gadam/nmt/adam_warmup_f_3 1 >> results_f_5.txt

CUDA_VISIBLE_DEVICES=2 fairseq-train data-bin/iwslt14.tokenized.en-de -a transformer_iwslt_de_en --optimizer adam2 --lr 0.0003 -s en -t de --label-smoothing 0.1 --dropout 0.3 --max-tokens 4000 --warmup-init-lr 1e-8 --min-lr '1e-09' --lr-scheduler linear --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --max-update 70000 --warmup-updates 4000 --adam-betas '(0.9, 0.999)' --save-dir /cps/gadam/nmt/adam_warmup_f_4 --tb-tag adam_warmup_f_4 --user-dir ./my_module --restore-file x.pt --seed 4444

bash eval.sh /cps/gadam/nmt/adam_warmup_f_4 2 >> results_f_5.txt
