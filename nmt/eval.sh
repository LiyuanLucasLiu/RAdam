#!/bin/bash
echo "Model path" $SAVEDIR
GPUDEV=${2:-0}
SAVEDIR=${1}
MODELDIR=$SAVEDIR/model_ed.pt
if [ -f $MODELDIR  ]; then
    echo $MODELDIR "already exists"
else
    echo "Start averaging model"
    python average_checkpoints.py --inputs $SAVEDIR --num-epoch-checkpoints 10  --output $MODELDIR | grep 'Finish'
    echo "End averaging model"
fi

CUDA_VISIBLE_DEVICES=$GPUDEV fairseq-generate data-bin/iwslt14.tokenized.de-en \
                    --path $MODELDIR \
                    --batch-size 128 --beam 5 --remove-bpe \
                    --user-dir ./my_module 2>&1 | grep BLEU4

# CUDA_VISIBLE_DEVICES=$GPUDEV fairseq-generate data-bin/iwslt14.tokenized.en-de \
#                     --path $MODELDIR \
#                     --batch-size 128 --beam 5 --remove-bpe \
#                     --user-dir ./my_module 2>&1 | grep BLEU4
