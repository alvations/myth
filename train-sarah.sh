#!/bin/bash

SRC=$1           # en
TRG=$2           # ja
RANDSEED=$3      # 42
DOMAIN=$4        # jparacrawl
LAYERS=$5        # 6
HEADS=$6         # 8
DIMEMB=$7        # 1024
DIMTRA=$8        # 4096
VOCABSIZE=$9     # 32000
LR=${11}         # 0.0001
DROPOUT=${12}    # 0.1

MODELDIR=/path/to/modeldir/$DOMAIN-$LAYERS-$HEADS-$DIMEMB-$DIMTRA-$VOCABSIZE-$LR-$DROPOUT/$SRC-$TRG-r$RANDSEED

mkdir -p $MODELDIR

TRAIN_SRC=/path/to/jparacrawl/$DOMAIN/$SRC-$TRG/train.$SRC
TRAIN_TRG=/path/to/jparacrawl/$DOMAIN/$SRC-$TRG/train.$TRG
VALID_SRC=/path/to/jparacrawl/$DOMAIN/$SRC-$TRG/valid.$SRC
VALID_TRG=/path/to/jparacrawl/$DOMAIN/$SRC-$TRG/valid.$TRG
TRAINLOG=$MODELDIR/train.log
VALIDLOG=$MODELDIR/valid.log

GPUS="0 1 2 3"
WORKSPACE=8500  # Assumes 11GB RAM on GPU

MARIAN=$HOME/marian-dev/build/marian

$MARIAN --model $MODELDIR/model.npz --type transformer \
--train-sets $TRAIN_SRC $TRAIN_TRG \
--vocabs $MODELDIR/vocab.src.spm $MODELDIR/vocab.trg.spm \
--dim-vocabs $VOCABSIZE $VOCABSIZE \
--mini-batch-fit \
--valid-freq 500 --save-freq 500 --disp-freq 50 \
--valid-metrics ce-mean-words perplexity  \
--valid-sets $VALID_SRC $VALID_TRG \
--quiet-translation \
--beam-size 12 --normalize=0.6 \
--valid-mini-batch 16 \
--early-stopping 5 --cost-type=ce-mean-words \
--log $TRAINLOG --valid-log $VALIDLOG \
--enc-depth $LAYERS --dec-depth $LAYERS \
--transformer-preprocess n --transformer-postprocess da \
--tied-embeddings-all --dim-emb $DIMEMB --transformer-dim-ffn $DIMTRA \
--transformer-dropout $DROPOUT --transformer-dropout-attention $DROPOUT \
--transformer-dropout-ffn $DROPOUT --label-smoothing $DROPOUT \
--learn-rate $LR \
--lr-warmup 8000 --lr-decay-inv-sqrt 8000 --lr-report \
--optimizer-params 0.9 0.98 1e-09 --clip-norm 5 \
--devices $GPUS --workspace $WORKSPACE  --optimizer-delay 2 --sync-sgd --seed $RANDSEED \
--exponential-smoothing \
--shuffle-in-ram
