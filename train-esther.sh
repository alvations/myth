#!/bin/bash

SRC=fr           # en
TRG=xx           # ja
RANDSEED=5      # 42
DOMAIN=transliterate        # jparacrawl
ELAYERS=1       # 6
DLAYERS=1
HEADS=1         # 8
DIMEMB=1024        # 1024
DIMTRA=4096      # 4096
VOCABSIZE=8000     # 32000
LR=0.0001         # 0.0001
DROPOUT=0.8    # 0.1

MODELDIR=$HOME/stash/fdi-models/fdi-$ELAYERS+$DLAYERS-$HEADS-$DIMEMB-$DIMTRA-$VOCABSIZE-$LR-$DROPOUT/$SRC-$TRG-r$RANDSEED/

mkdir -p $MODELDIR

cp $HOME/stash/fdi-models/vocab.src.spm $MODELDIR/

DATADIR=$HOME/stash/fdi-data

TRAIN_SRC=$DATADIR/train.$SRC-$TRG.$SRC
TRAIN_TRG=$DATADIR/train.$SRC-$TRG.$TRG
VALID_SRC=$DATADIR/valid.$SRC-$TRG.$SRC
VALID_TRG=$DATADIR/valid.$SRC-$TRG.$TRG
TRAINLOG=$MODELDIR/train.log
VALIDLOG=$MODELDIR/valid.log

GPUS="0"
WORKSPACE=43185  # Assumes 11GB RAM on GPU

MARIAN=$HOME/marian/build/marian

$MARIAN --model $MODELDIR/model.npz --type transformer \
--train-sets $TRAIN_SRC $TRAIN_TRG \
--vocabs $MODELDIR/vocab.src.spm $MODELDIR/vocab.src.spm \
--dim-vocabs $VOCABSIZE $VOCABSIZE \
--valid-freq 500 --save-freq 500 --disp-freq 00 \
--valid-metrics ce-mean-words perplexity  \
--valid-sets $VALID_SRC $VALID_TRG \
--quiet-translation \
--beam-size 12 --normalize=0.6 \
--valid-mini-batch 16 \
--early-stopping 5 --cost-type=ce-mean-words \
--log $TRAINLOG --valid-log $VALIDLOG \
--enc-depth $ELAYERS --dec-depth $DLAYERS \
--transformer-preprocess n --transformer-postprocess da \
--tied-embeddings-all --dim-emb $DIMEMB --transformer-dim-ffn $DIMTRA \
--transformer-dropout $DROPOUT --transformer-dropout-attention $DROPOUT \
--transformer-dropout-ffn $DROPOUT --label-smoothing $DROPOUT \
--learn-rate $LR \
--lr-warmup 8000 --lr-decay-inv-sqrt 8000 --lr-report \
--optimizer-params 0.9 0.98 1e-09 --clip-norm 5 \
--devices $GPUS --workspace $WORKSPACE --seed $RANDSEED \
--exponential-smoothing \
--keep-best \
--max-length 1000 --valid-max-length 5000 --max-length-crop \
--shuffle-in-ram --mini-batch-fit \
--sentencepiece-options "--character_coverage=1.0 --user_defined_symbols=BE,CA,CH,FR"
