#!/usr/bin/env bash

cuda="0"
dataset="linnaeus-IOB"

mkdir -p tmp

# CUDA_LAUNCH_BLOCKING=1 kernprof -l examples/NERCRF.py
CUDA_VISIBLE_DEVICES=$cuda python3 examples/NERCRF.py --mode LSTM --num_epochs 80 --batch_size 16 --hidden_size 256 \
	  --char_dim 30 --num_filters 30 --tag_space 128 \
	  --learning_rate 0.001 --momentum 0 --alpha 0.95 --lr_decay 0.97 --schedule 1 --gamma 0.0 \
	  --dropout std --p 0.5 --unk_replace 0.0 --bigram \
	  --embedding "glove" --embedding_dict "data/glove/glove.6B/glove.6B.100d.gz" \
	  --elmo_cuda 0 --attention "none" --data_reduce 1.0 \
	  --train "data/"$dataset"/train.tsv" --dev "data/"$dataset"/devel.tsv" --test "data/"$dataset"/test.tsv" \
