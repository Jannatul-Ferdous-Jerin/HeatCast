#!/bin/bash
export PYTHONPATH="/home/rmedu/Music/HEAT-WAVE/TimeCMA-heat-wave":$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

data_path="heat-wave"
seq_len=96
batch_size=32

# pred_len 96
pred_len=7
learning_rate=1e-4
channel=64
e_layer=6
d_layer=2
dropout_n=0.1

log_path="./Results/${data_path}/"
mkdir -p $log_path
log_file="${log_path}i${seq_len}_o${pred_len}_lr${learning_rate}_c${channel}_el${e_layer}_dl${d_layer}_dn${dropout_n}_bs${batch_size}.log"
nohup python train.py \
  --data_path $data_path \
  --batch_size $batch_size \
  --num_nodes 15\
  --seq_len $seq_len \
  --pred_len $pred_len \
  --epochs 20 \
  --seed 2024 \
  --channel $channel \
  --learning_rate $learning_rate \
  --dropout_n $dropout_n \
  --e_layer $e_layer \
  --d_layer $d_layer > $log_file &