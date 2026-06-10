#!/bin/bash
export PYTHONPATH="/home/rmedu/Music/Heat_Alert/":$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

data_paths=("23p75__90p50")
divides=("train" "val" "test")
num_nodes=13
input_len=96         # <-- Set to 4 weeks (28 days)
output_len=7         # <-- Set to 1 week (7 days)
lead_time=2        # <-- Add lead_time variable (e.g., 2 for a 1-week gap)

for data_path in "${data_paths[@]}"; do
  for divide in "${divides[@]}"; do
    log_file="./Results/emb_logs/${data_path}_${divide}_lead${lead_time}.log" # Optional: make log file name more descriptive
    
    # Add the --lead_time argument to the python command
    nohup python storage/store_emb.py \
        --divide $divide \
        --data_path $data_path \
        --num_nodes $num_nodes \
        --input_len $input_len \
        --output_len $output_len \
        --lead_time $lead_time > $log_file &
  done
done