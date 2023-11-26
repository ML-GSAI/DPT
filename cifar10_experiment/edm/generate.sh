#!/bin/bash
num=$1
save_dir=$2
model=$3
nproc_per_node=$4

torchrun --standalone --nproc_per_node=$nproc_per_node generate.py --outdir=$save_dir/0 --seeds=0-$num --class=0 --network=$model 
torchrun --standalone --nproc_per_node=$nproc_per_node generate.py --outdir=$save_dir/1 --seeds=0-$num --class=1 --network=$model
torchrun --standalone --nproc_per_node=$nproc_per_node generate.py --outdir=$save_dir/2 --seeds=0-$num --class=2 --network=$model
torchrun --standalone --nproc_per_node=$nproc_per_node generate.py --outdir=$save_dir/3 --seeds=0-$num --class=3 --network=$model
torchrun --standalone --nproc_per_node=$nproc_per_node generate.py --outdir=$save_dir/4 --seeds=0-$num --class=4 --network=$model
torchrun --standalone --nproc_per_node=$nproc_per_node generate.py --outdir=$save_dir/5 --seeds=0-$num --class=5 --network=$model
torchrun --standalone --nproc_per_node=$nproc_per_node generate.py --outdir=$save_dir/6 --seeds=0-$num --class=6 --network=$model
torchrun --standalone --nproc_per_node=$nproc_per_node generate.py --outdir=$save_dir/7 --seeds=0-$num --class=7 --network=$model
torchrun --standalone --nproc_per_node=$nproc_per_node generate.py --outdir=$save_dir/8 --seeds=0-$num --class=8 --network=$model
torchrun --standalone --nproc_per_node=$nproc_per_node generate.py --outdir=$save_dir/9 --seeds=0-$num --class=9 --network=$model