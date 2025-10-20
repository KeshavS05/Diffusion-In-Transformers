#!/bin/bash
#SBATCH --account=beig-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1 
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=3G
#SBATCH --time=01:00:00

#SBATCH --job-name=train_dit_tiny
#SBATCH --output=%x-%j.out 

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dit

torchrun --nproc_per_node=1 train.py \
  --data-path tiny-imagenet-200/train \
  --results-dir results_tiny \
  --model DiT-S/2 \
  --image-size 256 \
  --num-classes 200 \
  --epochs 5 \
  --global-batch-size 8 \
  --num-workers 4 \
  --vae mse \
  --ckpt-every 1000
