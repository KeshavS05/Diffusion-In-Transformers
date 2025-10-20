#!/bin/bash
#SBATCH --account=beig-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1 
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=3G
#SBATCH --time=01:00:00

#SBATCH --job-name=sample_dit_tiny
#SBATCH --output=%x-%j.out 

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dit

torchrun --nproc_per_node=1 sample_ddp.py \
  --model DiT-S/2 \
  --vae mse \
  --sample-dir ~/Diffusion-In-Transformers/DiT/samples \
  --num-classes 200 \
  --ckpt ~/Diffusion-In-Transformers/DiT/results_tiny/002-DiT-S-2/checkpoints/0010400.pt
