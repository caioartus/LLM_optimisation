#!/bin/bash
# Train a model with nanoGPT's train.py on one GPU.
#
# Usage (from the repo root):
#   mkdir -p logs                                   # once; SLURM needs it to exist
#   sbatch scripts/train.sbatch                     # default config below
#   sbatch scripts/train.sbatch configs/cluster_test.py --max_iters=500
#   sbatch --gres=gpu:a100:1 scripts/train.sbatch   # pick a GPU model (name is cluster-specific)
#
# Flags given to sbatch on the command line override the #SBATCH lines below.

#SBATCH --job-name=train
#SBATCH --gres=gpu:a100-40:1            
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out      # %x = job name, %j = job id

set -euo pipefail

CONFIG="${1:-configs/cluster_test.py}"
shift || true                        # everything after the config goes to train.py

export PATH="$HOME/.local/bin:$PATH" # where uv usually lives
# export WANDB_MODE=offline          # uncomment if compute nodes have no internet

echo "job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "config: $CONFIG  extra args: $*"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
export PYTHONUNBUFFERED=1
uv run --frozen python train.py "$CONFIG" "$@"

echo "finished at $(date)"
