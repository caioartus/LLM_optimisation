#!/bin/bash
#SBATCH --job-name=prep-data
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/prep-%j.out

uv run python data/prepare.py
