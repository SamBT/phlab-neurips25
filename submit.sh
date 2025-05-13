#!/bin/bash
#SBATCH --partition=iaifi_gpu_priority
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/output-%j.out
#SBATCH --error=slurm_logs/error-%j.err

source ~/.bash_profile
mamba activate torch_gpu
cd /n/home11/sambt/phlab-neurips25
python cli.py fit "$@"