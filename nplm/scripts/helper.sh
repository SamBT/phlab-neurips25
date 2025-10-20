#!/bin/bash
#SBATCH --partition=gpu_test
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=logs/output-%j.out
#SBATCH --error=logs/error-%j.err

source ~/.bash_profile
mamba activate torch_gpu
cd /n/home11/sambt/phlab-neurips25/nplm/scripts

for noise in 0.01 0.02 0.05 0.1; do for NSIG in 0 10 20 50 100 200 500; do python run_toys-NysMMD.py -a 4 -n jetclass_T0.1_v2_noise${noise} -r /n/home11/sambt/phlab-neurips25/experiments/jetclass/jetClass_embeddings_T0.1_noise${noise}.npz -d /n/home11/sambt/phlab-neurips25/experiments/jetclass/jetClass_embeddings_T0.1_noise${noise}.npz -l 1 -t 500 -s $NSIG --nref 50000 --nbkg 10000; done; done
