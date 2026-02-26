#!/bin/bash

#SBATCH --account=ssc
#SBATCH --time=02-00:00:00
#SBATCH --nodes=1
##SBATCH --partition=debug
#SBATCH --job-name=RL_Case5_Seed6
#SBATCH -o out_RL_Case5_Seed6

module purge
module load tmux
module load git
module load intel-oneapi-mpi
module load intel-oneapi-mkl
module load intel-oneapi-compilers
module load conda
conda activate /kfs2/projects/rlfarmcontr/RLControl_Phase1End/mambaenv

python sac_post_generate_sequence_5_10env_large_1.py
