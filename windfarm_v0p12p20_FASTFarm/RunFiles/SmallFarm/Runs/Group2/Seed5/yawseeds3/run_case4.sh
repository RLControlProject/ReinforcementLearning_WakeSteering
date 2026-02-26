#!/bin/bash

#SBATCH --account=rlfarmcontr
#SBATCH --time=02-00:00:00
#SBATCH --nodes=1
##SBATCH --partition=debug
#SBATCH --job-name=RL_Case4_Seed5
#SBATCH -o out_RL_Case4_Seed5

module purge
module load tmux
module load git
module load intel-oneapi-mpi
module load intel-oneapi-mkl
module load intel-oneapi-compilers
module load mamba
mamba activate /kfs3/scratch/agupta/Projects/RL_Incubator/tempdel/scratchenv

rm -f log.txt

python sac_post_generate_sequence_4_10env_large_1.py
