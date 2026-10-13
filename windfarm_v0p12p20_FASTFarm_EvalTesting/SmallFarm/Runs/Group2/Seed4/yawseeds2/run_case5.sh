#!/bin/bash

#SBATCH --account=rlfarmcontr
#SBATCH --time=02-00:00:00
#SBATCH --nodes=1
##SBATCH --partition=debug
#SBATCH --job-name=RL_Case5_Seed4
#SBATCH -o out_RL_Case5_Seed4

module purge
module load tmux
module load git
module load intel-oneapi-mpi
module load intel-oneapi-mkl
module load intel-oneapi-compilers
module load mamba
mamba activate /kfs3/scratch/agupta/Projects/RL_Incubator/tempdel/scratchenv

rm -f log.txt

python sac_post_generate_sequence_5_10env_large_1.py
