#!/bin/bash

#SBATCH --account=rlfarmcontr
#SBATCH --qos=high
#SBATCH --time=00-04:00:00
#SBATCH --nodes=1
##SBATCH --partition=debug
#SBATCH --job-name=baseline0.05
#SBATCH -o out_base0.05

module purge
module load tmux
module load git
module load intel-oneapi-mpi
module load intel-oneapi-mkl
module load intel-oneapi-compilers
module load mamba
mamba activate /kfs2/projects/rlfarmcontr/RLControl_Phase1End/mambaenv

python BaselinePerformance.py 0.05
