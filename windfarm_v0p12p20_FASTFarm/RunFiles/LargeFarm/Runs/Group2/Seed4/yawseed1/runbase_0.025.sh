#!/bin/bash

#SBATCH --account=ssc
#SBATCH --time=00-04:00:00
#SBATCH --nodes=1
##SBATCH --partition=debug
#SBATCH --job-name=Baseline_Seed4_0.025
#SBATCH -o out_Baseline_Seed4_0.025

module load mamba
mamba activate /kfs2/projects/rlfarmcontr/RLControl_Phase1End/mambaenv

python BaselinePerformance.py 0.025
