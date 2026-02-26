#!/bin/bash

#SBATCH --account=ssc
#SBATCH --time=00-04:00:00
#SBATCH --nodes=1
##SBATCH --partition=debug
#SBATCH --job-name=Baseline_Seed6_0.075
#SBATCH -o out_Baseline_Seed6_0.075

module load mamba
mamba activate /kfs2/projects/rlfarmcontr/RLControl_Phase1End/mambaenv

python BaselinePerformance.py 0.075
