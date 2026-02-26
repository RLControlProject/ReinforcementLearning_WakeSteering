#!/bin/bash

#SBATCH --account=ssc
#SBATCH --time=02-00:00:00
#SBATCH --nodes=1
##SBATCH --partition=debug
#SBATCH --job-name=Baseline_Seed5
#SBATCH -o out_Baseline_Seed5

module load mamba
mamba activate /kfs2/projects/rlfarmcontr/RLControl_Phase1End/mambaenv

python BaselinePerformance.py 0.025
python BaselinePerformance.py 0.05
python BaselinePerformance.py 0.075
