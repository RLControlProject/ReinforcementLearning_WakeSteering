#!/bin/bash

#SBATCH --account=rlfarmcontr
#SBATCH --time=02-00:00:00
#SBATCH --nodes=1
#SBATCH --job-name=run_baseline
#SBATCH -o out_baseline

module load mamba
mamba activate /kfs3/scratch/agupta/Projects/RL_Incubator/tempdel/scratchenv

python BaselinePerformance.py 0.025
python BaselinePerformance.py 0.05
python BaselinePerformance.py 0.075
