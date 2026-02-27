#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=9:59:59        # Wall clock time (HH:MM:SS) - once the job exceeds this time, the job will be terminated (default is 5 minutes)
##SBATCH --account=FY250146        # WC ID
#SBATCH --account=fy250142        # WC ID (FLOWMAS)
#SBATCH --job-name=RL   # Name of job
#SBATCH --partition=batch # partition/queue name: short or batch
#SBATCH --qos=normal           # Quality of Service: long, large, priority or normal 
#SBATCH --array=1-3             # Job array for 3 scripts
#SBATCH --reservation=flight-cldera
#SBATCH --licenses=tscratch

module load aue/anaconda3/2024.06-1
eval "$(conda shell.bash hook)" # this is apparently how you can do "conda init" in a shell script
conda activate RLControlEnv

cd /ascldap/users/kbrown1/tscratch/Reinforcement_Learning/RLControl/windfarm_v0p12p20_1seed

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    trial_name="3_10env_medium"
    output_dir="sac_optuna_search_sequence/trials/trial_${trial_name}_1_1"
    mkdir -p "$output_dir"
    python3 sac_optuna_search_sequence_${trial_name}_1.py > "$output_dir/output.txt" 2>&1
elif [ $SLURM_ARRAY_TASK_ID -eq 2 ]; then
    trial_name="4_10env_medium"
    output_dir="sac_optuna_search_sequence/trials/trial_${trial_name}_1_1"
    mkdir -p "$output_dir"
    python3 sac_optuna_search_sequence_${trial_name}_1.py > "$output_dir/output.txt" 2>&1
elif [ $SLURM_ARRAY_TASK_ID -eq 3 ]; then
    trial_name="5_10env_medium"
    output_dir="sac_optuna_search_sequence/trials/trial_${trial_name}_1_1"
    mkdir -p "$output_dir"
    python3 sac_optuna_search_sequence_${trial_name}_1.py > "$output_dir/output.txt" 2>&1
fi