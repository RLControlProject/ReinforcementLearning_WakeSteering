module load mamba
mamba activate /kfs2/projects/rlfarmcontr/RLControl_Phase1End/mambaenv

python BaselinePerformance.py 0.025
python BaselinePerformance.py 0.05
python BaselinePerformance.py 0.075
