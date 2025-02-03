#!/bin/bash -x
#SBATCH --mem=32G
#SBATCH --qos=high     
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=10:00:00


module load Python3/3.11.2
# activate you virtual environment or conda
source /you/venv/directory/bin/activate
# the next 3 lines install our linear attention library (fastmax_cuda). NOTE: gcc version must be < 13.0.0
module load gcc
module load cuda
python setup.py install
python test_fastmax.py

# Note: for d = 64, linear attention will be faster than flash attention for roughly > 16k tokens