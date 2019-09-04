#!/bin/bash -l
#SBATCH --time=12:59:59
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=gpu
#SBATCH --output=evaluation-test-%j.log
#SBATCH --error=evaluation-test-%j.log

module load daint-gpu
module load cray-python
module load TensorFlow/1.12.0-CrayGNU-18.08-cuda-9.1-python3
source /users/nperraud/upgan/bin/activate

srun python model_selection_kids.py
