#!/bin/bash -l
#SBATCH --time=12:59:59
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=gpu
#SBATCH --output=evaluation-test-%j.log
#SBATCH --error=evaluation-test-%j.log

module load daint-gpu
module load cray-python
module load TensorFlow/1.7.0-CrayGNU-17.12-cuda-8.0-python3

source $SCRATCH/cgan/bin/activate

srun python model_selection_kids.py
