#!/bin/bash -l
#SBATCH --time=23:59:59
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=gpu
#SBATCH --output=out-test-%j.log
#SBATCH --error=out-test-%j-e.log
#SBATCH --account=sd01

module load daint-gpu
module load cray-python
module load TensorFlow/1.7.0-CrayGNU-18.08-cuda-9.1-python3

source $SCRATCH/cgan/bin/activate

srun python train.py
