#!/bin/bash -l
#SBATCH --time=23:59:00
#SBATCH --nodes=1
#SBATCH --account=sd01
#SBATCH --ntasks=1
#SBATCH --constraint=gpu
#SBATCH --output=medical-256-%j.log
#SBATCH --error=medical-256-e-%j.log

module load daint-gpu
module load cray-python
module load TensorFlow/1.7.0-CrayGNU-18.08-cuda-9.1-python3

source $HOME/upgan/bin/activate

srun python medical-64-256.py
