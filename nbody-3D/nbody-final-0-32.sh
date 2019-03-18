#!/bin/bash -l
#SBATCH --time=23:59:00
#SBATCH --nodes=1
#SBATCH --account=sd01
#SBATCH --ntasks=1
#SBATCH --constraint=gpu
#SBATCH --output=nbody-final-32-%j.log
#SBATCH --error=nbody-final-32-e-%j.log

module load daint-gpu
module load cray-python
module load TensorFlow/1.7.0-CrayGNU-18.08-cuda-9.1-python3

source /scratch/snx3000/nperraud/upgan2/bin/activate

srun python nbody-final-0-32.py
