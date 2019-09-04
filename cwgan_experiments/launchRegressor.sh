#!/bin/bash -l
#SBATCH --time=17:59:59
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=gpu
#SBATCH --output=out-regressor-%j.log
#SBATCH --error=out-regressor-%j-e.log
#SBATCH --account=sd01

module load daint-gpu
module load cray-python
module load TensorFlow/1.12.0-CrayGNU-18.08-cuda-9.1-python3
source /users/nperraud/upgan/bin/activate

srun python train_regressor.py
