import os
import sys
sys.path.insert(0, '../')
from gantools.data import transformation
from gantools.regressor import Regressor
from gantools.gansystem import NNSystem
from gantools import utils

from cosmotools.data import load


# Note: some of the parameters don't make sense for the fake dataset
ns = 128 # Resolution of the image
try_resume = True # Try to resume previous simulation


time_str = '2D_mac'
global_path = '../saved_results/Regressor/'

name = 'Kids_Regressor_' + str(ns) + '_smart_' + time_str


bn = False

# Parameters for the regressor
params_regressor = dict()
params_regressor['full'] = [512, 256, 128]
params_regressor['nfilter'] = [32, 64, 128, 256, 512]
params_regressor['batch_norm'] = [bn, bn, bn, bn, bn]
params_regressor['shape'] = [[7, 7], [5, 5], [5, 5], [5,5], [3,3]]
params_regressor['stride'] = [1, 2, 2, 2, 2]

# Optimization parameters
params_optimization = dict()
params_optimization['learning_rate'] = 3e-5
params_optimization['batch_size'] = 64
params_optimization['epoch'] = 100

# all parameters
params = dict()
params['net'] = dict() # All the parameters for the model
params['net']['regressor'] = params_regressor
params['net']['shape'] = [ns, ns, 1] # Shape of the image
params['net']['cond_params'] = 2
params['optimization'] = params_optimization
params['summary_every'] = 2000 # Tensorboard summaries every ** iterations
params['print_every'] = 1000 # Console summaries every ** iterations
params['save_every'] = 10000 # Save the model every ** iterations
params['summary_dir'] = os.path.join(global_path, name +'_summary/')
params['save_dir'] = os.path.join(global_path, name + '_checkpoints/')

resume, params = utils.test_resume(try_resume, params)

reg = NNSystem(Regressor, params)

dataset = load.load_params_dataset(filename='kids_reg_train.h5', batch=10000, transform=transformation.random_transpose_2d, shape=[ns, ns])

reg.train(dataset, resume=resume)