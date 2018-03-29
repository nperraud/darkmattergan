import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import sys
sys.path.insert(0, '../')

import matplotlib
matplotlib.use('Agg')

import data
from model import LapPatchWGANsingleModel
from gan import CosmoGAN
import utils

import numpy as np


# # Parameters


ns = 128
scaling = 4
try_resume = False
Mpch=350

def forward(X):
    return np.log(X**(1/2)+np.e)-2

def backward(Xmap, max_value=2e5):
    Xmap = np.clip(Xmap, -1.0, forward(max_value))
    tmp = np.exp((Xmap+2))-np.e
    return np.round(tmp*tmp)


time_str = 'single_{}'.format(Mpch)
global_path = '../../../saved_result/'

name = 'LapPatchWGAN{}'.format(ns)





params_optimization = dict()
params_optimization['gamma_gp'] = 10
params_optimization['weight_l2'] = 0.1
params_optimization['batch_size'] = 16
params_optimization['gen_optimizer'] = 'rmsprop' # rmsprop / adam / sgd
params_optimization['disc_optimizer'] = 'rmsprop' # rmsprop / adam /sgd
params_optimization['disc_learning_rate'] = 3e-5
params_optimization['gen_learning_rate'] = 3e-5
params_optimization['beta1'] = 0.5
params_optimization['beta2'] = 0.99
params_optimization['epsilon'] = 1e-8
params_optimization['epoch'] = 2000




up_scaling = scaling
new_ns = ns//scaling
latent_dim = new_ns**2
bn = False
params_discriminator = dict()
params_discriminator['stride'] = [2, 2, 2, 2, 1]
params_discriminator['nfilter'] = [16, 256, 512, 256, 16]
params_discriminator['shape'] = [[5, 5], [5, 5], [5, 5], [5, 5], [3, 3]]
params_discriminator['batch_norm'] = [bn, bn, bn, bn, bn]
params_discriminator['full'] = [32]
params_discriminator['summary'] = True
params_discriminator['minibatch_reg'] = False

params_generator = dict()
params_generator['stride'] = [2, 2, 1, 1, 1, 1]
params_generator['y_layer'] = 2
params_generator['latent_dim'] = latent_dim
params_generator['nfilter'] = [64, 256, 512, 256, 64, 1]
params_generator['shape'] = [[3, 3], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5]]
params_generator['batch_norm'] = [bn, bn, bn, bn, bn]
params_generator['full'] = []
params_generator['summary'] = True
params_generator['non_lin'] = None
params_generator['upsampling'] = up_scaling

params_cosmology = dict()
params_cosmology['clip_max_real'] = True
params_cosmology['log_clip'] = 0.1
params_cosmology['sigma_smooth'] = 1
params_cosmology['forward_map'] = forward
params_cosmology['backward_map'] = backward
params_cosmology['Npsd'] = 500

params = dict()
params['generator'] = params_generator
params['discriminator'] = params_discriminator
params['optimization'] = params_optimization
params['cosmology'] = params_cosmology


params['prior_distribution'] = 'gaussian'
params['sum_every'] = 200
params['viz_every'] = 200
params['save_every'] = 5000
params['normalize'] = False
params['image_size'] = [new_ns*up_scaling, new_ns*up_scaling]
params['name'] = name
params['summary_dir'] = global_path + params['name'] + '_' + time_str +'_summary/'
params['save_dir'] = global_path + params['name'] + '_' + time_str + '_checkpoints/'


resume, params = utils.test_resume(try_resume, params)

obj = CosmoGAN(params, LapPatchWGANsingleModel)



dataset = data.load.load_2d_dataset(resolution=256,Mpch=Mpch, forward_map=forward,spix=ns)


obj.train(dataset=dataset, resume=resume)
