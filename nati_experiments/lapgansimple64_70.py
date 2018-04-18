import sys
sys.path.insert(0, '../')

import matplotlib
matplotlib.use('Agg')

import data
from model import LapPatchWGANsimpleModel
from gan import CosmoGAN
import utils

import numpy as np


# # Parameters


ns = 64
try_resume = True
Mpch=70
upscaling = 4

def forward(X):
    return np.log(X**(1/2)+np.e)-2

def backward(Xmap, max_value=2e5):
    Xmap = np.clip(Xmap, -1.0, forward(max_value))
    tmp = np.exp((Xmap+2))-np.e
    return np.round(tmp*tmp)


time_str = 'test_{}'.format(Mpch)
global_path = '../../../saved_result/'

name = 'LapPatchsimple{}'.format(ns)
new_ns = ns//upscaling
latent_dim = new_ns**2
bn = False

params_discriminator = dict()
params_discriminator['stride'] = [2, 2, 2, 1]
params_discriminator['nfilter'] = [16, 64, 128, 16]
params_discriminator['shape'] = [[5, 5], [5, 5], [5, 5], [3, 3]]
params_discriminator['batch_norm'] = [bn, bn, bn, bn]
params_discriminator['full'] = [64]
params_discriminator['summary'] = True
params_discriminator['minibatch_reg'] = False

params_generator = dict()
params_generator['stride'] = [2, 2, 1, 1, 1]
params_generator['y_layer'] = 2
params_generator['latent_dim'] = latent_dim
params_generator['nfilter'] = [64, 128, 64, 32, 1]
params_generator['shape'] = [[3, 3], [5, 5], [5, 5], [5, 5], [5, 5]]
params_generator['batch_norm'] = [bn, bn, bn, bn]
params_generator['full'] = []
params_generator['summary'] = True
params_generator['non_lin'] = None
params_generator['upsampling'] = upscaling

params_optimization = dict()
params_optimization['gamma_gp'] = 10
params_optimization['batch_size'] = 8
params_optimization['gen_optimizer'] = 'rmsprop' # rmsprop / adam / sgd
params_optimization['disc_optimizer'] = 'rmsprop' # rmsprop / adam /sgd
params_optimization['disc_learning_rate'] = 3e-5
params_optimization['gen_learning_rate'] = 3e-5
params_optimization['beta1'] = 0.9
params_optimization['beta2'] = 0.999
params_optimization['epsilon'] = 1e-8
params_optimization['epoch'] = 50


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

params['normalize'] = False
params['image_size'] = [ns, ns, 4]
params['prior_distribution'] = 'gaussian'
params['sum_every'] = 500
params['viz_every'] = 500
params['print_every'] = 100
params['save_every'] = 2000
params['name'] = name
params['summary_dir'] = global_path + params['name'] + '_' + time_str +'_summary/'
params['save_dir'] = global_path + params['name'] + '_' + time_str + '_checkpoints/'


resume, params = utils.test_resume(try_resume, params)

obj = CosmoGAN(params, LapPatchWGANsimpleModel)

dataset = data.load.load_dataset(spix=ns, resolution=256,Mpch=Mpch, forward_map=forward, patch=True)

obj.train(dataset=dataset, resume=resume)