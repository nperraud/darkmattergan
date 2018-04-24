# coding: utf-8

import sys
sys.path.insert(0, '../')

import data
import numpy as np
from model import WGanModel
from gan import CosmoGAN
import utils
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]=""

# Parameters

ns = 32
try_resume = True
Mpch = 70
scaling = 4

time_str = 'downsample4_{}'.format(Mpch)
global_path = '../../../saved_result/'

name = 'WGAN{}'.format(ns)


def forward(X):
    return np.log(np.sqrt(X) + np.e) - 2


def backward(Xmap, max_value=2e5):
    Xmap = np.clip(Xmap, -1.0, forward(max_value))
    tmp = np.exp((Xmap + 2)) - np.e
    return np.round(tmp * tmp)


bn = False

params_discriminator = dict()
params_discriminator['stride'] = [2, 2, 2, 1]
params_discriminator['nfilter'] = [16, 64, 128, 16]
params_discriminator['shape'] = [[5, 5], [5, 5], [5, 5], [5, 5]]
params_discriminator['batch_norm'] = [bn, bn, bn, bn]
params_discriminator['full'] = [32]
params_discriminator['minibatch_reg'] = False
params_discriminator['summary'] = True

params_generator = dict()
params_generator['stride'] = [2, 2, 1, 1, 1]
params_generator['latent_dim'] = 100
params_generator['nfilter'] = [32, 64, 128, 64, 1]
params_generator['shape'] = [[3, 3], [5, 5], [5, 5], [5, 5], [5, 5]]
params_generator['batch_norm'] = [bn, bn, bn, bn]
params_generator['full'] = [8 * 8 * 32]
params_generator['summary'] = True
params_generator['non_lin'] = None

params_optimization = dict()
params_optimization['gamma_gp'] = 10
params_optimization['batch_size'] = 16
params_optimization['gen_optimizer'] = 'rmsprop'  # rmsprop / adam / sgd
params_optimization['disc_optimizer'] = 'rmsprop'  # rmsprop / adam /sgd
params_optimization['disc_learning_rate'] = 3e-5
params_optimization['gen_learning_rate'] = 3e-5
params_optimization['beta1'] = 0.9
params_optimization['beta2'] = 0.999
params_optimization['epsilon'] = 1e-8
params_optimization['epoch'] = 1000

params_cosmology = dict()
params_cosmology['clip_max_real'] = True
params_cosmology['log_clip'] = 0.1
params_cosmology['sigma_smooth'] = 1
params_cosmology['forward_map'] = forward
params_cosmology['backward_map'] = backward
params_cosmology['Nstats'] = 640

params = dict()
params['generator'] = params_generator
params['discriminator'] = params_discriminator
params['optimization'] = params_optimization
params['cosmology'] = params_cosmology

params['normalize'] = False
params['image_size'] = [ns, ns]
params['prior_distribution'] = 'gaussian'
params['sum_every'] = 500
params['viz_every'] = 500
params['print_every'] = 100
params['save_every'] = 2000
params['name'] = name
params['summary_dir'] = global_path + params['name'] + '_' + time_str + '_summary/'
params['save_dir'] = global_path + params['name'] + '_' + time_str + '_checkpoints/'

resume, params = utils.test_resume(try_resume, params)

# Build the model

wgan = CosmoGAN(params, WGanModel)

dataset = data.load.load_dataset(
    spix=ns, resolution=256, Mpch=Mpch, scaling=scaling, forward_map=forward)

# Train the model
wgan.train(dataset, resume=resume)
