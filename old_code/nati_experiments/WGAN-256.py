# coding: utf-8

import sys
sys.path.insert(0, '../')

import matplotlib
matplotlib.use('Agg')

import data
from model import WGanModel
from gan import CosmoGAN
import utils
from data import fmap
import tensorflow as tf
import functools
import os 

# Parameters

ns = 256
try_resume = True
Mpch = 70
shift = 3
c = 20000
res = 256
forward = functools.partial(fmap.stat_forward, shift=shift, c=c)
backward = functools.partial(fmap.stat_backward, shift=shift, c=c)


def non_lin(x):
	return tf.nn.relu(x)

time_str = 'stat_c_{}_shift_{}_laplacian_Mpch_{}_res_{}'.format(c, shift, Mpch, res)
global_path = '/scratch/snx3000/nperraud/saved_result'

name = 'WGAN{}'.format(ns)

bn = False

params_discriminator = dict()
params_discriminator['stride'] = [2, 2, 2, 2, 2, 1]
params_discriminator['nfilter'] = [16, 128, 256, 512, 128, 32]
params_discriminator['shape'] = [[5, 5], [5, 5], [5, 5], [5, 5], [3, 3], [3, 3]]
params_discriminator['batch_norm'] = [bn, bn, bn, bn, bn, bn]
params_discriminator['full'] = [64]
params_discriminator['minibatch_reg'] = False
params_discriminator['summary'] = True

params_generator = dict()
params_generator['stride'] = [2, 2, 2, 2, 1, 1]
params_generator['latent_dim'] = 16*16*64
params_generator['nfilter'] = [64, 256, 512, 256, 64, 1]
params_generator['shape'] = [[3, 3], [3, 3], [5, 5], [5, 5], [5, 5], [5, 5]]
params_generator['batch_norm'] = [False, False, False, False, False]
params_generator['full'] = []
params_generator['summary'] = True
params_generator['non_lin'] = non_lin

params_optimization = dict()
params_optimization['gamma_gp'] = 10
params_optimization['batch_size'] = 16
params_optimization['gen_optimizer'] = 'adam' # rmsprop / adam / sgd
params_optimization['disc_optimizer'] = 'adam' # rmsprop / adam /sgd
params_optimization['disc_learning_rate'] = 1e-5
params_optimization['gen_learning_rate'] = 1e-5
params_optimization['beta1'] = 0.5
params_optimization['beta2'] = 0.99
params_optimization['epsilon'] = 1e-8
params_optimization['epoch'] = 1000

params_cosmology = dict()
params_cosmology['clip_max_real'] = True
params_cosmology['log_clip'] = 0.1
params_cosmology['sigma_smooth'] = 1
params_cosmology['forward_map'] = forward
params_cosmology['backward_map'] = backward
params_cosmology['Nstats'] = 2000


params = dict()
params['generator'] = params_generator
params['discriminator'] = params_discriminator
params['optimization'] = params_optimization
params['cosmology'] = params_cosmology

params['normalize'] = False
params['image_size'] = [ns, ns]
params['prior_distribution'] = 'laplacian'
params['sum_every'] = 200
params['viz_every'] = 200
params['save_every'] = 5000
params['name'] = name
params['summary_dir'] = os.path.join(global_path, params['name'] + '_' + time_str +'_summary/')
params['save_dir'] = os.path.join(global_path, params['name'] + '_' + time_str + '_checkpoints/')

resume, params = utils.test_resume(try_resume, params)
# params['optimization']['disc_learning_rate'] = 3e-6
# params['optimization']['gen_learning_rate'] = 3e-6


# Build the model
wgan = CosmoGAN(params, WGanModel)

dataset = data.load.load_dataset(resolution=res, Mpch=Mpch, forward_map=forward, spix=ns)

wgan.train(dataset=dataset, resume=resume)
