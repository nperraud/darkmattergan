# coding: utf-8

import sys
sys.path.insert(0, '../')

import data
from model import WGanModel
from gan import CosmoGAN
import utils
import functools
import os
from data import fmap
import tensorflow as tf
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]=""

# Parameters

ns = 32
try_resume = True
Mpch = 70
shift = 3
c = 20000
res = 256

scaling = 4

time_str = 'stat_c_{}_shift_{}_laplacian_Mpch_{}_res_{}'.format(c, shift, Mpch, res)
global_path = '/scratch/snx3000/nperraud/saved_result'

name = 'WGAN{}_downsample_4'.format(ns)



forward = functools.partial(fmap.stat_forward, shift=shift, c=c)
backward = functools.partial(fmap.stat_backward, shift=shift, c=c)
def non_lin(x):
	return tf.nn.relu(x)

bn = False
latent_dim=100

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
params_generator['latent_dim'] = latent_dim
params_generator['nfilter'] = [32, 64, 128, 64, 1]
params_generator['shape'] = [[3, 3], [5, 5], [5, 5], [5, 5], [5, 5]]
params_generator['batch_norm'] = [bn, bn, bn, bn]
params_generator['full'] = [8 * 8 * 32]
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
params_cosmology['forward_map'] = forward
params_cosmology['backward_map'] = backward
params_cosmology['Nstats'] = 5000


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

# Build the model

wgan = CosmoGAN(params, WGanModel)

dataset = data.load.load_dataset(
    spix=ns, resolution=256, Mpch=Mpch, scaling=scaling, forward_map=forward)

# Train the model
wgan.train(dataset, resume=resume)
