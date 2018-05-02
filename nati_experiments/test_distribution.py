# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

import sys
sys.path.insert(0, '../')

import matplotlib
matplotlib.use('Agg')

import data
from model import WGanModel
from gan import CosmoGAN
import utils
import numpy as np
import tensorflow as tf

# Parameters
ns = 8

# def gen(N):
# 	return np.abs(np.random.laplace(loc=0.0,scale=1.0,size=[N,128,128]))+1

# def gen(N):
# 	return np.abs(np.random.normal(loc=0.0,scale=1.0,size=[N,128,128]))+1


# def gen(N):
# 	X = np.abs(np.random.normal(loc=0.0,scale=1.0,size=[N,128,128]))
# 	return X*X+1

def gen(N):
	X = np.abs(np.random.normal(loc=0.0,scale=1.0,size=[N,128,128]))
	return X*X*X+1

def forward(X):
    return X-1

def backward(X):
    return X+1

time_str = 'normal_cube'
global_path = '../../../saved_result/'

name = 'WGAN_distribution_large_'.format(ns)

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
params_optimization['epoch'] = 10

latent_dim = ns*ns
bn = False
params_discriminator = dict()
params_discriminator['stride'] = [2, 1, 1, 1, 1]
params_discriminator['nfilter'] = [32, 64, 128, 64, 32]
params_discriminator['shape'] = [[5, 5], [3, 3], [3, 3], [3, 3], [3, 3]]
params_discriminator['batch_norm'] = [bn, bn, bn, bn, bn]
params_discriminator['full'] = [64]
params_discriminator['summary'] = True
params_discriminator['minibatch_reg'] = False

params_generator = dict()
params_generator['stride'] = [2, 1, 1, 1, 1, 1]
params_generator['latent_dim'] = latent_dim
params_generator['nfilter'] = [16, 64, 128, 64, 32, 1]
params_generator['shape'] = [[5, 5], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]]
params_generator['batch_norm'] = [bn, bn, bn, bn, bn]
params_generator['full'] = [16*16]
params_generator['summary'] = True
params_generator['non_lin'] = tf.nn.relu

params_cosmology = dict()
params_cosmology['forward_map'] = forward
params_cosmology['backward_map'] = backward
params_cosmology['Nstats'] = 5000

params = dict()
params['generator'] = params_generator
params['discriminator'] = params_discriminator
params['optimization'] = params_optimization
params['cosmology'] = params_cosmology


params['prior_distribution'] = 'gaussian'
params['sum_every'] = 100
params['viz_every'] = 100
params['save_every'] = 1000
params['normalize'] = False
params['image_size'] = [ns, ns]
params['name'] = name
params['summary_dir'] = global_path + params['name'] + '_' + time_str +'_summary/'
params['save_dir'] = global_path + params['name'] + '_' + time_str + '_checkpoints/'


resume = False

obj = CosmoGAN(params, WGanModel)

dataset = data.load.load_distribution(gen=gen, spix=ns, forward_map=forward)

obj.train(dataset=dataset, resume=resume)
