# coding: utf-8

import sys
sys.path.insert(0, '../')

import matplotlib
matplotlib.use('Agg')

import os
import data
from model import WGanModel
from gan import CosmoGAN
import utils
from data import fmap, path, Dataset
import tensorflow as tf
import numpy as np

# Parameters
ns = 64
try_resume = False
Mpch = 100


forward = fmap.forward
backward = fmap.backward


time_str = 'original_single_{}'.format(Mpch)
global_path = '/scratch/snx3000/rosenthj/results/'

name = 'WGAN{}'.format(ns)

bn = False

params_discriminator = dict()
params_discriminator['stride'] = [2, 2, 2, 2, 1, 1]
params_discriminator['nfilter'] = [16, 128, 256, 512, 128, 64]
params_discriminator['shape'] = [[5, 5],[5, 5],[5, 5], [3, 3], [3, 3], [3, 3]]
params_discriminator['batch_norm'] = [bn, bn, bn, bn, bn, bn]
params_discriminator['full'] = [64]
params_discriminator['minibatch_reg'] = False
params_discriminator['summary'] = True

params_generator = dict()
params_generator['stride'] = [2, 2, 2, 2, 1, 1]
params_generator['latent_dim'] = 100
params_generator['nfilter'] = [64, 256, 512, 256, 64, 1]
params_generator['shape'] = [[3, 3], [3, 3], [5, 5], [5, 5], [5, 5], [5, 5]]
params_generator['batch_norm'] = [bn, bn, bn, bn, bn]
params_generator['full'] = [4*4*64]
params_generator['summary'] = True
params_generator['non_lin'] = tf.nn.relu

params_optimization = dict()
params_optimization['gamma_gp'] = 10
params_optimization['batch_size'] = 16
params_optimization['gen_optimizer'] = 'rmsprop' # rmsprop / adam / sgd
params_optimization['disc_optimizer'] = 'rmsprop' # rmsprop / adam /sgd
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
params_cosmology['Nstats'] = 1000


params = dict()
params['generator'] = params_generator
params['discriminator'] = params_discriminator
params['optimization'] = params_optimization
params['cosmology'] = params_cosmology

params['normalize'] = False
params['image_size'] = [ns, ns]
params['prior_distribution'] = 'gaussian'
params['sum_every'] = 200
params['viz_every'] = 200
params['save_every'] = 5000
params['name'] = name
params['summary_dir'] = global_path + 'summaries_A8/' + params['name'] + '_' + time_str +'_summary/'
params['save_dir'] = global_path + 'models_A8/' + params['name'] + '_' + time_str + '_checkpoints/'

print("All params")
print(params)
print("\nDiscriminator Params")
print(params['discriminator'])
print("\nGenerator Params")
print(params['generator'])
print("\nOptimization Params")
print(params['optimization'])
print("\nCosmo Params")
print(params['cosmology'])
print()

if not os.path.exists(params['summary_dir']):
    os.makedirs(params['summary_dir'])
utils.save_dict_pickle(params['summary_dir'] + 'params.pkl', params)
utils.save_dict_for_humans(params['summary_dir'] + 'params.txt', params)
if not os.path.exists(params['save_dir']):
    os.makedirs(params['save_dir'])
utils.save_dict_pickle(params['save_dir'] + 'params.pkl', params)
utils.save_dict_for_humans(params['save_dir'] + 'params.txt', params)

resume, params = utils.test_resume(try_resume, params)


# Build the model
wgan = CosmoGAN(params, WGanModel)

filename = '/scratch/snx3000/rosenthj/data/nbody_{}Mpc_out_z0p000.h5'.format(Mpch)
images = utils.load_hdf5(filename=filename, dataset_name='data', mode='r')
images = forward(images)
images = np.reshape(images, (1, images.shape[0], images.shape[1], images.shape[2]))
dataset = Dataset.Dataset_2d(images, spix=ns, shuffle=True)

dataset = data.load.load_dataset(resolution=512, Mpch=Mpch, forward_map=forward, spix=ns)

wgan.train(dataset=dataset, resume=resume)
