
# coding: utf-8

import os, sys
import pickle

import sys
sys.path.insert(0, '../')

import data
import numpy as np
from model import LapGanModel, LapGanModelTanh
from gan import CosmoGAN
import utils

# Parameters

ns = 128
nsamples = 7500
k = 10
scalings = [2,2,2]
try_resume = True




time_str = '3hopes_final'

global_path = '../../../saved_result/'






level = 1

up_scaling = scalings[level]
new_ns = ns//np.prod(scalings[:level])
latent_dim = (new_ns//up_scaling)**2
bn = False
params_discriminator = dict()
params_discriminator['stride'] = [2, 2, 2, 2 , 1]
params_discriminator['nfilter'] = [64, 256, 512, 256, 64]
params_discriminator['shape'] = [[5, 5], [5, 5], [5, 5], [3, 3], [3, 3]]
params_discriminator['batch_norm'] = [bn, bn, bn, bn, bn]
params_discriminator['full'] = [64]
params_discriminator['summary'] = True
params_discriminator['non_lin'] = None
params_discriminator['minibatch_reg'] = True

params_generator = dict()
params_generator['stride'] = [1, 1, 2, 1, 1, 1]
params_generator['latent_dim'] = latent_dim
params_generator['nfilter'] = [64, 256, 512, 256, 64, 1]
params_generator['shape'] = [[5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5]]
params_generator['batch_norm'] = [bn, bn, bn, bn, bn]
params_generator['summary'] = True
params_generator['non_lin'] = 'tanh'
params_generator['upsampling'] = up_scaling

params_optimization = dict()
params_optimization['gamma_gp'] = 10
params_optimization['batch_size'] = 16
params_optimization['gen_optimizer'] = 'rmsprop' # rmsprop / adam / sgd
params_optimization['disc_optimizer'] = 'rmsprop' # rmsprop / adam /sgd
params_optimization['disc_learning_rate'] = 3e-5
params_optimization['gen_learning_rate'] = 3e-5
params_optimization['beta1'] = 0.5
params_optimization['beta2'] = 0.99
params_optimization['epsilon'] = 1e-8
params_optimization['epoch'] = 75


params_cosmology = dict()
params_cosmology['clip_max_real'] = False
params_cosmology['log_clip'] = 0.1
params_cosmology['sigma_smooth'] = 1
params_cosmology['k'] = k
params_cosmology['Npsd'] = 500

params = dict()
params['generator'] = params_generator
params['discriminator'] = params_discriminator
params['optimization'] = params_optimization
params['cosmology'] = params_cosmology

params['normalize'] = False
params['image_size'] = [new_ns, new_ns]
params['prior_distribution'] = 'gaussian'
params['sum_every'] = 200
params['viz_every'] = 200
params['save_every'] = 5000
params['name'] = 'LAPWGAN{}_level{}_'.format(ns, level)
params['summary_dir'] = global_path + params['name'] + '_' + time_str +'_summary/'
params['save_dir'] = global_path + params['name'] + '_' + time_str + '_checkpoints/'


resume, params = utils.test_resume(try_resume, params)


# Build the model

wgan = CosmoGAN(params, LapGanModel)

images, raw_images = data.load_samples(nsamples = nsamples, permute=True, k=k)
images = data.make_smaller_samples(images, ns)
raw_images = data.make_smaller_samples(raw_images, ns)   
down_sampled_images = data.down_sample_images(images, scalings)


# Train the model
wgan.train(down_sampled_images[level], resume=resume)






