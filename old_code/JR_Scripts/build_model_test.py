# coding: utf-8

import sys
sys.path.insert(0, '../')

import matplotlib
matplotlib.use('Agg')

import os
# import skimage.measure
from model import TemporalGenericGanModel
from gan import TimeCosmoGAN
import utils
from data import fmap, path, Dataset
import tensorflow as tf
import numpy as np
import functools


def save_dict(params):
    if not os.path.exists(params['summary_dir']):
        os.makedirs(params['summary_dir'])
    utils.save_dict_pickle(params['summary_dir'] + 'params.pkl', params)
    utils.save_dict_for_humans(params['summary_dir'] + 'params.txt', params)
    if not os.path.exists(params['save_dir']):
        os.makedirs(params['save_dir'])
    utils.save_dict_pickle(params['save_dir'] + 'params.pkl', params)
    utils.save_dict_for_humans(params['save_dir'] + 'params.txt', params)


# Parameters
ns = 32
try_resume = False
Mpch = 500

time_encoding = 'channel_encoding'
ten = ''

if time_encoding == 'channel_encoding':
    ten = 'ce'
elif time_encoding == 'scale_full':
    ten = 'sf'
elif time_encoding == 'scale_half':
    ten = 'sh'

shift = 3
bandwidth = 20000
forward = functools.partial(fmap.stat_forward, shift=shift, c=bandwidth)
backward = functools.partial(fmap.stat_backward, shift=shift, c=bandwidth)

time_str = '0r-24-6r_0811_{}'.format(Mpch)
global_path = './test/'

name = 'TWGANv4:v2{}_6-5_'.format(ns)

bn = False

params_discriminator = dict()
params_discriminator['stride'] = [2, 2, 2, 1, 1]
params_discriminator['nfilter'] = [16, 128, 256, 128, 64]
params_discriminator['shape'] = [[5, 5],[5, 5],[5, 5], [3, 3], [3, 3]]
params_discriminator['batch_norm'] = [bn] * len(params_discriminator['nfilter'])
params_discriminator['full'] = [64]
#params_discriminator['cdf'] = 256
#params_discriminator['channel_cdf'] = 128
#params_discriminator['moment'] = [5,5]
params_discriminator['minibatch_reg'] = False
params_discriminator['summary'] = True
params_cdf = dict()
params_cdf['cdf_out'] = 32
params_cdf['channel_cdf'] = 16
params_discriminator['cdf_block'] = params_cdf
params_hist = dict()
params_hist['bla'] = 5
#params_discriminator['histogram'] = params_hist

params_generator = dict()
params_generator['stride'] = [2, 2, 2, 1, 1, 1]
params_generator['nfilter'] = [64, 256, 256, 128, 64, 1]
params_generator['latent_dim'] = utils.get_latent_dim(ns, params_generator)
params_generator['shape'] = [[3, 3], [3, 3], [3, 3], [5, 5], [5, 5], [5, 5]]
params_generator['batch_norm'] = [bn] * (len(params_generator['nfilter']) - 1)
params_generator['full'] = []
params_generator['summary'] = True
params_generator['non_lin'] = tf.nn.relu

params_optimization = dict()
params_optimization['gamma_gp'] = 1
params_optimization['batch_size'] = 16
params_optimization['gen_optimizer'] = 'adam' # rmsprop / adam / sgd
params_optimization['disc_optimizer'] = 'adam' # rmsprop / adam /sgd
params_optimization['disc_learning_rate'] = 1e-5
params_optimization['gen_learning_rate'] = 1e-5
params_optimization['beta1'] = 0.9
params_optimization['beta2'] = 0.99
params_optimization['epsilon'] = 1e-8
params_optimization['epoch'] = 1000
params_optimization['JS-regularization'] = True

params_cosmology = dict()
params_cosmology['clip_max_real'] = True
params_cosmology['log_clip'] = 0.1
params_cosmology['sigma_smooth'] = 1
params_cosmology['forward_map'] = forward
params_cosmology['backward_map'] = backward
params_cosmology['Nstats'] = 1000

params_time = dict()
params_time['num_classes'] = 4
params_time['classes'] = [6, 4, 2, 0]
params_time['class_weights'] = [0.8, 0.9, 1.0, 1.1]
params_time['model_idx'] = 4
params_time['use_diff_stats'] = False

params_time['model'] = dict()
params_time['model']['time_encoding'] = time_encoding
params_time['model']['relative'] = False

params_optimization['batch_size_gen'] = params_optimization['batch_size'] * params_time['num_classes']

params = dict()
params['generator'] = params_generator
params['discriminator'] = params_discriminator
params['optimization'] = params_optimization
params['cosmology'] = params_cosmology
params['time'] = params_time

params['normalize'] = False
params['image_size'] = [ns, ns]
params['prior_distribution'] = 'gaussian'
params['sum_every'] = 200
params['viz_every'] = 200
params['save_every'] = 5000
params['name'] = name
params['summary_dir'] = global_path + 'summaries_32_C2_v5/' + params['name'] + '_' + time_str +'_summary/'
params['save_dir'] = global_path + 'models_32_C2/' + params['name'] + '_' + time_str + '_checkpoints/'

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
print("\nTime Params")
print(params['time'])
print()

# Build the model
twgan = TimeCosmoGAN(params, TemporalGenericGanModel)
