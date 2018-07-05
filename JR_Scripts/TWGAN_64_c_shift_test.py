import sys
sys.path.insert(0, '../')

import matplotlib
matplotlib.use('Agg')

import data
from model import TemporalGanModelv3, TemporalGanModelv4, TemporalGanModelv5
from gan import TimeCosmoGAN
import utils
from data import fmap, Dataset
import tensorflow as tf
import functools
import os
import numpy as np
import tensorflow as tf


# Parameters
ns = 64
model_idx = 2
try_resume = False
Mpc_orig = 500
Mpc = Mpc_orig // (512 // ns)
shift = 16
c = 40000

# General Params
params = dict()
params['normalize'] = False
params['image_size'] = [64, 64]
params['prior_distribution'] = "laplacian"
params['sum_every'] = 200
params['viz_every'] = 200
params['save_every'] = 5000
params['name'] = "TWGANv5:v232_6-5_"
params['summary_dir'] = "/scratch/snx3000/rosenthj/results/summaries_TCosmo_64/TWGANv5:v264_6-5__0-6r_cs_16x8chCDF-Mom500_summary/"
params['save_dir'] = "/scratch/snx3000/rosenthj/results/models_TCosmo_64/TWGANv5:v264_6-5__0-6r_cs_16x8chCDF-_checkpoints/"
params['print_every'] = 100
params['resume'] = False
params['has_enc'] = False

# Generator Params
params_generator = dict()
params_generator['stride'] = [2, 2, 2, 2, 1, 1, 1]
params_generator['nfilter'] = [64, 256, 512, 256, 128, 64, 1]
params_generator['latent_dim'] = 1024
params_generator['shape'] = [[3, 3], [3, 3], [3, 3], [5, 5], [5, 5], [5, 5], [5, 5]]
params_generator['batch_norm'] = [False, False, False, False, False, False]
params_generator['full'] = []
params_generator['summary'] = True
params_generator['non_lin'] = tf.nn.relu
params_generator['y_layer'] = None
params_generator['one_pixel_mapping'] = []
params_generator['is_3d'] = False
params['generator'] = params_generator

# Discriminator Params
params_discriminator = dict()
params_discriminator['stride'] = [2, 2, 2, 2, 1, 1]
params_discriminator['nfilter'] = [16, 128, 256, 256, 128, 64]
params_discriminator['shape'] = [[5, 5], [5, 5], [5, 5], [5, 5], [3, 3], [3, 3]]
params_discriminator['batch_norm'] = [False, False, False, False, False, False]
params_discriminator['full'] = [64]
params_discriminator['cdf'] = 16
params_discriminator['channel_cdf'] = 8
params_discriminator['minibatch_reg'] = False
params_discriminator['summary'] = True
params_discriminator['non_lin'] = None
params_discriminator['one_pixel_mapping'] = []
params_discriminator['moment'] = None
params_discriminator['is_3d'] = False
params['discriminator'] = params_discriminator

# Optimization Params
params_optimization = dict()
params_optimization['gamma_gp'] = 10
params_optimization['batch_size'] = 8
params_optimization['gen_optimizer'] = 'adam'
params_optimization['disc_optimizer'] = 'adam'
params_optimization['disc_learning_rate'] = 3e-05
params_optimization['gen_learning_rate'] = 3e-05
params_optimization['beta1'] = 0.9
params_optimization['beta2'] = 0.99
params_optimization['epsilon'] = 1e-08
params_optimization['epoch'] = 1000
params_optimization['batch_size_gen'] = 64
params_optimization['optimizer'] = 'adam'
params_optimization['learning_rate'] = 3e-05
params_optimization['n_critic'] = 5
params_optimization['enc_optimizer'] = 'adam'
params_optimization['enc_learning_rate'] = 3e-05
params['optimization'] = params_optimization

# Cosmology Params
params_cosmology = dict()
params_cosmology['clip_max_real'] = True
params_cosmology['log_clip'] = 0.1
params_cosmology['sigma_smooth'] = 1
params_cosmology['forward_map'] = functools.partial(fmap.stat_forward, shift=shift, c=c)
params_cosmology['backward_map'] = functools.partial(fmap.stat_forward, shift=shift, c=c)
params_cosmology['Nstats'] = 1000
params['cosmology'] = params_cosmology

# Time Params
params_time = dict()
params_time['num_classes'] = 4
params_time['classes'] = [6, 4, 2, 0]
params_time['class_weights'] = [0.8, 0.9, 1.0, 1.1]
params_time['model_idx'] = 4
params_time['use_diff_stats'] = False
params['time'] = params_time

resume, params = utils.test_resume(try_resume, params)

model = None
if params_time['model_idx'] == 2:
    model = TemporalGanModelv3
if params_time['model_idx'] == 3:
    model = TemporalGanModelv4
if params_time['model_idx'] == 4:
    model = TemporalGanModelv5

# Build the model
twgan = TimeCosmoGAN(params, model)

img_list = []

filename = '/scratch/snx3000/rosenthj/data/nbody_{}Mpc_All.h5'.format(Mpc_orig)
for box_idx in params['time']['classes']:
    images = utils.load_hdf5(filename=filename, dataset_name=str(box_idx), mode='r')
    images = params['cosmology']['forward_map'](images)
    # while images.shape[1] > ns:
    #    images = skimage.measure.block_reduce(images, (1,2,2), np.sum)
    img_list.append(images)

images = np.array(img_list)
print("Images shape: {}".format(images.shape))
dataset = Dataset.Dataset_time(images, spix=ns, shuffle=True)

twgan.train(dataset=dataset, resume=resume)