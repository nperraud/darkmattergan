# coding: utf-8

import sys
sys.path.insert(0, '../')

import matplotlib
matplotlib.use('Agg')

import os
# import skimage.measure
from model import TemporalGenericGanModel
from gan import TimeCosmoGAN
import utils, blocks
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


def get_class_weights(cls):
    weights = []
    for i in range(len(cls)):
        weights.append(1.3 - (0.08*cls[i]))
    return weights

# Parameters
ns = 64
model_idx = 5
divisor = 3
try_resume = False
Mpc_orig = 500
Mpc = Mpc_orig // (512 // ns)
cl = []
for i in range(len(sys.argv)-1):
    cl.append(int(sys.argv[i+1]))

time_encoding = 'channel_encoding'
ten = ''

if time_encoding == 'channel_encoding':
    ten = 'ce'
elif time_encoding == 'scale_full':
    ten = 'sf'
elif time_encoding == 'scale_half':
    ten = 'sh'


def get_model_name(params):
    r = 'R' if params['time']['model']['relative'] else ''
    act = '_selu' if params['generator']['activation'] == blocks.selu else ''
    act = '_lrelu' if params['generator']['activation'] == blocks.lrelu else act
    sn = '_sn' if params['discriminator']['spectral_norm'] else ''
    l = 'F' if params['optimization'].get('JS-regularization', False) else 'W'
    return 'T{}{}GAN{}:{}d{}{}{}{}-{}'.format(r, l, ten, Mpc, divisor, act, sn, len(params['generator']['nfilter']),
                                               len(params['discriminator']['nfilter']))


shift = 3
bandwidth = 20000
forward = functools.partial(fmap.stat_forward, shift=shift, c=bandwidth)
backward = functools.partial(fmap.stat_backward, shift=shift, c=bandwidth)

#time_str = '0r-24-6r_0811_16x8chCDF-Mom{}'.format(Mpch)
cl_str = ''
for cl_id in cl:
    cl_str = cl_str + str(cl_id)
time_str = '{}r_Hlr3e5_bs4_v2ad_sf{}'.format(cl_str, Mpc)
global_path = '/scratch/snx3000/rosenthj/results/'

bnd = False

discriminator_net = utils.NetParamHelper()
discriminator_net.add_conv_layer(32 * (len(cl) + 1), stride=1, shape=5)
discriminator_net.add_conv_layer(32, stride=1, shape=1)
discriminator_net.add_conv_layer(96, stride=2, shape=5)
discriminator_net.add_conv_layer(128, stride=2, shape=5)
discriminator_net.add_conv_layer(256, stride=2, shape=5)
discriminator_net.add_conv_layer(256, stride=2, shape=5)
discriminator_net.add_conv_layer(128, stride=1, shape=3)
discriminator_net.add_conv_layer(128, stride=1, shape=3)
discriminator_net.add_conv_layer(128, stride=1, shape=3)
discriminator_net.add_conv_layer(64, stride=1, shape=3)
discriminator_net.add_full(64)

params_discriminator = discriminator_net.params
params_discriminator['spectral_norm'] = True
params_discriminator['separate_first'] = True
# params_discriminator['cdf'] = 32
# params_discriminator['channel_cdf'] = 16
# params_discriminator['moment'] = [5,5]
params_discriminator['minibatch_reg'] = False
params_discriminator['summary'] = True
params_cdf = dict()
# params_cdf['cdf_in'] = 32
params_cdf['channel_cdf'] = 32
params_cdf['cdf_out'] = 64
#params_discriminator['cdf_block'] = params_cdf
params_hist = dict()
params_hist['full'] = 64
params_hist['bins'] = 32
params_hist['initial_range'] = 3
# params_discriminator['histogram'] = params_hist

bng = False

generator_net = utils.NetParamHelper()
generator_net.add_conv_layer(64, stride=2, shape=3)
generator_net.add_conv_layer(256, stride=2, shape=3)
generator_net.add_conv_layer(512, stride=2, shape=3)
generator_net.add_conv_layer(256, stride=2, shape=5)
generator_net.add_conv_layer(128, stride=1, shape=3)
generator_net.add_conv_layer(128, stride=1, shape=3)
generator_net.add_conv_layer(128, stride=1, shape=3)
generator_net.add_conv_layer(128, stride=1, shape=3)
generator_net.add_conv_layer(64, stride=1, shape=3)
generator_net.add_conv_layer(1, stride=1, shape=5, batch_norm=None)

params_generator = generator_net.params
params_generator['latent_dim'] = utils.get_latent_dim(ns, params_generator)
params_generator['summary'] = True
params_generator['non_lin'] = tf.nn.relu
params_generator['activation'] = blocks.selu

params_optimization = dict()
params_optimization['gamma_gp'] = 10
# params_optimization['JS-regularization'] = True
params_optimization['batch_size'] = 4
params_optimization['gen_optimizer'] = 'adam' # rmsprop / adam / sgd
params_optimization['disc_optimizer'] = 'adam' # rmsprop / adam /sgd
params_optimization['disc_learning_rate'] = 3e-5
params_optimization['gen_learning_rate'] = 3e-5
params_optimization['beta1'] = 0.5
params_optimization['beta2'] = 0.99
params_optimization['epsilon'] = 1e-8
params_optimization['epoch'] = 1000
params_optimization['n_critic'] = 5

params_cosmology = dict()
params_cosmology['clip_max_real'] = True
params_cosmology['log_clip'] = 0.1
params_cosmology['sigma_smooth'] = 1
params_cosmology['forward_map'] = forward
params_cosmology['backward_map'] = backward
params_cosmology['Nstats'] = 1000

params_time = dict()
params_time['classes'] = cl
params_time['num_classes'] = len(cl)
params_time['class_weights'] = get_class_weights(cl)
# params_time['class_weights'] = [(1.3 - (0.08*cl[0])), (1.3 - (0.08*cl[1]))]
# params_time['class_weights'] = [0.8, 1.2]
assert len(params_time['classes']) == len(params_time['class_weights'])
params_time['use_diff_stats'] = False

params_time['model'] = dict()
params_time['model']['time_encoding'] = time_encoding
params_time['model']['relative'] = False  # Dont forget n_critics when changing this

params_optimization['batch_size_gen'] = params_optimization['batch_size'] * params_time['num_classes']

params = dict()
params['generator'] = params_generator
params['discriminator'] = params_discriminator
params['optimization'] = params_optimization
params['cosmology'] = params_cosmology
params['time'] = params_time

name = get_model_name(params)
dir_suffix = ''
if params_time['num_classes'] > 1:
    dir_suffix = '_C{}'.format(params_time['num_classes'])

params['normalize'] = False
params['image_size'] = [ns, ns]
params['prior_distribution'] = 'laplacian'
params['sum_every'] = 800
params['viz_every'] = 800
params['save_every'] = 10000
params['name'] = name
params['summary_dir'] = global_path + 'summaries_{}x{}{}/'.format(ns,ns,dir_suffix) + params['name'] + '_' + time_str +'_summary/'
params['save_dir'] = global_path + 'models_{}x{}{}/'.format(ns,ns,dir_suffix) + params['name'] + '_' + time_str + '_checkpoints/'


resume, params = utils.test_resume(try_resume, params)

# Build the model
twgan = TimeCosmoGAN(params, TemporalGenericGanModel)

utils.print_param_dict(twgan.params)

img_list = []

filename = '/scratch/snx3000/rosenthj/data/nbody_{}Mpc_All.h5'.format(Mpc_orig)
for box_idx in params['time']['classes']:
    images = utils.load_hdf5(filename=filename, dataset_name=str(box_idx), mode='r')
    images = forward(images / divisor)
    # while images.shape[1] > ns:
    #    images = skimage.measure.block_reduce(images, (1,2,2), np.sum)
    img_list.append(images)

images = np.array(img_list)
print ("Images shape: {}".format(images.shape))
dataset = Dataset.Dataset_time(images, spix=ns, shuffle=True)

save_dict(params)

twgan.train(dataset=dataset, resume=resume)
