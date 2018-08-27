# coding: utf-8

import sys
sys.path.insert(0, '../')

import matplotlib
matplotlib.use('Agg')

import os
from gan import TimeCosmoGAN
import utils
import pickle
from data import Dataset
import numpy as np


def load_gan_params(gan_path):
    """Load GAN object from path."""
    with open(os.path.join(gan_path, 'params.pkl'), 'rb') as f:
        params = pickle.load(f)
    return params

# Parameters
ns = 64
try_resume = True
Mpc_orig = 500
Mpc = Mpc_orig // (512 // ns)
divisor = 3

params = load_gan_params(sys.argv[1])
utils.print_param_dict(params)

encoder_net = utils.NetParamHelper()
encoder_net.add_conv_layer(64, stride=1, shape=3)
encoder_net.add_conv_layer(128, stride=1, shape=3)
encoder_net.add_conv_layer(128, stride=1, shape=3)
encoder_net.add_conv_layer(128, stride=1, shape=3)
encoder_net.add_conv_layer(128, stride=1, shape=3)
encoder_net.add_conv_layer(256, stride=2, shape=5)
encoder_net.add_conv_layer(512, stride=2, shape=3)
encoder_net.add_conv_layer(256, stride=2, shape=3)
encoder_net.add_conv_layer(64, stride=2, shape=3)
params_encoder = encoder_net.params
params_encoder['full'] = []
params_encoder['summary'] = True

params['encoder'] = params_encoder
params['optimization']['enc_optimizer'] = params['optimization']['gen_optimizer']
params['optimization']['enc_learning_rate'] = params['optimization']['gen_learning_rate']

# Build the model
twgan = TimeCosmoGAN(params)

utils.print_param_dict(twgan.params)

img_list = []

filename = '/scratch/snx3000/rosenthj/data/nbody_{}Mpc_All.h5'.format(Mpc_orig)
for box_idx in params['time']['classes']:
    images = utils.load_hdf5(filename=filename, dataset_name=str(box_idx), mode='r')
    images = params['cosmology']['forward_map'](images / divisor)
    #while images.shape[1] > ns:
    #    images = skimage.measure.block_reduce(images, (1,2,2), np.sum)
    img_list.append(images)

images = np.array(img_list)
print ("Images shape: {}".format(images.shape))
dataset = Dataset.Dataset_time(images, spix=ns, shuffle=True)

twgan.train_encoder(dataset=dataset)