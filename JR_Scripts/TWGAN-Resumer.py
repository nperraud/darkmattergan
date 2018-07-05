# coding: utf-8

import sys
sys.path.insert(0, '../')

import matplotlib
matplotlib.use('Agg')

import os
# import skimage.measure
from model import TemporalGanModelv3, TemporalGanModelv4, TemporalGanModelv5
from gan import TimeCosmoGAN
import utils
import pickle
from data import fmap, path, Dataset
import tensorflow as tf
import numpy as np
import functools


def load_gan_params(pathgan):
    """Load GAN object from path."""
    with open(os.path.join(pathgan, 'params.pkl'), 'rb') as f:
        params = pickle.load(f)
    return params

# Parameters
ns = 64
model_idx = 2
try_resume = True
Mpc_orig = 500
Mpc = Mpc_orig // (512 // ns)
params = load_gan_params(sys.argv[1])
utils.print_params_to_py_style_output(params)

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

resume, params = utils.test_resume(try_resume, params)

if resume:
    # Build the model
    twgan = TimeCosmoGAN(params)

    img_list = []

    filename = '/scratch/snx3000/rosenthj/data/nbody_{}Mpc_All.h5'.format(Mpc_orig)
    for box_idx in params['time']['classes']:
        images = utils.load_hdf5(filename=filename, dataset_name=str(box_idx), mode='r')
        images = params['cosmology']['forward_map'](images)
        #while images.shape[1] > ns:
        #    images = skimage.measure.block_reduce(images, (1,2,2), np.sum)
        img_list.append(images)

    images = np.array(img_list)
    print ("Images shape: {}".format(images.shape))
    dataset = Dataset.Dataset_time(images, spix=ns, shuffle=True)

    twgan.train(dataset=dataset, resume=resume)
else:
    print("Could not resume")
