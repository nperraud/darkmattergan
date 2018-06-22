# coding: utf-8

import sys
sys.path.insert(0, '../')

import matplotlib
matplotlib.use('Agg')

import os
from gan import TimeCosmoGAN
from model import TemporalGanModelv3, TemporalGanModelv4, TemporalGanModelv5
import utils
import pickle
from data import Dataset
import tensorflow as tf
import numpy as np
import itertools


def load_gan_params(pathgan):
    """Load GAN object from path."""
    with open(os.path.join(pathgan, 'params.pkl'), 'rb') as f:
        params = pickle.load(f)
    return params


def save_dict(params):
    if not os.path.exists(params['summary_dir']):
        os.makedirs(params['summary_dir'])
    utils.save_dict_pickle(params['summary_dir'] + 'params.pkl', params)
    utils.save_dict_for_humans(params['summary_dir'] + 'params.txt', params)
    if not os.path.exists(params['save_dir']):
        os.makedirs(params['save_dir'])
    utils.save_dict_pickle(params['save_dir'] + 'params.pkl', params)
    utils.save_dict_for_humans(params['save_dir'] + 'params.txt', params)

Mpc = 500
ns = 64

params = load_gan_params(sys.argv[1])

sum_dir = params['summary_dir']
save_dir = params['save_dir']
q = ""
p = ""
for i in range(5):
    q = q + sum_dir.split('/')[i] + '/'
    p = p + save_dir.split('/')[i] + '/'
q = q + 'summaries_TCosmo_64/' + sum_dir.split('/')[6].replace("32","64") + '/'
p = p + 'models_TCosmo_64/' + save_dir.split('/')[6].replace("32","64") + '/'
params['summary_dir'] = q
params['save_dir'] = p


s = params['discriminator']['stride']
params['discriminator']['stride'] = [s[0], s[1], s[2], 2, s[3], s[4]]
s = params['discriminator']['nfilter']
params['discriminator']['nfilter'] = [s[0], s[1], s[2], s[2], s[3], s[4]]
s = params['discriminator']['shape']
params['discriminator']['shape'] = [s[0], s[1], s[2], [5, 5], s[3], s[4]]
params['discriminator']['batch_norm'] = [False] * len(params['discriminator']['nfilter'])

s = params['generator']['stride']
params['generator']['stride'] = [s[0], s[1], s[2], 2, s[3], s[4], s[5]]
s = params['generator']['nfilter']
params['generator']['nfilter'] = [s[0], s[1], 2*s[1], s[2], s[3], s[4], s[5]]
params['generator']['latent_dim'] = utils.get_latent_dim(ns, params['generator'])
s = params['generator']['shape']
params['generator']['shape'] = [s[0], s[1], s[2], [5, 5], s[3], s[4], s[5]]
params['generator']['batch_norm'] = [False] * (len(params['generator']['nfilter']) - 1)

params['optimization']['batch_size'] = 8

params['image_size'] = [ns, ns]

try_resume = False
resume, params = utils.test_resume(try_resume, params)

model = None
if params['time']['model_idx'] == 2:
    model = TemporalGanModelv3
if params['time']['model_idx'] == 3:
    model = TemporalGanModelv4
if params['time']['model_idx'] == 4:
    model = TemporalGanModelv5

# Build the model
twgan = TimeCosmoGAN(params, model)

img_list = []

filename = '/scratch/snx3000/rosenthj/data/nbody_{}Mpc_All.h5'.format(Mpc)
for box_idx in params['time']['classes']:
    images = utils.load_hdf5(filename=filename, dataset_name=str(box_idx), mode='r')
    images = params['cosmology']['forward_map'](images)
    img_list.append(images)

images = np.array(img_list)
dataset = Dataset.Dataset_time(images, spix=params['image_size'][0], shuffle=True)

save_dict(params)

twgan.train(dataset=dataset, resume=resume)