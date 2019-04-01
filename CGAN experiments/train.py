import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from copy import deepcopy

import sys
sys.path.insert(0, '../')
from gantools import data
from gantools import utils
from gantools import plot
from gantools.model import ConditionalParamWGAN
from gantools.gansystem import GANsystem
from gantools import evaluation

ns = 128 # Resolution of the image
try_resume = True # Try to resume previous simulation

def non_lin(x):
    return tf.nn.relu(x)

dataset_train_shuffled_name = 'kids_train_shuffled.h5'

dataset = data.load.load_params_dataset(filename=dataset_train_shuffled_name, batch=15000, shape=[ns, ns], transform=data.transformation.random_transpose_2d)

time_str = '2D'
global_path = '/scratch/snx3000/nperraud/saved_results/'

name = 'KidsConditional{}'.format(ns) + '_smart_' + time_str

bn = False

params_discriminator = dict()
params_discriminator['stride'] = [1, 2, 2, 2, 2]
params_discriminator['nfilter'] = [32, 64, 128, 256, 512]
params_discriminator['shape'] = [[7, 7], [5, 5], [5, 5], [5,5], [3,3]]
params_discriminator['batch_norm'] = [bn, bn, bn, bn, bn]
params_discriminator['full'] = [512, 256, 128]
params_discriminator['minibatch_reg'] = False
params_discriminator['summary'] = True
params_discriminator['data_size'] = 2

params_generator = dict()
params_generator['stride'] = [2, 2, 2, 2, 1]
params_generator['latent_dim'] = 64
params_generator['nfilter'] = [256, 128, 64, 32, 1]
params_generator['shape'] = [[3, 3], [5, 5], [5, 5], [5, 5], [7,7]]
params_generator['batch_norm'] = [bn, bn, bn, bn]
params_generator['full'] = [256, 512, 8 * 8 * 512]
params_generator['summary'] = True
params_generator['non_lin'] = non_lin
params_generator['data_size'] = 2

params_optimization = dict()
params_optimization['optimizer'] = 'rmsprop'
params_optimization['batch_size'] = 64
params_optimization['learning_rate'] = 1e-5
params_optimization['epoch'] = 10

# all parameters
params = dict()
params['net'] = dict() # All the parameters for the model
params['net']['generator'] = params_generator
params['net']['discriminator'] = params_discriminator
params['net']['shape'] = [ns, ns, 1] # Shape of the image
params['net']['gamma_gp'] = 10 # Gradient penalty

# Conditional params
params['net']['prior_normalization'] = False
params['net']['cond_params'] = 2
params['net']['init_range'] = [[0.101, 0.487], [0.487, 1.331]]
params['net']['prior_distribution'] = "gaussian_length"
params['net']['final_range'] = [0.1*np.sqrt(params_generator['latent_dim']), 1*np.sqrt(params_generator['latent_dim'])]

params['optimization'] = params_optimization
params['optimization']['discriminator'] = deepcopy(params_optimization)
params['optimization']['generator'] = deepcopy(params_optimization)
params['summary_every'] = 5000 # Tensorboard summaries every ** iterations
params['print_every'] = 2500 # Console summaries every ** iterations
params['save_every'] = 25000 # Save the model every ** iterations
params['duality_every'] = 5
params['summary_dir'] = os.path.join(global_path, name +'_summary/')
params['save_dir'] = os.path.join(global_path, name + '_checkpoints/')
params['Nstats'] = 2000

resume, params = utils.test_resume(try_resume, params)
# If a model is reloaded and some parameters have to be changed, then it should be done here.
# For example, setting the number of epoch to 5 would be:
params['optimization']['epoch'] = 5
params['summary_dir'] = os.path.join(global_path, name +'_summary/')
params['save_dir'] = os.path.join(global_path, name + '_checkpoints/')

wgan = GANsystem(ConditionalParamWGAN, params)

wgan.train(dataset, resume=resume)
