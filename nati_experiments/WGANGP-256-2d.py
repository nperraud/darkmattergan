# coding: utf-8

import sys
sys.path.insert(0, '../')

import matplotlib
matplotlib.use('Agg')
import os
import tensorflow as tf

from gantools import data
from gantools import utils
from gantools import plot
from gantools.model import WGAN, CosmoWGAN
from gantools.gansystem import GANsystem
from gantools.data import fmap
from gantools import evaluation
import functools
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np

# Parameters
ns = 32 # Resolution of the image
try_resume = True # Try to resume previous simulation
Mpch = 70 # Type of dataset (select 70 or 350)
# Do not change these for now
shift = 3
c = 40000
forward = functools.partial(fmap.stat_forward, shift=shift, c=c)
backward = functools.partial(fmap.stat_backward, shift=shift, c=c)
def non_lin(x):
    return tf.nn.relu(x)

dataset = data.load.load_dataset(spix=ns, Mpch=Mpch, forward_map=forward)

X = dataset.get_all_data().flatten()
time_str = '2D'
global_path = 'saved_results'

name = '4newWGAN{}'.format(ns)

bn = False

# Parameters for the discriminator
params_discriminator = dict()
params_discriminator['stride'] = [2, 2, 2, 1]
params_discriminator['nfilter'] = [16, 64, 256, 32]
params_discriminator['shape'] = [[5, 5],[5, 5], [5, 5], [3, 3]]
params_discriminator['batch_norm'] = [bn, bn, bn, bn]
params_discriminator['full'] = [32]
params_discriminator['minibatch_reg'] = False
params_discriminator['summary'] = True
params_discriminator['is_3d'] = False

# Parameters for the generator
params_generator = dict()
params_generator['stride'] = [1, 1, 2, 1, 1]
params_generator['latent_dim'] = 16*16*32
params_generator['nfilter'] = [32, 64, 256, 32, 1]
params_generator['shape'] = [[5, 5], [5, 5],[5, 5], [5, 5], [5, 5]]
params_generator['batch_norm'] = [bn, bn, bn, bn]
params_generator['full'] = []
params_generator['summary'] = True
params_generator['non_lin'] = non_lin
params_generator['is_3d'] = False

# Optimization parameters
d_opt = dict()
d_opt['optimizer'] = "rmsprop"
d_opt['learning_rate'] = 3e-5
d_opt['kwargs'] = dict()
params_optimization = dict()
params_optimization['batch_size'] = 16
params_optimization['epoch'] = 10
params_optimization['discriminator'] = deepcopy(d_opt)
params_optimization['generator'] = deepcopy(d_opt)
params_optimization['n_critic'] = 5

# all parameters
params = dict()
params['net'] = dict() # All the parameters for the model
params['net']['generator'] = params_generator
params['net']['discriminator'] = params_discriminator
params['net']['prior_distribution'] = 'gaussian'
params['net']['shape'] = [ns, ns, 1] # Shape of the image
params['net']['is_3d'] = False
params['net']['gamma_gp'] = 10 # Gradient penalty

params['optimization'] = params_optimization
params['summary_every'] = 100 # Tensorboard summaries every ** iterations
params['print_every'] = 50 # Console summaries every ** iterations
params['save_every'] = 1000 # Save the model every ** iterations
params['summary_dir'] = os.path.join(global_path, name +'_summary/')
params['save_dir'] = os.path.join(global_path, name + '_checkpoints/')


resume, params = utils.test_resume(try_resume, params)
# params['optimization']['disc_learning_rate'] = 3e-6
# params['optimization']['gen_learning_rate'] = 3e-6


# Build the model
wgan = GANsystem(CosmoWGAN, params )


# # Train the model

# In[83]:


# wgan.train(dataset, resume=resume)
wgan.calculate_metrics(dataset, resume=resume)

# def compute_metrics(real, fake):
#     from gantools.metric import ganlist
#     metric_list = ganlist.cosmo_metric_list()
#     d = []
#     for metr in metric_list:
#         d.append(metr(fake, real))
#     plt.plot(range(0, len(d)), d)
#     return (np.mean(np.array(d)), *d)
#
# def single_metric(real, fake):
#     return compute_metrics(real, fake)[0]
# N = 2000 # Number of samples
# gen_sample = np.squeeze(wgan.generate(N=N))
# raw_images = backward(dataset.get_samples(dataset.N))
# gen_sample_raw = backward(gen_sample)
# print("The global metric is {}".format(single_metric(raw_images, gen_sample_raw)))

