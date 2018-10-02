import os

import numpy as np
import tensorflow as tf

from gantools import data
from gantools import utils
from gantools.model import WGAN, LapWGAN
from gantools.gansystem import GANsystem

downscale = 16
try_resume = True
ns=2**15//downscale

time_str = 'nsynth_{}'.format(ns)


global_path = '../saved_results/nsynth'
name = 'WGAN' + '_' + time_str


non_lin = tf.nn.tanh
bn = False

params_discriminator = dict()
params_discriminator['stride'] = [1, 2, 2, 2, 2, 1]
params_discriminator['nfilter'] = [16, 64, 256, 256, 256, 32]
params_discriminator['shape'] = [[9], [9], [7], [5], [5], [3]]
params_discriminator['batch_norm'] = [bn, bn, bn, bn, bn, bn]
params_discriminator['full'] = [32]
params_discriminator['minibatch_reg'] = False
params_discriminator['summary'] = True
params_discriminator['data_size'] = 1

params_generator = dict()
params_generator['stride'] = [1, 1, 1, 1, 1 , 1, 1]
params_generator['latent_dim'] = ns
params_generator['nfilter'] = [4, 64, 256, 256, 256, 32, 1]
params_generator['shape'] = [[5], [7], [9], [9], [9], [9], [9] ]
params_generator['batch_norm'] = [bn, bn, bn, bn, bn, bn]
params_generator['full'] = []
params_generator['summary'] = True
params_generator['non_lin'] = non_lin
params_generator['data_size'] = 1

params_optimization = dict()
params_optimization['batch_size'] = 16
params_optimization['epoch'] = 500


# all parameters
params = dict()
params['net'] = dict() # All the parameters for the model
params['net']['generator'] = params_generator
params['net']['discriminator'] = params_discriminator
params['net']['prior_distribution'] = 'gaussian'
params['net']['shape'] = [ns, 1] # Shape of the image
params['net']['gamma_gp'] = 10 # Gradient penalty
params['net']['upsampling'] = 4

params['optimization'] = params_optimization
params['summary_every'] = 100 # Tensorboard summaries every ** iterations
params['print_every'] = 50 # Console summaries every ** iterations
params['save_every'] = 1000 # Save the model every ** iterations
params['summary_dir'] = os.path.join(global_path, name +'_summary/')
params['save_dir'] = os.path.join(global_path, name + '_checkpoints/')
params['Nstats'] = 100

resume, params = utils.test_resume(try_resume, params)

wgan = GANsystem(LapWGAN, params)
dataset = data.load.load_nsynth_dataset(scaling=downscale)
wgan.train(dataset, resume=resume)
