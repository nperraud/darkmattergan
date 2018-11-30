import os

import tensorflow as tf

from gantools import data
from gantools import utils
from gantools.model import UpscalePatchWGANBorders
from gantools.gansystem import GANsystem

downscale = 4
try_resume = True
ns=2**15//downscale

time_str = 'piano_{}'.format(ns)


global_path = '../saved_results/piano'
name = 'WGAN' + '_' + time_str

bn = False

md = 64

params_discriminator = dict()
params_discriminator['stride'] = [1, 2, 2, 2, 1]
params_discriminator['nfilter'] = [2*md, 2*md, 2*md, 2*md, 1*md]
params_discriminator['shape'] = [[25], [25], [25], [25], [25]]
params_discriminator['batch_norm'] = [bn, bn, bn, bn, bn]
params_discriminator['full'] = []
params_discriminator['minibatch_reg'] = False
params_discriminator['summary'] = True
params_discriminator['data_size'] = 1

params_generator = dict()
params_generator['stride'] = [1, 1, 1, 1, 1]
params_generator['latent_dim'] = 16*16
params_generator['nfilter'] = [md, md, md, md, 1]
params_generator['shape'] = [[25], [25], [25], [25], [25]]
params_generator['batch_norm'] = [bn, bn, bn, bn]
params_generator['full'] = []
params_generator['summary'] = True
params_generator['non_lin'] = tf.nn.tanh
params_generator['data_size'] = 1
params_generator['borders'] = dict()
params_generator['borders']['width_full'] = None
params_generator['borders']['nfilter'] = [4, 7]
params_generator['borders']['batch_norm'] = [bn, bn]
params_generator['borders']['shape'] = [[25], [25]]
params_generator['borders']['stride'] = [2, 4]
params_generator['borders']['data_size'] = 1
params_generator['borders']['width_full'] = 32

params_optimization = dict()
params_optimization['batch_size'] = 64
params_optimization['epsilon'] = 1e-8
params_optimization['epoch'] = 10000

# all parameters
params = dict()
params['net'] = dict() # All the parameters for the model
params['net']['generator'] = params_generator
params['net']['discriminator'] = params_discriminator
params['net']['prior_distribution'] = 'gaussian'
params['net']['shape'] = [512, 2] # Shape of the image
params['net']['gamma_gp'] = 10 # Gradient penalty
params['net']['upsampling'] = 4
params['net']['fs'] = 16000//downscale

params['optimization'] = params_optimization
params['summary_every'] = 2000 # Tensorboard summaries every ** iterations
params['print_every'] = 1000 # Console summaries every ** iterations
params['save_every'] = 10000 # Save the model every ** iterations
params['summary_dir'] = os.path.join(global_path, name +'_summary/')
params['save_dir'] = os.path.join(global_path, name + '_checkpoints/')

resume, params = utils.test_resume(try_resume, params)

wgan = GANsystem(UpscalePatchWGANBorders, params)
dataset = data.load.load_audio_dataset(scaling=downscale, patch=False, spix=512, augmentation=True, smooth=4, type='piano')
wgan.train(dataset, resume=resume)
