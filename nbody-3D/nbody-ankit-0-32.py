import tensorflow as tf
import os
from gantools import data, utils
from gantools.model import CosmoWGAN
from gantools.gansystem import GANsystem

# In[5]:

ns = 32
try_resume = True

time_str = '0_to_32'
global_path = '../saved_results/nbody/'
name = 'WGAN_ankit_' + time_str

params_discriminator = {
    'batch_norm': [False, False, False, False, False, False],
    'data_size': 3,
    'full': [64, 16],
    'inception': True,
    'minibatch_reg': False,
    'nfilter': [64, 64, 32, 16, 8, 2],
    'non_lin': None,
    'one_pixel_mapping': [],
    'stride': [2, 2, 1, 1, 1, 1],
    'summary': True}

params_generator = {'batch_norm': [False,
    False,
    False,
    False,
    False,
    False,
    False],
   'data_size': 3,
   'full': [64],
   'inception': True,
   'latent_dim': 100,
   'nfilter': [8, 32, 64, 64, 64, 32, 32, 1],
   'non_lin': tf.nn.relu,
   'one_pixel_mapping': [],
   'residual': False,
   'stride': [2, 2, 2, 2, 1, 1, 1, 1],
   'summary': True}

params_optimization = {
    'batch_size': 8,
    'discriminator': {'kwargs': {}, 'learning_rate': 3e-05, 'optimizer': 'rmsprop'},
    'epoch': 10000,
    'generator': {'kwargs': {}, 'learning_rate': 3e-05, 'optimizer': 'rmsprop'},
    'n_critic': 10}


params_cosmology = dict()
params_cosmology['forward_map'] = data.fmap.log_norm_forward
params_cosmology['backward_map'] = data.fmap.log_norm_backward

params = dict()
params['net'] = dict()
params['net']['shape'] = [ns, ns, ns, 1]
params['net']['generator'] = params_generator
params['net']['gamma'] = 5
params['net']['discriminator'] = params_discriminator
params['net']['cosmology'] = params_cosmology
params['net']['prior_distribution'] = 'gaussian'

params['optimization'] = params_optimization
params['summary_every'] = 100  # Tensorboard summaries every ** iterations
params['print_every'] = 50  # Console summaries every ** iterations
params['save_every'] = 1000  # Save the model every ** iterations
params['summary_dir'] = os.path.join(global_path, name + '_summary/')
params['save_dir'] = os.path.join(global_path, name + '_checkpoints/')
params['Nstats'] = 10

resume, params = utils.test_resume(try_resume, params)

wgan = GANsystem(CosmoWGAN, params)

dataset = data.load.load_nbody_dataset(
    spix=ns,
    Mpch=350,
    resolution=256,
    scaling=8,
    patch=False,
    augmentation=True,
    forward_map=data.fmap.log_norm_forward,
    is_3d=True)

wgan.train(dataset, resume=resume)
