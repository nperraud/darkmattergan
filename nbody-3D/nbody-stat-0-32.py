import tensorflow as tf
import os
from gantools import data, utils
from gantools.model import CosmoWGAN
from gantools.gansystem import GANsystem

forward = data.fmap.stat_forward
backward = data.fmap.stat_backward

# In[5]:

ns = 32
try_resume = True

time_str = 'stat-0_to_32'
global_path = '../saved_results/nbody/'
name = 'WGAN_' + time_str


bn = False

md=64

params_discriminator = dict()
params_discriminator['stride'] = [1, 1, 2, 2, 2]
params_discriminator['nfilter'] = [md, md, 2*md, 4*md, 8*md]
params_discriminator['shape'] = [[4, 4, 4],[4, 4, 4], [4, 4, 4],[4, 4, 4], [4, 4, 4]]
params_discriminator['batch_norm'] = [bn, bn, bn, bn, bn ]
params_discriminator['full'] = []
params_discriminator['minibatch_reg'] = False
params_discriminator['summary'] = True
params_discriminator['data_size'] = 3
params_discriminator['inception'] = False
params_discriminator['spectral_norm'] = True
# params_discriminator['histogram'] = dict()
# params_discriminator['histogram']['bins'] = 100
# params_discriminator['histogram']['data_size'] = 3
# params_discriminator['histogram']['full'] = 128
# params_discriminator['histogram']['spectral_norm'] = True

params_generator = dict()
params_generator['stride'] = [2, 2, 2, 1, 1]
params_generator['latent_dim'] = 256
params_generator['in_conv_shape'] =[4, 4, 4]
params_generator['nfilter'] = [4*md, 2*md, md, md, 1]
params_generator['shape'] = [[4, 4, 4],[4, 4, 4], [4, 4, 4],[4, 4, 4], [4, 4, 4]]
params_generator['batch_norm'] = [bn, bn, bn, bn]
params_generator['full'] = [256*md]
params_generator['summary'] = True
params_generator['non_lin'] = None
params_generator['data_size'] = 3
params_generator['inception'] = False
params_generator['spectral_norm'] = True

params_optimization = dict()
params_optimization['batch_size'] = 8
params_optimization['epoch'] = 100000
params_optimization['n_critic'] = 5
# params_optimization['generator'] = dict()
# params_optimization['generator']['optimizer'] = 'adam'
# params_optimization['generator']['kwargs'] = {'beta1':0, 'beta2':0.9}
# params_optimization['generator']['learning_rate'] = 0.0004
# params_optimization['discriminator'] = dict()
# params_optimization['discriminator']['optimizer'] = 'adam'
# params_optimization['discriminator']['kwargs'] = {'beta1':0, 'beta2':0.9}
# params_optimization['discriminator']['learning_rate'] = 0.0001

params_cosmology = dict()
params_cosmology['forward_map'] = forward
params_cosmology['backward_map'] = backward

params = dict()
params['net'] = dict()
params['net']['shape'] = [ns, ns, ns, 1]
params['net']['generator'] = params_generator
params['net']['gamma'] = 10
params['net']['discriminator'] = params_discriminator
params['net']['cosmology'] = params_cosmology
params['net']['loss_type'] = 'wasserstein'

params['optimization'] = params_optimization
params['summary_every'] = 100  # Tensorboard summaries every ** iterations
params['print_every'] = 50  # Console summaries every ** iterations
params['save_every'] = 1000  # Save the model every ** iterations
params['summary_dir'] = os.path.join(global_path, name + '_summary/')
params['save_dir'] = os.path.join(global_path, name + '_checkpoints/')
params['Nstats'] = 10

resume, params = utils.test_resume(try_resume, params)
params['summary_dir'] = os.path.join(global_path, name + '_summary/')
params['save_dir'] = os.path.join(global_path, name + '_checkpoints/')

wgan = GANsystem(CosmoWGAN, params)

dataset = data.load.load_nbody_dataset(
    spix=ns,
    Mpch=350,
    resolution=256,
    scaling=8,
    patch=False,
    augmentation=True,
    forward_map=forward,
    is_3d=True)

wgan.train(dataset, resume=resume)
