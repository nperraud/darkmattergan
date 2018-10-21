import tensorflow as tf
import os
from gantools import data, utils
from gantools.model import UpscalePatchWGAN, CosmoWGAN
from gantools.gansystem import GANsystem

ns = 32
try_resume = False
latent_dim = 32 * 32 * 32

time_str = '64_to_256'
global_path = '../saved_results/nbody/'
name = 'WGAN_' + time_str

bn = False

md = 8

params_discriminator = dict()
params_discriminator['stride'] = [2, 2, 2, 1, 1, 1]
params_discriminator['nfilter'] = [md, md * 8, md * 8, md * 8, md, md]
params_discriminator['inception'] = True
params_discriminator['batch_norm'] = [bn, bn, bn, bn, bn, bn]
params_discriminator['full'] = []
params_discriminator['summary'] = True
params_discriminator['minibatch_reg'] = False
params_discriminator['data_size'] = 3

params_generator = dict()
params_generator['stride'] = [1, 1, 1, 1, 1, 1]
params_generator['latent_dim'] = latent_dim
params_generator['nfilter'] = [md, 2 * md, 2 * md, 2 * md, 2 * md, 1]
params_generator['inception'] = True
params_generator['batch_norm'] = [bn, bn, bn, bn, bn]
params_generator['full'] = []
params_generator['summary'] = True
params_generator['non_lin'] = tf.nn.relu
params_generator['data_size'] = 3

params_cosmology = dict()
params_cosmology['forward_map'] = data.fmap.log_norm_forward
params_cosmology['backward_map'] = data.fmap.log_norm_backward

params_optimization = dict()
params_optimization['n_critic'] = 10
params_optimization['batch_size'] = 8
params_optimization['epoch'] = 10000

params = dict()
params['net'] = dict()
params['net']['shape'] = [ns, ns, ns, 8]
params['net']['generator'] = params_generator
params['net']['gamma'] = 10
params['net']['discriminator'] = params_discriminator
params['net']['upscaling'] = 4
params['net']['cosmology'] = params_cosmology

params['optimization'] = params_optimization
params['summary_every'] = 100  # Tensorboard summaries every ** iterations
params['print_every'] = 50  # Console summaries every ** iterations
params['save_every'] = 1000  # Save the model every ** iterations
params['summary_dir'] = os.path.join(global_path, name + '_summary/')
params['save_dir'] = os.path.join(global_path, name + '_checkpoints/')
params['Nstats'] = 10

resume, params = utils.test_resume(try_resume, params)


class CosmoUpscalePatchWGAN(UpscalePatchWGAN, CosmoWGAN):
    pass


wgan = GANsystem(CosmoUpscalePatchWGAN, params)

dataset = data.load.load_nbody_dataset(
    spix=ns,
    scaling=1,
    resolution=256,
    Mpch=350,
    patch=True,
    augmentation=True,
    is_3d=True)

wgan.train(dataset, resume=resume)