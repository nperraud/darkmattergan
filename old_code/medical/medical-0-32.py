import tensorflow as tf
import os
from gantools import data, utils
from gantools.model import WGAN
from gantools.gansystem import GANsystem


ns = 32
try_resume = True

time_str = '0_to_32'
global_path = '../saved_results/medical/'
name = 'WGAN_' + time_str


bn = False

md=64

params_discriminator = dict()
params_discriminator['stride'] = [1, 1, 2, 2, 2]
params_discriminator['nfilter'] = [md, md, 2*md, 4*md, 8*md]
params_discriminator['shape'] = [[5, 5, 5],[5, 5, 5], [5, 5, 5],[5, 5, 5], [5, 5, 5]]
params_discriminator['batch_norm'] = [bn, bn, bn, bn, bn ]
params_discriminator['full'] = []
params_discriminator['minibatch_reg'] = False
params_discriminator['summary'] = True
params_discriminator['data_size'] = 3
params_discriminator['inception'] = False
params_discriminator['spectral_norm'] = True

params_generator = dict()
params_generator['stride'] = [2, 2, 2, 1, 1]
params_generator['latent_dim'] = 100
params_generator['in_conv_shape'] =[4, 4, 4]
params_generator['nfilter'] = [4*md, 2*md, md, md, 1]
params_generator['shape'] = [[5, 5, 5],[5, 5, 5], [5, 5, 5],[5, 5, 5], [5, 5, 5]]
params_generator['batch_norm'] = [bn, bn, bn, bn]
params_generator['full'] = [4*4*4*8*md]
params_generator['summary'] = True
params_generator['non_lin'] = None
params_generator['data_size'] = 3
params_generator['inception'] = False
params_generator['spectral_norm'] = True

params_optimization = dict()
params_optimization['batch_size'] = 8
params_optimization['epoch'] = 100000
params_optimization['n_critic'] = 5

params = dict()
params['net'] = dict()
params['net']['shape'] = [ns, ns, ns, 1]
params['net']['generator'] = params_generator
params['net']['gamma'] = 10
params['net']['discriminator'] = params_discriminator
params['net']['loss_type'] = 'normalized_wasserstein' # loss ('hinge' or 'wasserstein')

params['optimization'] = params_optimization
params['summary_every'] = 100  # Tensorboard summaries every ** iterations
params['print_every'] = 50  # Console summaries every ** iterations
params['save_every'] = 1000  # Save the model every ** iterations
params['summary_dir'] = os.path.join(global_path, name + '_summary/')
params['save_dir'] = os.path.join(global_path, name + '_checkpoints/')
params['Nstats'] = 10

resume, params = utils.test_resume(try_resume, params)
params['optimization']['epoch'] = 100000

wgan = GANsystem(WGAN, params)

dataset = data.load.load_medical_dataset(spix=ns, scaling=8, patch=False, augmentation=True)

wgan.train(dataset, resume=resume)
