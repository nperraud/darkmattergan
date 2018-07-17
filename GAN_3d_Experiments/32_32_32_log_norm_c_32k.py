import sys
sys.path.insert(0, '../')

#import matplotlib
#matplotlib.use('Agg')

import numpy as np
import tensorflow as tf
import os, functools
import data, utils
from model import WGanModel
from gan import CosmoGAN


if __name__ == "__main__":
	ns = 32
	nsamples = 1000
	try_resume = True
	latent_dim = 100
	Mpch = 350

	forward = functools.partial(data.fmap.log_norm_forward, c=32000.0, shift=10.373523)
	backward = functools.partial(data.fmap.log_norm_backward, c=32000.0, shift=10.373523)


	time_str = '32_32_32_log_norm_c_32k' 
	global_path = '../saved_result/'
	name = 'WGAN'

	bn = False

	params_discriminator = dict()
	params_discriminator['stride'] = [2, 2, 1, 1, 1, 1]
	params_discriminator['nfilter'] = [64, 64, 32, 16, 8, 2]
	params_discriminator['inception'] = True
	params_discriminator['batch_norm'] = [bn, bn, bn, bn, bn, bn]
	params_discriminator['full'] = [64, 16]
	params_discriminator['summary'] = True
	params_discriminator['minibatch_reg'] = False

	params_generator = dict()
	params_generator['stride'] = [2, 2, 2, 2, 1, 1, 1, 1]
	params_generator['latent_dim'] = latent_dim
	params_generator['nfilter'] = [8, 32, 64, 64, 64, 32, 32, 1]
	params_generator['inception'] = True
	params_generator['batch_norm'] = [bn, bn, bn, bn, bn, bn, bn]
	params_generator['full'] = [2*2*2*8]
	params_generator['summary'] = True
	params_generator['non_lin'] = tf.nn.relu

	params_optimization = dict()
	params_optimization['n_critic'] = 10
	params_optimization['gamma_gp'] = 5
	params_optimization['batch_size'] = 8
	params_optimization['gen_optimizer'] = 'rmsprop' # rmsprop / adam / sgd
	params_optimization['disc_optimizer'] = 'rmsprop' # rmsprop / adam /sgd
	params_optimization['disc_learning_rate'] = 3e-5
	params_optimization['gen_learning_rate'] = 3e-5
	params_optimization['beta1'] = 0.9
	params_optimization['beta2'] = 0.999
	params_optimization['epsilon'] = 1e-8
	params_optimization['epoch'] = 10000

	params_cosmology = dict()
	params_cosmology['clip_max_real'] = False
	params_cosmology['log_clip'] = 0.1
	params_cosmology['sigma_smooth'] = 1
	params_cosmology['forward_map'] = forward
	params_cosmology['backward_map'] = backward
	params_cosmology['Nstats'] = 2000
	
	params = dict()
	params['generator'] = params_generator
	params['discriminator'] = params_discriminator
	params['optimization'] = params_optimization
	params['cosmology'] = params_cosmology
	
	params['normalize'] = False
	params['image_size'] = [ns, ns, ns]
	params['prior_distribution'] = 'gaussian'
	params['sum_every'] = 200
	params['viz_every'] = 200
	params['print_every'] = 100
	params['save_every'] = 1000
	params['num_hists_at_once'] = 30
	params['name'] = name
	params['summary_dir'] = global_path + params['name'] + '_' + time_str +'summary/'
	params['save_dir'] = global_path + params['name'] + '_' + time_str + 'checkpoints/'


	resume, params = utils.test_resume(try_resume, params)

	wgan = CosmoGAN(params, WGanModel, is_3d=True)
	dataset = data.load.load_dataset_file(spix=ns, resolution=256,Mpch=Mpch, scaling=8, forward_map=params_cosmology['forward_map'], patch=False, is_3d=True)
	wgan.train(dataset, resume=resume)