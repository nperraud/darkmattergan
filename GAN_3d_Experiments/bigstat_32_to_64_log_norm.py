import sys
sys.path.insert(0, '../')

#import matplotlib
#matplotlib.use('Agg')

import numpy as np
import tensorflow as tf
import os, functools
import data, utils
from model import upscale_WGAN_pixel_CNN
from gan import CosmoGAN

def current_time_str():
    import time, datetime
    d = datetime.datetime.fromtimestamp(time.time())
    return str(d.year)+ '_' + str(d.month)+ '_' + str(d.day)+ '_' + str(d.hour)+ '_' + str(d.minute)


if __name__ == "__main__":
	ns = 32
	try_resume = True
	downsampling = 2
	latent_dim = ns**3
	Mpch = 350

	time_str = 'log_norm'
	global_path = '../saved_result/'
	name = 'bigstat_32_to_64'

	bn = False

	params_discriminator = dict()
	params_discriminator['stride'] = [2, 2, 2, 2, 1, 1]
	params_discriminator['nfilter'] = [128, 128, 64, 32, 16, 8]
	params_discriminator['shape'] = [[5, 5, 5], [5, 5, 5], [5, 5, 5], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
	params_discriminator['batch_norm'] = [bn, bn, bn, bn, bn, bn]
	params_discriminator['full'] = [64]
	params_discriminator['summary'] = True
	params_discriminator['minibatch_reg'] = False

	params_generator = dict()
	params_generator['downsampling'] = downsampling
	params_generator['stride'] = [1, 1, 1, 1, 1, 1, 1, 1]
	params_generator['y_layer'] = 0
	params_generator['latent_dim'] = latent_dim
	params_generator['nfilter'] = [8, 32, 64, 128, 128, 64, 64, 1]
	params_generator['shape'] = [[3, 3, 3], [3, 3, 3], [5, 5, 5], [5, 5, 5], [5, 5, 5], [5, 5, 5], [5, 5, 5], [5, 5, 5]]
	params_generator['batch_norm'] = [bn, bn, bn, bn, bn, bn, bn]
	params_generator['full'] = []
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
	params_optimization['epoch'] = 2000
	
	params_cosmology = dict()
	params_cosmology['clip_max_real'] = False
	params_cosmology['log_clip'] = 0.1
	params_cosmology['sigma_smooth'] = 1
	params_cosmology['forward_map'] = data.fmap.log_norm_forward
	params_cosmology['backward_map'] = data.fmap.log_norm_backward
	params_cosmology['Nstats'] = 1000
	
	params = dict()
	params['generator'] = params_generator
	params['discriminator'] = params_discriminator
	params['optimization'] = params_optimization
	params['cosmology'] = params_cosmology
	
	params['normalize'] = False
	params['image_size'] = [ns, ns, ns, 8]
	params['prior_distribution'] = 'gaussian'
	params['sum_every'] = 200
	params['viz_every'] = 200
	params['print_every'] = 100
	params['big_every'] = 500
	params['save_every'] = 1000
	params['num_hists_at_once'] = 30
	params['name'] = name
	params['summary_dir'] = global_path + params['name'] + '_' + time_str +'summary/'
	params['save_dir'] = global_path + params['name'] + '_' + time_str + 'checkpoints/'


	resume, params = utils.test_resume(try_resume, params)
	params['name'] = name
	params['summary_dir'] = global_path + params['name'] + '_' + time_str +'summary/'
	params['save_dir'] = global_path + params['name'] + '_' + time_str + 'checkpoints/'

	wgan = CosmoGAN(params, upscale_WGAN_pixel_CNN, is_3d=True)
	dataset = data.load.load_dataset_file(spix=ns, resolution=256,Mpch=Mpch, scaling=4, forward_map=params_cosmology['forward_map'], patch=True, is_3d=True)
	wgan.train(dataset, resume=resume)