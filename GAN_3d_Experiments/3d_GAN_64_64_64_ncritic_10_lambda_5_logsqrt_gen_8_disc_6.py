import sys
sys.path.insert(0, '../')

#import matplotlib
#matplotlib.use('Agg')

import numpy as np
import tensorflow as tf
import os
import data, utils
from model import WGanModel
from gan import CosmoGAN

def current_time_str():
    import time, datetime
    d = datetime.datetime.fromtimestamp(time.time())
    return str(d.year)+ '_' + str(d.month)+ '_' + str(d.day)+ '_' + str(d.hour)+ '_' + str(d.minute)

def load_3d_synthetic_samples():
	images, raw_images = data.load_3d_synthetic_samples(nsamples = nsamples,dim=ns, k=k)
	print("images shape: ", np.shape(images))
	print("raw_images shape: ", np.shape(raw_images))

def forward(X):
	return np.log(np.sqrt(X)+np.e)-2

def backward(Xmap, max_value=2e5):
	Xmap = np.clip(Xmap, -1.0, forward(max_value))
	tmp = np.exp((Xmap+2))-np.e
	return np.round(tmp*tmp)

if __name__ == "__main__":
	ns = 32
	nsamples = 1000
	k = 10
	try_resume = True

	#images, raw_images = load_3d_synthetic_samples(nsamples = nsamples,dim=ns, k=k)

	dataset = data.load.load_dataset(resolution=256, Mpch=350, spix=ns, forward_map=forward, scaling=4, is_3d=True)

	time_str = '64_64_64_ncritic_10_lambda_5_logsqrt_gen_8_disc_6' 
	global_path = '../saved_result/'
	name = 'WGAN{}'.format(ns)

	bn = False

	params_discriminator = dict()
	params_discriminator['stride'] = [2, 2, 2, 2, 1, 1]
	params_discriminator['nfilter'] = [128, 128, 64, 32, 16, 16]
	params_discriminator['shape'] = [[5, 5, 5], [5, 5, 5], [5, 5, 5], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
	params_discriminator['batch_norm'] = [bn, bn, bn, bn, bn, bn]
	params_discriminator['full'] = [32]
	params_discriminator['summary'] = True

	params_generator = dict()
	params_generator['stride'] = [2, 2, 2, 2, 1, 1, 1, 1]
	params_generator['latent_dim'] = 100
	params_generator['nfilter'] = [8, 32, 64, 128, 128, 64, 64, 1]
	params_generator['shape'] = [[3, 3, 3], [3, 3, 3], [5, 5, 5], [5, 5, 5], [5, 5, 5], [5, 5, 5], [5, 5, 5], [5, 5, 5]]
	params_generator['batch_norm'] = [bn, bn, bn, bn, bn, bn, bn]
	params_generator['full'] = [2*2*2*8]
	params_generator['summary'] = True
	params_generator['non_lin'] = None
	
	params_optimization = dict()
	params_optimization['n_critic'] = 10
	params_optimization['gamma_gp'] = 5
	params_optimization['batch_size'] = 8
	params_optimization['gen_optimizer'] = 'rmsprop' # rmsprop / adam / sgd
	params_optimization['disc_optimizer'] = 'rmsprop' # rmsprop / adam /sgd
	params_optimization['disc_learning_rate'] = 3e-5
	params_optimization['gen_learning_rate'] = 3e-5
	params_optimization['beta1'] = 0.0
	params_optimization['beta2'] = 0.9
	params_optimization['epsilon'] = 1e-8
	params_optimization['epoch'] = 500
	
	params_cosmology = dict()
	params_cosmology['clip_max_real'] = False
	params_cosmology['log_clip'] = 0.1
	params_cosmology['sigma_smooth'] = 1
	params_cosmology['forward_map'] = forward
	params_cosmology['backward_map'] = backward
	params_cosmology['k'] = k
	params_cosmology['Npsd'] = 500
	params_cosmology['max_num_psd'] = 100
	
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
	params['save_every'] = 4000
	params['name'] = name
	params['summary_dir'] = global_path + params['name'] + '_' + time_str +'summary/'
	params['save_dir'] = global_path + params['name'] + '_' + time_str + 'checkpoints/'

	resume, params = utils.test_resume(try_resume, params)

	wgan = CosmoGAN(params, WGanModel, is_3d=True)
	wgan.train(dataset, resume=resume)