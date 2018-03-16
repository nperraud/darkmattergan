import numpy as np
import tensorflow as tf
import os, sys, time
import utils, optimization, metrics, plot, data
from model import WGanModel, LapGanModel
from gan import CosmoGAN

def current_time_str():
    import time, datetime
    d = datetime.datetime.fromtimestamp(time.time())
    return str(d.year)+ '_' + str(d.month)+ '_' + str(d.day)+ '_' + str(d.hour)+ '_' + str(d.minute)

if __name__ == "__main__":
	ns = 16
	nsamples = 1000
	k = 10

	#mages, raw_images = data.load_3d_synthetic_samples(nsamples = nsamples,dim=ns, k=k)
	#print("images shape: ", np.shape(images))
	#print("raw_images shape: ", np.shape(raw_images))

	path_3d_hists = '../3d_smaller_cubes'
	hists_3d, raw_hists_3d = data.load_3d_hists(path_3d_hists, k)
	print("hists_3d shape: ", np.shape(hists_3d))
	print("raw_hists_3d shape: ", np.shape(raw_hists_3d))
	nsamples = len(raw_hists_3d)
	print("Total samples={}".format(nsamples))	

	time_str = current_time_str() 
	global_path = '../saved_result/'
	name = 'WGAN{}'.format(ns)

	bn = False

	params_discriminator = dict()
	params_discriminator['stride'] = [2, 2, 2, 1]
	params_discriminator['nfilter'] = [32, 32, 32, 16]
	params_discriminator['shape'] = [[5, 5, 5],[5, 5, 5], [3, 3, 3], [3, 3, 3]]
	params_discriminator['batch_norm'] = [bn, bn, bn, bn]
	params_discriminator['full'] = [32]
	params_discriminator['summary'] = True

	params_generator = dict()
	params_generator['stride'] = [2, 2, 2, 1, 1]
	params_generator['latent_dim'] = 100
	params_generator['nfilter'] = [8, 32, 64, 64, 1]
	params_generator['shape'] = [[3, 3, 3], [3, 3, 3], [5, 5, 5], [5, 5, 5], [5, 5, 5]]
	params_generator['batch_norm'] = [bn, bn, bn, bn]
	params_generator['full'] = [2*2*2*8]
	params_generator['summary'] = True
	params_generator['non_lin'] = 'tanh'
	
	params_optimization = dict()
	params_optimization['gamma_gp'] = 10
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
	params['print_every'] = 200
	params['save_every'] = 4000
	params['name'] = name
	params['summary_dir'] = global_path + params['name'] + '_' + time_str +'summary/'
	params['save_dir'] = global_path + params['name'] + '_' + time_str + 'checkpoints/'
	params['file_input'] = True
	params['samples_dir_paths'] = utils.get_3d_hists_dir_paths('../3d_smaller_cubes/')

	wgan = CosmoGAN(params, WGanModel, is_3d=True)
	wgan.train(images)