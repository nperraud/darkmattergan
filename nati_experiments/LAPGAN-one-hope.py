
# coding: utf-8



import numpy as np
import os
import sys
import pickle
sys.path.insert(0, '../')

os.environ["CUDA_VISIBLE_DEVICES"]="0"
import data

from model import LapGanModel
from gan import CosmoGAN

# # Parameters


ns = 128
scalings = [8]
nsamples = 7500
k = 10
try_resume = True

# def current_time_str():
#     import time, datetime
#     d = datetime.datetime.fromtimestamp(time.time())
#     return str(d.year)+ '_' + str(d.month)+ '_' + str(d.day)+ '_' + str(d.hour)+ '_' + str(d.minute)

# time_str = current_time_str() 

time_str = 'test'
global_path = '../../../saved_result/'

name = 'LAPWGAN1hope{}'.format(ns)



params_optimization = dict()
params_optimization['gamma_gp'] = 10
params_optimization['weight_l2'] = 0.1
params_optimization['batch_size'] = 16
params_optimization['gen_optimizer'] = 'rmsprop' # rmsprop / adam / sgd
params_optimization['disc_optimizer'] = 'rmsprop' # rmsprop / adam /sgd
params_optimization['disc_learning_rate'] = 1e-5
params_optimization['gen_learning_rate'] = 1e-5
params_optimization['beta1'] = 0.5
params_optimization['beta2'] = 0.99
params_optimization['epsilon'] = 1e-8
params_optimization['epoch'] = 100





params_optimization['epoch'] = 150

level = 0

up_scaling = scalings[level]
new_ns = ns//np.prod(scalings[:level+1])
latent_dim = new_ns**2
bn = False
params_discriminator = dict()
params_discriminator['stride'] = [2, 2, 2, 1, 1]
params_discriminator['nfilter'] = [16, 256, 512, 256, 16]
params_discriminator['shape'] = [[5, 5], [5, 5], [5, 5], [5, 5], [3, 3]]
params_discriminator['batch_norm'] = [bn, bn, bn, bn, bn]
params_discriminator['full'] = [32]
params_discriminator['summary'] = True
params_discriminator['minibatch_reg'] = True

params_generator = dict()
params_generator['stride'] = [2, 2, 2, 1, 1, 1]
params_generator['latent_dim'] = new_ns*new_ns
params_generator['nfilter'] = [64, 256, 512, 256, 64, 1]
params_generator['shape'] = [[5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5]]
params_generator['batch_norm'] = [bn, bn, bn, bn, bn]
params_generator['full'] = []
params_generator['summary'] = True
params_generator['non_lin'] = 'tanh'
params_generator['upsampling'] = up_scaling

params_cosmology = dict()
params_cosmology['clip_max_real'] = False
params_cosmology['log_clip'] = 0.1
params_cosmology['sigma_smooth'] = 1
params_cosmology['k'] = k
params_cosmology['Npsd'] = 500

params = dict()
params['generator'] = params_generator
params['discriminator'] = params_discriminator
params['optimization'] = params_optimization
params['cosmology'] = params_cosmology

params['prior_distribution'] = 'gaussian'
params['sum_every'] = 200
params['viz_every'] = 200
params['save_every'] = 5000

params['normalize'] = False
params['image_size'] = [new_ns*up_scaling, new_ns*up_scaling]
params['name'] = name
params['summary_dir'] = global_path + params['name'] + '_' + time_str +'_summary/'
params['save_dir'] = global_path + params['name'] + '_' + time_str + '_checkpoints/'



resume = False

if try_resume:
    try:
        with open(params['save_dir']+'params.pkl', 'rb') as f:
            params = pickle.load(f)
        resume = True
        print('Resume, the training will start from the last iteration!')
    except:
        print('No resume, the training will start from the beginning!')


obj = CosmoGAN(params, LapGanModel)



images, raw_images = data.load_samples(nsamples = nsamples, permute=True, k=k)
images = data.make_smaller_samples(images, ns)
raw_images = data.make_smaller_samples(raw_images, ns)



down_sampled_images = data.down_sample_images(images, scalings)



obj.train(X=down_sampled_images[level], resume=resume)
