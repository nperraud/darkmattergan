# coding: utf-8

import os
import data

from model import LapGanModel
from gan import GAN

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# Parameters
ns = 64
nsamples = 4000
k = 10


# Data handling
images, raw_images = data.load_samples(nsamples = nsamples, permute=True, k=k)
images = data.make_smaller_samples(images, ns)
raw_images = data.make_smaller_samples(raw_images, ns)        


# Train a small GAN

def current_time_str():
    import time, datetime
    d = datetime.datetime.fromtimestamp(time.time())
    return str(d.year)+ '_' + str(d.month)+ '_' + str(d.day)+ '_' + str(d.hour)+ '_' + str(d.minute)


time_str = current_time_str()
global_path = '../../saved_result/'

up_scaling = 2
latent_dim = (ns//up_scaling)**2
bn = False

params_discriminator = dict()
params_discriminator['stride'] = [2, 2, 2,2 , 1]
params_discriminator['nfilter'] = [16, 128, 256, 128, 64]
params_discriminator['shape'] = [[5, 5],[5, 5], [5, 5], [3, 3], [3, 3]]
params_discriminator['batch_norm'] = [bn, bn, bn, bn, bn]
params_discriminator['full'] = [128]
params_discriminator['minibatch_reg'] = True
params_discriminator['summary'] = True

params_generator = dict()
params_generator['stride'] = [1, 1, 2, 1, 1]
params_generator['latent_dim'] = latent_dim
params_generator['nfilter'] = [64, 256, 128 ,64, 1]
params_generator['shape'] = [[5, 5], [5, 5], [5, 5], [5, 5], [5, 5]]
params_generator['batch_norm'] = [bn, bn, bn, bn]
params_generator['summary'] = True
params_generator['non_lin'] = 'tanh'
params_generator['upsampling'] = up_scaling

params_optimization = dict()
params_optimization['gamma_gp'] = 10
params_optimization['weight_l2'] = 0.1
params_optimization['batch_size'] = 16
params_optimization['gen_optimizer'] = 'adam' # rmsprop / adam / sgd
params_optimization['disc_optimizer'] = 'adam' # rmsprop / adam /sgd
params_optimization['disc_learning_rate'] = 1e-5
params_optimization['gen_learning_rate'] = 1e-5
params_optimization['beta1'] = 0.0
params_optimization['beta2'] = 0.9
params_optimization['epsilon'] = 1e-8
params_optimization['epoch'] = 50

params = dict()
params['generator'] = params_generator
params['discriminator'] = params_discriminator
params['optimization'] = params_optimization

params['image_size'] = [ns, ns]
params['sum_every'] = 200
params['viz_every'] = 200
params['save_every'] = 2000
params['name'] = 'LAPWGAN{}'.format(ns)
params['summary_dir'] = global_path + params['name'] + '_' + time_str +'summary/'
params['save_dir'] = global_path + params['name'] + '_' + time_str + 'checkpoints/'

params['clip_max_real'] = False
params['log_clip'] = 0.1
params['sigma_smooth'] = 1
params['k'] = k

obj = GAN(params, LapGanModel)


obj.train(images)
