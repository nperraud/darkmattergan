
# coding: utf-8

import sys
sys.path.insert(0, '../')

import data

from model import WGanModel
from gan import CosmoGAN
import pickle


# Parameters

ns = 64
nsamples = 10000
k = 10
try_resume = True


# def current_time_str():
#     import time, datetime
#     d = datetime.datetime.fromtimestamp(time.time())
#     return str(d.year)+ '_' + str(d.month)+ '_' + str(d.day)+ '_' + str(d.hour)+ '_' + str(d.minute)

# time_str = current_time_str()

time_str = 'final'
global_path = '../../../saved_result/'

name = 'WGAN{}'.format(ns)

bn = False

params_discriminator = dict()
params_discriminator['stride'] = [2, 2, 2, 2, 1]
params_discriminator['nfilter'] = [16, 64, 128,  256, 64]
params_discriminator['shape'] = [[5, 5],[5, 5], [3, 3], [3, 3], [3, 3]]
params_discriminator['batch_norm'] = [bn, bn, bn, bn, bn]
params_discriminator['full'] = [128]
params_discriminator['summary'] = True

params_generator = dict()
params_generator['stride'] = [2, 2, 2, 2, 1]
params_generator['latent_dim'] = 100
params_generator['nfilter'] = [64, 256, 128, 64, 1]
params_generator['shape'] = [[3, 3], [3, 3], [5, 5], [5, 5], [5, 5]]
params_generator['batch_norm'] = [bn, bn, bn, bn]
params_generator['full'] = [4*4*64]
params_generator['summary'] = True
params_generator['non_lin'] = 'tanh'

params_optimization = dict()
params_optimization['gamma_gp'] = 10
params_optimization['batch_size'] = 16
params_optimization['gen_optimizer'] = 'rmsprop' # rmsprop / adam / sgd
params_optimization['disc_optimizer'] = 'rmsprop' # rmsprop / adam /sgd
params_optimization['disc_learning_rate'] = 3e-5
params_optimization['gen_learning_rate'] = 3e-5
params_optimization['beta1'] = 0.9
params_optimization['beta2'] = 0.9999
params_optimization['epsilon'] = 1e-8
params_optimization['epoch'] = 50

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

params['normalize'] = False
params['image_size'] = [ns, ns]
params['prior_distribution'] = 'gaussian'
params['sum_every'] = 200
params['viz_every'] = 200
params['save_every'] = 2000
params['name'] = 'WGAN{}'.format(ns)
params['summary_dir'] = global_path + params['name'] + '_' + time_str +'summary/'
params['save_dir'] = global_path + params['name'] + '_' + time_str + 'checkpoints/'

resume = False

if try_resume:
    try:
        with open(params['save_dir'] + 'params.pkl', 'rb') as f:
            params = pickle.load(f)
        resume = True
        print('Resume, the training will start from the last iteration!')
    except:
        print('No resume, the training will start from the beginning!')



params['optimization']['epoch']=300
# Build the model
wgan = CosmoGAN(params, WGanModel)

images, raw_images = data.load_samples(nsamples = nsamples, permute=True, k=k)
images = data.make_smaller_samples(images, ns)
raw_images = data.make_smaller_samples(raw_images, ns)   

# Train the model
wgan.train(images, resume=resume)