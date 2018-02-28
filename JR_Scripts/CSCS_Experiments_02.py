
# coding: utf-8

import os, sys, utils
import pickle
import numpy as np
from Data_Generators import time_toy_generator
from model import WNGanModel
from gan import CosmoGAN

import sys
# sys.path.insert(0, '../')


def current_time_str():
    import time, datetime
    d = datetime.datetime.fromtimestamp(time.time())
    return str(d.year) + '_' + str(d.month)+ '_' + str(d.day) + '_' + str(d.hour) + '_' + str(d.minute)


# # Parameters

# In[4]:

num_classes = 1
ns = 128
k = 8

try_resume = False

time_str = 'Feb28'
global_path = '../../../saved_result/'
#time_str = current_time_str()

name = 'WNGanModel{}'.format(ns)

params_optimization = dict()
params_optimization['gamma_gp'] = 10
params_optimization['weight_l2'] = 0.1
params_optimization['batch_size'] = 8
params_optimization['gen_optimizer'] = 'adam'
params_optimization['disc_optimizer'] = 'adam'
params_optimization['disc_learning_rate'] = 3e-5
params_optimization['gen_learning_rate'] = 3e-5
params_optimization['beta1'] = 0.99
params_optimization['beta2'] = 0.999
params_optimization['epsilon'] = 1e-9
params_optimization['epoch'] = 100

# params_optimization = dict()
# params_optimization['gamma_gp'] = 10
# params_optimization['weight_l2'] = 0.1
# params_optimization['batch_size'] = 16
# params_optimization['gen_optimizer'] = 'rmsprop' # rmsprop / adam / sgd
# params_optimization['disc_optimizer'] = 'rmsprop' # rmsprop / adam /sgd
# params_optimization['disc_learning_rate'] = 3e-5
# params_optimization['gen_learning_rate'] = 3e-5
# params_optimization['beta1'] = 0.5
# params_optimization['beta2'] = 0.99
# params_optimization['epsilon'] = 1e-8
# params_optimization['epoch'] = 100


bn = True

params_discriminator = dict()
params_discriminator['stride'] = [2, 2, 2, 2, 2, 1]
params_discriminator['nfilter'] = [16, 32, 64, 128, 256, 64]
params_discriminator['shape'] = [[3,3], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5]]
params_discriminator['batch_norm'] = [bn, bn, bn, bn, bn, bn]
params_discriminator['full'] = [32]
params_discriminator['summary'] = True

params_generator = dict()
params_generator['stride'] = [2, 2, 2, 2, 2, 1]
params_generator['latent_dim'] = 25
params_generator['nfilter'] = [64, 256, 128, 64, 32, 1]
params_generator['shape'] = [[5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [3, 3]]
params_generator['batch_norm'] = [bn, bn, bn, bn, bn]
params_generator['full'] = [1024]
params_generator['summary'] = True
params_generator['non_lin'] = 'tanh'

# params_discriminator = dict()
# params_discriminator['stride'] = [2, 2, 2, 1, 1]
# params_discriminator['nfilter'] = [16, 256, 512, 256, 16]
# params_discriminator['shape'] = [[5, 5], [5, 5], [5, 5], [5, 5], [3, 3]]
# params_discriminator['batch_norm'] = [bn, bn, bn, bn, bn]
# params_discriminator['full'] = [32]
# params_discriminator['summary'] = True
# params_discriminator['minibatch_reg'] = True

# params_generator = dict()
# params_generator['stride'] = [2, 2, 2, 1, 1, 1]
# params_generator['latent_dim'] = (new_ns//2)**2
# params_generator['nfilter'] = [64, 256, 512, 256, 64, 1]
# params_generator['shape'] = [[3, 3], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5]]
# params_generator['batch_norm'] = [bn, bn, bn, bn, bn]
# params_generator['full'] = []
# params_generator['summary'] = True
# params_generator['non_lin'] = 'tanh'

params_cosmology = dict()
params_cosmology['clip_max_real'] = False
params_cosmology['log_clip'] = 0.1
params_cosmology['sigma_smooth'] = 1
params_cosmology['k'] = k
params_cosmology['Npsd'] = 500

# params_cosmology = dict()
# params_cosmology['clip_max_real'] = False
# params_cosmology['log_clip'] = 0.1
# params_cosmology['sigma_smooth'] = 1
# params_cosmology['k'] = k
# params_cosmology['Npsd'] = 500

params = dict()
params['generator'] = params_generator
params['discriminator'] = params_discriminator
params['optimization'] = params_optimization
params['cosmology'] = params_cosmology

params['num_classes'] = num_classes
params['prior_distribution'] = 'gaussian'
params['sum_every'] = 200
params['viz_every'] = 200
params['save_every'] = 5000
params['normalize'] = False
params['image_size'] = [128, 128]
params['name'] = name
params['summary_dir'] = global_path + params['name'] + '_' + time_str +'_summary/'
params['save_dir'] = global_path + params['name'] + '_' + time_str + '_checkpoints/'
# params['summary_dir'] = 'tboard/' + params['name'] + '_' + time_str + 'summary/'
# params['save_dir'] = 'checkp/' + params['name'] + '_' + time_str + 'checkpoints/'

# params['prior_distribution'] = 'gaussian'
# params['sum_every'] = 200
# params['viz_every'] = 200
# params['save_every'] = 5000
# params['normalize'] = False
# params['image_size'] = [new_ns*up_scaling, new_ns*up_scaling]
# params['name'] = name
# params['summary_dir'] = global_path + params['name'] + '_' + time_str +'_summary/'
# params['save_dir'] = global_path + params['name'] + '_' + time_str + '_checkpoints/'

resume = False

if try_resume:
    try:
        with open(params['save_dir']+'params.pkl', 'rb') as f:
            params = pickle.load(f)
        resume = True
        print('Resume, the training will start from the last iteration!')
    except:
        print('No resume, the training will start from the beginning!')

gan = CosmoGAN(params, WNGanModel)

print("All params")
print(params)
print("\nDiscriminator Params")
print(params['discriminator'])
print("\nGenerator Params")
print(params['generator'])
print("\nOptimization Params")
print(params['optimization'])
print()

data = time_toy_generator.gen_dataset(width=128, images_per_time_step=8000, point_density_factor=3)
data = np.asarray([data[9]])
data = data.swapaxes(0, 1)
data = data.reshape((data.shape[0] * data.shape[1], data.shape[2], data.shape[3]))
data = data.astype(np.float32)
data = utils.forward_map(data, params['cosmology']['k'])

# In[6]:

gan.train(data)
