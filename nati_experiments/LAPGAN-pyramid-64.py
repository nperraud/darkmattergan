
# coding: utf-8

# In[2]:



# In[3]:


import numpy as np
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import utils, optimization, metrics, plot, data

from model import WGanModel, LapGanModel
from gan import GAN

# # Parameters

# In[4]:


ns = 64
scalings = [2, 2, 2]
nsamples = 4000
k = 10


# # Data handling

# Load the data

# In[5]:


images, raw_images = data.load_samples(nsamples = nsamples, permute=True, k=k)
images = data.make_smaller_samples(images, ns)
raw_images = data.make_smaller_samples(raw_images, ns)


# In[6]:


down_sampled_images = data.down_sample_images(images, scalings)


# Let us plot 16 images


# # A) Different GAN training

# In[47]:


def current_time_str():
    import time, datetime
    d = datetime.datetime.fromtimestamp(time.time())
    return str(d.year)+ '_' + str(d.month)+ '_' + str(d.day)+ '_' + str(d.hour)+ '_' + str(d.minute)

time_str = current_time_str() 
global_path = '../../saved_result/'


# In[51]:


params0 = dict()
params0['prior_distribution'] = 'gaussian'
params0['clip_max_real'] = False
params0['log_clip'] = 0.1
params0['sigma_smooth'] = 1
params0['k'] = k
params0['sum_every'] = 200
params0['viz_every'] = 400
params0['save_every'] = 2000

params_optimization = dict()
params_optimization['gamma_gp'] = 10
params_optimization['weight_l2'] = 0.1
params_optimization['batch_size'] = 16
params_optimization['gen_optimizer'] = 'adam' # rmsprop / adam / sgd
params_optimization['disc_optimizer'] = 'adam' # rmsprop / adam /sgd
params_optimization['disc_learning_rate'] = 3e-5
params_optimization['gen_learning_rate'] = 3e-5
params_optimization['beta1'] = 0.0
params_optimization['beta2'] = 0.9
params_optimization['epsilon'] = 1e-8
params_optimization['epoch'] = 50








level = 3

new_ns = ns//np.prod(scalings)
latent_dim = new_ns**2
bn = False
params_discriminator = dict()
params_discriminator['stride'] = [2, 2 , 1]
params_discriminator['nfilter'] = [16, 32, 64]
params_discriminator['shape'] = [[5, 5], [3, 3], [3, 3]]
params_discriminator['batch_norm'] = [bn, bn, bn]
params_discriminator['full'] = [128]
params_discriminator['summary'] = True
params_discriminator['minibatch_reg'] = True

params_generator = dict()
params_generator['stride'] = [2, 2, 1]
params_generator['latent_dim'] = latent_dim
params_generator['nfilter'] = [128 ,64, 1]
params_generator['shape'] = [[3, 3], [3, 3], [5, 5]]
params_generator['batch_norm'] = [bn, bn]
params_generator['full'] = [2*2*128]
params_generator['summary'] = True
params_generator['non_lin'] = None


params = params0.copy()
params['generator'] = params_generator
params['discriminator'] = params_discriminator
params['optimization'] = params_optimization

params['normalize'] = True
params['image_size'] = [new_ns, new_ns]
params['name'] = 'LAPWGAN{}_level{}_'.format(ns, level)
params['summary_dir'] = global_path + params['name'] + '_' + time_str +'summary/'
params['save_dir'] = global_path + params['name'] + '_' + time_str + 'checkpoints/'


obj = GAN(params, WGanModel)
obj.train(down_sampled_images[level])










level = 2

up_scaling = scalings[level]
new_ns = ns//np.prod(scalings[:level+1])
latent_dim = new_ns**2
bn = False
params_discriminator = dict()
params_discriminator['stride'] = [2, 2, 2 , 1]
params_discriminator['nfilter'] = [16, 128, 128, 64]
params_discriminator['shape'] = [[5, 5],[5, 5], [3, 3], [3, 3]]
params_discriminator['batch_norm'] = [bn, bn, bn, bn]
params_discriminator['full'] = [128]
params_discriminator['summary'] = True
params_discriminator['minibatch_reg'] = True

params_generator = dict()
params_generator['stride'] = [1, 1, 2, 1]
params_generator['latent_dim'] = latent_dim
params_generator['nfilter'] = [64, 128 ,64, 1]
params_generator['shape'] = [[5, 5], [5, 5], [5, 5], [5, 5]]
params_generator['batch_norm'] = [bn, bn, bn]
params_generator['summary'] = True
params_generator['non_lin'] = 'tanh'
params_generator['upsampling'] = up_scaling

params = params0.copy()
params['generator'] = params_generator
params['discriminator'] = params_discriminator
params['optimization'] = params_optimization

params['normalize'] = False
params['image_size'] = [new_ns*up_scaling, new_ns*up_scaling]
params['name'] = 'LAPWGAN{}_level{}_'.format(ns, level)
params['summary_dir'] = global_path + params['name'] + '_' + time_str +'summary/'
params['save_dir'] = global_path + params['name'] + '_' + time_str + 'checkpoints/'


obj = GAN(params, LapGanModel)
obj.train(X=down_sampled_images[level])







level = 1

up_scaling = scalings[level]
new_ns = ns//np.prod(scalings[:level+1])
latent_dim = new_ns**2
bn = False
params_discriminator = dict()
params_discriminator['stride'] = [2, 2, 2 , 1]
params_discriminator['nfilter'] = [16, 128, 128, 64]
params_discriminator['shape'] = [[5, 5],[5, 5], [3, 3], [3, 3]]
params_discriminator['batch_norm'] = [bn, bn, bn, bn]
params_discriminator['full'] = [128]
params_discriminator['summary'] = True
params_discriminator['minibatch_reg'] = True

params_generator = dict()
params_generator['stride'] = [1, 1, 2, 1]
params_generator['latent_dim'] = latent_dim
params_generator['nfilter'] = [64, 128 ,64, 1]
params_generator['shape'] = [[5, 5], [5, 5], [5, 5], [5, 5]]
params_generator['batch_norm'] = [bn, bn, bn]
params_generator['summary'] = True
params_generator['non_lin'] = 'tanh'
params_generator['upsampling'] = up_scaling


params = params0.copy()
params['generator'] = params_generator
params['discriminator'] = params_discriminator
params['optimization'] = params_optimization

params['normalize'] = False
params['image_size'] = [new_ns*up_scaling, new_ns*up_scaling]
params['name'] = 'LAPWGAN{}_level{}_'.format(ns, level)
params['summary_dir'] = global_path + params['name'] + '_' + time_str +'summary/'
params['save_dir'] = global_path + params['name'] + '_' + time_str + 'checkpoints/'

obj = GAN(params, LapGanModel)
obj.train(down_sampled_images[level])







level = 0

up_scaling = scalings[level]
new_ns = ns//np.prod(scalings[:level+1])
latent_dim = new_ns**2
bn = False
params_discriminator = dict()
params_discriminator['stride'] = [2, 2, 2,2 , 1]
params_discriminator['nfilter'] = [16, 128, 256, 128, 64]
params_discriminator['shape'] = [[5, 5],[5, 5], [5, 5], [3, 3], [3, 3]]
params_discriminator['batch_norm'] = [bn, bn, bn, bn, bn]
params_discriminator['full'] = [128]
params_discriminator['summary'] = True
params_discriminator['minibatch_reg'] = True

params_generator = dict()
params_generator['stride'] = [1, 1, 2, 1, 1]
params_generator['latent_dim'] = latent_dim
params_generator['nfilter'] = [64, 256, 128 ,64, 1]
params_generator['shape'] = [[5, 5], [5, 5], [5, 5], [5, 5], [5, 5]]
params_generator['batch_norm'] = [bn, bn, bn, bn]
params_generator['summary'] = True
params_generator['non_lin'] = 'tanh'
params_generator['upsampling'] = up_scaling

params = params0.copy()
params['generator'] = params_generator
params['discriminator'] = params_discriminator
params['optimization'] = params_optimization

params['normalize'] = False
params['image_size'] = [new_ns*up_scaling, new_ns*up_scaling]
params['name'] = 'LAPWGAN{}_level{}_'.format(ns, level)
params['summary_dir'] = global_path + params['name'] + '_' + time_str +'summary/'
params['save_dir'] = global_path + params['name'] + '_' + time_str + 'checkpoints/'


obj = GAN(params, LapGanModel)
obj.train(down_sampled_images[level])





