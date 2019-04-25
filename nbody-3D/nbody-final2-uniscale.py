import tensorflow as tf
import os
from gantools import data, utils
from gantools.model import UpscalePatchWGAN, CosmoWGAN
from gantools.gansystem import GANsystem, UpscaleGANsystem

from functools import partial
forward = partial(data.fmap.stat_forward, c=20000, shift=1)
backward = partial(data.fmap.stat_backward, c=20000, shift=1)

ns = 32
try_resume = True

time_str = 'final2-uniscale'
global_path = '../saved_results/nbody/'
name = 'WGAN_' + time_str

bn = False

md=32

params_discriminator = dict()
params_discriminator['stride'] = [1, 2, 2, 1, 2]
params_discriminator['nfilter'] = [md, 2*md, 4*md, md, md]
params_discriminator['shape'] = [[4, 4, 4],[4, 4, 4], [4, 4, 4],[4, 4, 4], [4, 4, 4]]
params_discriminator['batch_norm'] = [bn, bn, bn, bn, bn ]
params_discriminator['full'] = [256]
params_discriminator['minibatch_reg'] = False
params_discriminator['summary'] = True
params_discriminator['data_size'] = 3
params_discriminator['inception'] = True
params_discriminator['spectral_norm'] = False
# params_discriminator['histogram'] = dict()
# params_discriminator['histogram']['bins'] = 100
# params_discriminator['histogram']['data_size'] = 3
# params_discriminator['histogram']['full'] = 128
# params_discriminator['histogram']['spectral_norm'] = True

params_generator = dict()
params_generator['stride'] = [1, 1, 2, 2, 1]
params_generator['latent_dim'] = 1024 + 8*8*8*8
params_generator['latent_dim_split'] = [8,8,8,8]
params_generator['in_conv_shape'] =[8, 8, 8]
params_generator['nfilter'] = [md, md, 4*md, 2*md, 1]
params_generator['shape'] = [[4, 4, 4],[4, 4, 4], [4, 4, 4],[4, 4, 4], [4, 4, 4]]
params_generator['batch_norm'] = [bn, bn, bn, bn]
params_generator['full'] = [256*md]
params_generator['summary'] = True
params_generator['non_lin'] = None
params_generator['data_size'] = 3
params_generator['inception'] = True
params_generator['spectral_norm'] = False
params_generator['borders'] = dict()
params_generator['borders']['stride'] = [2, 2, 2]
params_generator['borders']['nfilter'] = [md, md, 16]
params_generator['borders']['shape'] = [[4, 4, 4],[4, 4, 4], [4, 4, 4]]
params_generator['borders']['batch_norm'] = [bn, bn, bn]
params_generator['borders']['data_size'] = 3
params_generator['borders']['width_full'] = None

# Optimization parameters inspired from 'Self-Attention Generative Adversarial Networks'
# - Spectral normalization GEN DISC
# - Batch norm GEN
# - TTUR ('GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium')
# - ADAM  beta1=0 beta2=0.9, disc lr 0.0004, gen lr 0.0001
# - Hinge loss
# Parameters are similar to the ones in those papers...
# - 'PROGRESSIVE GROWING OF GANS FOR IMPROVED QUALITY, STABILITY, AND VARIATION'
# - 'LARGE SCALE GAN TRAINING FOR HIGH FIDELITY NATURAL IMAGE SYNTHESIS'
# - 'CGANS WITH PROJECTION DISCRIMINATOR'

params_optimization = dict()
params_optimization['batch_size'] = 8
params_optimization['epoch'] = 100
params_optimization['n_critic'] = 5
# params_optimization['generator'] = dict()
# params_optimization['generator']['optimizer'] = 'adam'
# params_optimization['generator']['kwargs'] = {'beta1':0, 'beta2':0.9}
# params_optimization['generator']['learning_rate'] = 0.0004
# params_optimization['discriminator'] = dict()
# params_optimization['discriminator']['optimizer'] = 'adam'
# params_optimization['discriminator']['kwargs'] = {'beta1':0, 'beta2':0.9}
# params_optimization['discriminator']['learning_rate'] = 0.0001

# Cosmology parameters
params_cosmology = dict()
params_cosmology['forward_map'] = forward
params_cosmology['backward_map'] = backward


# all parameters
params = dict()
params['net'] = dict() # All the parameters for the model
params['net']['generator'] = params_generator
params['net']['discriminator'] = params_discriminator
params['net']['cosmology'] = params_cosmology # Parameters for the cosmological summaries
params['net']['prior_distribution'] = 'gaussian'
params['net']['shape'] = [ns, ns, ns, 8] # Shape of the image
params['net']['loss_type'] = 'wasserstein' # loss ('hinge' or 'wasserstein')
params['net']['gamma_gp'] = 10 # Gradient penalty
params['net']['upsampling'] = None 

params['optimization'] = params_optimization
params['summary_every'] = 100 # Tensorboard summaries every ** iterations
params['print_every'] = 50 # Console summaries every ** iterations
params['save_every'] = 1000 # Save the model every ** iterations
params['summary_dir'] = os.path.join(global_path, name +'_summary/')
params['save_dir'] = os.path.join(global_path, name + '_checkpoints/')
params['Nstats'] = 30
params['Nstats_cubes'] = 5


resume, params = utils.test_resume(try_resume, params)

class CosmoUpscalePatchWGAN(UpscalePatchWGAN, CosmoWGAN):
    pass


wgan = GANsystem(CosmoUpscalePatchWGAN, params)

dataset = data.load.load_nbody_dataset(
    spix=ns,
    scaling=1,
    resolution=256,
    Mpch=350,
    patch=True,
    augmentation=True,
    forward_map=forward,
    is_3d=True)

wgan.train(dataset, resume=resume)
