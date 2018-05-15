import data.fmap as fmap
import numpy as np
import warnings


def arg_helper(params, d_param):
    for key in d_param.keys():
        params[key] = params.get(key, d_param[key])
        if type(params[key]) is dict:
            params[key] = arg_helper(params[key], d_param[key])
    return params


def default_params(params=None):

    # Global parameters
    # -----------------
    d_param = dict()
    d_param['print_every'] = 100
    # Print the losses in the consol every 'print_every' iterations
    d_param['save_every'] = 2000
    # Save the model every 'save_every' iterations
    d_param['sum_every'] = 200
    # Compute the numerical summaries every 'sum_every' iterations
    d_param['viz_every'] = 200
    # Buil the visual summaries every 'viz_every' iterations
    d_param['normalize'] = False
    # Apply a normalization step to the data
    d_param['resume'] = False
    # Resume training. Warning, needs exactly the same parameters
    d_param['prior_distribution'] = 'gaussian'
    # Prior distribution to sample from ('Gaussian','Uniform',...)
    d_param['image_size'] = [32, 32, 1]
    # size of input image

    # Discriminator parameters
    # ------------------------
    d_param['discriminator'] = dict()
    d_param['discriminator']['minibatch_reg'] = False
    # Minibatch regularization
    # d_param['discriminator']['minibatch_reg'] = False
    # print('Minibatch regularization set to False (Force)')
    d_param['discriminator']['non_lin'] =  None
    d_param['discriminator']['one_pixel_mapping'] = []

    # Optimization parameters
    # -----------------------
    d_param['optimization'] = dict()
    d_param['optimization']['disc_optimizer'] = "rmsprop"
    d_param['optimization']['gen_optimizer'] = "rmsprop"
    d_param['optimization']['enc_optimizer'] = "rmsprop"
    d_param['optimization']['gen_learning_rate'] = 3e-5
    d_param['optimization']['enc_learning_rate'] = 3e-5
    d_param['optimization']['disc_learning_rate'] = 3e-5
    d_param['optimization']['n_critic'] = 5

    # Generator parameters
    # --------------------
    d_param['generator'] = dict()
    d_param['generator']['y_layer'] = None
    d_param['generator']['one_pixel_mapping'] = []

    return arg_helper(params or {}, d_param)


def default_params_cosmology(params=None):

    forward = fmap.forward
    backward = fmap.backward

    d_param = default_params()
    # Cosmology parameters
    # --------------------
    d_param['cosmology'] = dict()
    d_param['cosmology']['Nstats'] = 500
    d_param['cosmology']['forward_map'] = forward
    d_param['cosmology']['backward_map'] = backward
    # Default transformation for the data

    if 'Npsd' in params['cosmology'].keys():
        params['cosmology']['Nstats'] = params['cosmology']['Npsd']
        warnings.warn('Use Nstats instead of Npsd!')

    return arg_helper(params or {}, d_param)


def default_params_time(params=dict()):
    
    d_param = default_params()
    d_param['time'] = dict()

    d_param['time']['num_classes'] = 1
    # Number of classes to condition on
    d_param['time']['classes'] = None
    # Which classes to utilize
    default_scaling = (np.arange(params['time']['num_classes']) + 1) / params['time']['num_classes']
    d_param['time']['class_weights'] = default_scaling
    # Default temporal weights for classes.
    return arg_helper(params or {}, d_param)
