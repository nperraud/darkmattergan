import data.fmap as fmap
import numpy as np
import warnings


def default_params(params=dict()):

    # Global parameters
    # -----------------
    params['print_every'] = params.get('print_every', 100)
    # Print the losses in the consol every 'print_every' iterations
    params['save_every'] = params.get('save_every', 100)
    # Save the model every 'save_every' iterations
    params['sum_every'] = params.get('sum_every', 100)
    # Compute the numerical summaries every 'sum_every' iterations
    params['viz_every'] = params.get('viz_every', 100)
    # Buil the visual summaries every 'viz_every' iterations
    params['normalize'] = params.get('normalize', False)
    # Apply a normalization step to the data
    params['resume'] = params.get('resume', False)
    # Resume training. Warning, needs exactly the same parameters
    params['prior_distribution'] = params.get('prior_distribution',
                                              'gaussian')
    # Prior distribution to sample from ('Gaussian','Uniform',...)
    params['image_size'] = params.get('image_size', [32, 32, 1])
    # size of input image


    # Discriminator parameters
    # ------------------------
    params['discriminator'] = params.get('discriminator', dict())
    params['discriminator']['minibatch_reg'] = params['discriminator'].get(
        'minibatch_reg', False)
    # Minibatch regularization
    # params['discriminator']['minibatch_reg'] = False
    # print('Minibatch regularization set to False (Force)')
    params['discriminator']['non_lin'] = params['discriminator'].get(
        'non_lin', None)
    params['discriminator']['one_pixel_mapping'] = params['discriminator'].get('one_pixel_mapping', [])

    # Optimization parameters
    # -----------------------
    params['optimization'] = params.get('optimization', dict())
    params['optimization']['disc_optimizer'] = params['optimization'].get(
        'disc_optimizer', "adam")
    params['optimization']['gen_optimizer'] = params['optimization'].get(
        'gen_optimizer', "adam")
    params['optimization']['gen_learning_rate'] = params['optimization'].get(
        'gen_learning_rate', 3e-5)
    params['optimization']['enc_learning_rate'] = params['optimization'].get(
        'enc_learning_rate', 3e-5)
    params['optimization']['disc_learning_rate'] = params['optimization'].get(
        'disc_learning_rate', 3e-5)
    params['optimization']['n_critic'] = params['optimization'].get(
        'n_critic', 5)

    # Generator parameters
    # --------------------
    params['generator'] = params.get('generator', dict())
    params['generator']['y_layer'] = params['generator'].get('y_layer', None)
    params['generator']['one_pixel_mapping'] = params['generator'].get('one_pixel_mapping', [])

    return params


def default_params_cosmology(params=dict()):

    forward = fmap.forward
    backward = fmap.backward 

    params = default_params(params)
    # Cosmology parameters
    # --------------------
    params['cosmology'] = params.get('cosmology', dict())
    params['cosmology']['clip_max_real'] = params['cosmology'].get(
        'clip_max_real', True)
    # Clip the generated data to the same maximum as the real data
    params['cosmology']['clip_max_real'] = True
    # This is needed for now as othersie the code may bug

    params['cosmology']['log_clip'] = params['cosmology'].get('log_clip', 0.1)
    params['cosmology']['sigma_smooth'] = params['cosmology'].get(
        'sigma_smooth', 1)
    # Apply a guausian filter to remove high frequency before executing the
    # computations. This is not working right now

    if 'Npsd' in params['cosmology'].keys():
        params['cosmology']['Nstats'] = params['cosmology']['Npsd']
        warnings.warn('Use Nstats instead of Npsd!')
    params['cosmology']['Nstats'] = params['cosmology'].get('Nstats', 500)

    params['cosmology']['forward_map'] = params['cosmology'].get('forward_map', forward)
    params['cosmology']['backward_map'] = params['cosmology'].get('backward_map', backward)
    # Default transformation for the data

    return params


def default_params_time(params=dict()):
    params['time']['num_classes'] = params['time'].get('num_classes', 1)
    # Number of classes to condition on
    params['time']['classes'] = params['time'].get('classes', None)
    # Which classes to utilize
    default_scaling = (np.arange(params['time']['num_classes']) + 1) / params['time']['num_classes']
    params['time']['class_weights'] = params['time'].get('class_weights', default_scaling)
    # Default temporal weights for classes.
    return params
