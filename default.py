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

    # Discriminator parameters
    # ------------------------
    params['discriminator'] = params.get('discriminator', dict())
    params['discriminator']['minibatch_reg'] = params['discriminator'].get(
        'minibatch_reg', False)

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

    # Generator parameters
    # --------------------
    params['generator'] = params.get('generator', dict())
    params['generator']['y_layer'] = params['generator'].get('y_layer', None)

    return params


def default_params_cosmology(params=dict()):

    params = default_params(params)
    # Cosmology parameters
    # --------------------
    params['cosmology'] = params.get('cosmology', dict())
    params['cosmology']['clip_max_real'] = params['cosmology'].get(
        'clip_max_real', False)
    # Clip the generated data to the same maximum as the real data
    params['cosmology']['log_clip'] = params['cosmology'].get('log_clip', 0.1)
    params['cosmology']['sigma_smooth'] = params['cosmology'].get(
        'sigma_smooth', 1)
    # Apply a guausian filter to remove high frequency before executing the computations
    params['cosmology']['k'] = params['cosmology'].get('k', 10)
    params['cosmology']['Npsd'] = params['cosmology'].get('Npsd', 500)

    return params