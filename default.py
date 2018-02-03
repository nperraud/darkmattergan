
def default_params(params=dict()):

    params['normalize'] = params.get('normalize', False)

    params['discriminator'] = params.get('discriminator', dict())
    params['discriminator']['minibatch_reg'] = params['discriminator'].get('minibatch_reg', False)

    params['optimization'] = params.get('optimization', dict())
    params['optimization']['disc_optimizer'] = params['optimization'].get('disc_optimizer', "adam")
    params['optimization']['gen_optimizer'] = params['optimization'].get('gen_optimizer', "adam")

    return params
