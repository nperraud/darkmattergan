from gantools.gansystem import GANsystem, 
from copy import deepcopy

def old2new_params(params_old, model, checky=False):
    obj = GANsystem(model)
    params = deepcopy(obj.params)
    params['Nstats'] = params_old['cosmology']['Nstats']
    params['net']['cosmology']['forward_map'] = params_old['cosmology']['forward_map']
    params['net']['cosmology']['backward_map'] = params_old['cosmology']['backward_map']
    params['net']['discriminator'] = params_old['discriminator']
    if params['net']['discriminator']['is_3d']:
        params['net']['discriminator']['data_size'] = 3
    else:
        params['net']['discriminator']['data_size'] = 2
    del params['net']['discriminator']['is_3d']
    params['net']['generator'] = params_old['generator']
    if params['net']['generator']['is_3d']:
        params['net']['generator']['data_size'] = 3
    else:
        params['net']['generator']['data_size'] = 2
    del params['net']['generator']['is_3d']
    if 'downsampling' in params['net']['generator'].keys():
        if not (params['net']['generator']['downsampling']==1):
            params['net']['upsampling'] = params['net']['generator']['downsampling']
        del params['net']['generator']['downsampling']
    
    if 'y_layer' in params['net']['generator']:
        if checky:
            assert(params['net']['generator']['y_layer']==0)
        del params['net']['generator']['y_layer']

    params['net']['gamma_gp'] = params_old['optimization']['gamma_gp']
    params['net']['prior_distribution'] = params_old['prior_distribution']
    params['net']['shape'] = params_old['image_size']
    params['save_dir'] = params_old['save_dir']
    params['summary_dir'] = params_old['summary_dir']
    params['save_every'] = params_old['save_every']
    params['summary_every'] = params_old['sum_every']
    params['print_every'] = params_old['print_every']
    params['optimization']['batch_size'] = params_old['optimization']['batch_size']
    params['optimization']['epoch'] = params_old['optimization']['epoch']
    return params