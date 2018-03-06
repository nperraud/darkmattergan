import sys
sys.path.insert(0, '../')

import dict_reader, utils,  sys
from Data_Generators import time_toy_generator
from model import WGanModel, WNGanModel, TemporalGanModelv3
from gan import CosmoGAN
import numpy as np


def current_time_str():
    import time, datetime
    d = datetime.datetime.fromtimestamp(time.time())
    return str(d.year) + '_' + str(d.month)+ '_' + str(d.day) + '_' + str(d.hour) + '_' + str(d.minute)


def main():
    # Load parameters
    param_paths = sys.argv[1]
    params = dict_reader.read_gan_dict(param_paths)
    params['name'] = 'WGAN{}'.format(params['image_size'][0])
    time_str = current_time_str()
    if 'save_dir' in params:
        params['summary_dir'] = params['summary_dir'] + '/' + params['name'] + '_' + time_str + '_summary/'
        params['save_dir'] = params['save_dir'] + '/' + params['name'] + '_' + time_str + '_checkpoints/'
    else:
        params['summary_dir'] = 'tboard/' + params['name'] + '_' + time_str + 'summary/'
        params['save_dir'] = 'checkp/' + params['name'] + '_' + time_str + 'checkpoints/'

    print("All params")
    print(params)
    print("\nDiscriminator Params")
    print(params['discriminator'])
    print("\nGenerator Params")
    print(params['generator'])
    print("\nOptimization Params")
    print(params['optimization'])
    print("\nCosmo Params")
    print(params['cosmology'])
    print()

    # Initialize model
    if params['model_idx'] == 0:
        model = WGanModel
    if params['model_idx'] == 1:
        model = WNGanModel
    if params['model_idx'] == 2:
        model = TemporalGanModelv3
    cosmo_gan = CosmoGAN(params, model)

    num_gaussians = 42
    if 'num_gaussians' in params:
        num_gaussians = params['num_gaussians']

    # Generate data
    data = time_toy_generator.gen_dataset(images_per_time_step=params['num_samples_per_class'],
                                          width=params['image_size'][0],
                                          num_gaussians=num_gaussians,
                                          point_density_factor=3)
    if params['num_classes'] == 5:
        data = np.asarray([data[1], data[3], data[5], data[7], data[9]])
    if params['num_classes'] == 4:
        data = np.asarray([data[0], data[3], data[6], data[9]])
    if params['num_classes'] == 2:
        data = np.asarray([data[5], data[9]])
    if params['num_classes'] == 1:
        data = np.asarray([data[9]])

    # Prep data
    data = data.swapaxes(0,1)
    data = data.reshape((data.shape[0] * data.shape[1], data.shape[2], data.shape[3]))
    data = data.astype(np.float32)
    data = utils.forward_map(data, params['k'])

    # Train model
    cosmo_gan.train(data)
    return 0


main()
