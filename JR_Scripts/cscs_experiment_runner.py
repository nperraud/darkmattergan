import sys
sys.path.insert(0, '../')

import utils, sys, os
from JR_Scripts import dict_reader, time_toy_generator
from model import WGanModel, WNGanModel, TemporalGanModelv3
from gan import CosmoGAN
import numpy as np
import pickle
from data import Dataset, fmap, path


def current_time_str():
    import time, datetime
    d = datetime.datetime.fromtimestamp(time.time())
    return str(d.year) + '_' + str(d.month)+ '_' + str(d.day) + '_' + str(d.hour) + '_' + str(d.minute)


def block_reduce(data):
    d = np.zeros((data.shape[0], data.shape[1]//2, data.shape[2]//2))
    d = d + data[:,::2,::2]
    d = d + data[:,::2,1::2]
    d = d + data[:,1::2,::2]
    d = d + data[:,1::2,1::2]
    return d


def main():
    # Load parameters
    param_paths = sys.argv[1]
    params = dict_reader.read_gan_dict(param_paths)
    if 'name' not in params:
        params['name'] = 'WGAN{}'.format(params['image_size'][0])
    time_str = current_time_str()
    if 'save_dir' in params:
        params['summary_dir'] = params['summary_dir'] + '/' + params['name'] + '_' + time_str + '_summary/'
        params['save_dir'] = params['save_dir'] + '/' + params['name'] + '_' + time_str + '_checkpoints/'
    else:
        params['summary_dir'] = 'tboard/' + params['name'] + '_' + time_str + 'summary/'
        params['save_dir'] = 'checkp/' + params['name'] + '_' + time_str + 'checkpoints/'

    #params['generator']['non_lin'] = None

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

    if not os.path.exists(params['summary_dir']):
        os.makedirs(params['summary_dir'])
    utils.save_dict_pickle(params['summary_dir'] + 'params.pkl', params)
    utils.save_dict_for_humans(params['summary_dir'] + 'params.txt', params)
    if not os.path.exists(params['save_dir']):
        os.makedirs(params['save_dir'])
    utils.save_dict_pickle(params['save_dir'] + 'params.pkl', params)
    utils.save_dict_for_humans(params['save_dir'] + 'params.txt', params)

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
    data = np.zeros((10, 3000, 128, 128))
    for i in range(10):
        x = utils.load_hdf5(path.root_path() + 'Mpc100_10_redshifts.h5', dataset_name=str(i))
        print(x.shape)
        data[9-i] = block_reduce(x)

    if params['num_classes'] == 8:
        data = np.asarray([data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9]])
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
    data = fmap.forward_map(data, params['cosmology']['k'], 0.98)
#    data = fmap.forward(data)

    data = Dataset.Dataset(data, shuffle=False)

    # Train model
    cosmo_gan.train(data)
    return 0


main()
