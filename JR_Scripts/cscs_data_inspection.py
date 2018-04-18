import sys
sys.path.insert(0, '../')

import utils, sys, os
from JR_Scripts import dict_reader, time_toy_generator
import numpy as np
from data import path


def current_time_str():
    import time, datetime
    d = datetime.datetime.fromtimestamp(time.time())
    return str(d.year) + '_' + str(d.month)+ '_' + str(d.day) + '_' + str(d.hour) + '_' + str(d.minute)


def main():
    # Load parameters
    param_paths = sys.argv[1]
    params = dict_reader.read_gan_dict(param_paths)

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

    # Generate data
    data = np.zeros((10, 100, 256, 256))
    for i in range(10):
        x = utils.load_hdf5(path.root_path() + 'Mpc500_10_redshifts.h5', dataset_name=str(i))
        print(x.shape)
        data[i] = x[:100]

    utils.save_hdf5(data, "/scratch/snx3000/rosenthj/data/data.h5", dataset_name="data")

    # Train model
    return 0


main()
