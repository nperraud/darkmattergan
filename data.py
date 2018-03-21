import numpy as np
import os, random
import utils
from data import gaussian_synthetic_data


def load_data_from_dir(dir_path, k=10):
    '''
    load training data, saved as ndarrays in files in directory dir_path
    '''
    raw_data = []
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        arr = utils.load_hdf5(filename=file_path, dataset_name='data', mode='r')
        raw_data.append(arr)
                    
    raw_data = np.array(raw_data)
    forward_mapped_data = utils.forward_map(raw_data, k)

    return forward_mapped_data, raw_data

def load_data_from_file(file_path, k=10):
    '''
    load training data, saved as ndarrays in file
    '''
    raw_data = []
    raw_data = utils.load_hdf5(filename=file_path, dataset_name='data', mode='r')
    if type(raw_data) is not np.ndarray:
        raise ValueError("Data stroed in file {} is not of type np.ndarray".format(file_path))

    forward_mapped_data = utils.forward_map(raw_data, k)

    return forward_mapped_data, raw_data


def load_3d_synthetic_samples(nsamples, dim, k):
    images = 2 * gaussian_synthetic_data.generate_cubes(
        nsamples=nsamples, cube_dim=dim) - 1.0
    raw_images = utils.backward_map(images)

    return images, raw_images


def load_2d_synthetic_samples(nsamples, dim, k):
    images = 2 * gaussian_synthetic_data.generate_squares(
        nsamples=nsamples, square_dim=dim) - 1.0
    raw_images = utils.backward_map(images)

    return images, raw_images


def load_samples(nsamples=1000, permute=False, k=10, spix=256):
    path = data_path(spix)
    input_pattern = 'Box_70*snapshot_050'
    file_ext = '.dat'

    queue = []
    for file in os.listdir(path):
        if file.endswith(file_ext) and (np.all(
            [x in file for x in input_pattern.split("*")])):
            queue.append(os.path.join(path, file))
    if nsamples > len(queue):
        print('They are {} "{}" files.'.format(len(queue), file_ext))
        raise ValueError(
            "The number of samples must be smaller "
            "or equal to the number of files"
        )
    else:
        print('Select {} samples out of {}.'.format(nsamples, len(queue)))
    raw_images = np.array(
        list(
            map(lambda i: np.fromfile(queue[i], dtype=np.float32),
                range(nsamples))))
    raw_images.resize([nsamples, spix, spix])

    if permute:
        p = np.random.permutation(nsamples)
        raw_images = raw_images[p]

    images = utils.forward_map(raw_images, k)

    return images, raw_images

