import numpy as np
import os, random
import utils
import functools
from data import gaussian_synthetic_data
from data import path
from data import transformation
from data.Dataset import Dataset_2d, Dataset_3d

import blocks


def load_data_from_dir(dir_path, k=10):
    '''
    load training data, saved as ndarrays in files in directory dir_path
    '''
    raw_data = []
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        arr = utils.load_hdf5(
            filename=file_path, dataset_name='data', mode='r')
        raw_data.append(arr)

    raw_data = np.array(raw_data)
    forward_mapped_data = utils.forward_map(raw_data, k)

    return forward_mapped_data, raw_data


def load_data_from_file(file_path, k=10):
    '''
    load training data, saved as ndarrays in file
    '''
    raw_data = []
    raw_data = utils.load_hdf5(
        filename=file_path, dataset_name='data', mode='r')
    if type(raw_data) is not np.ndarray:
        raise ValueError(
            "Data stroed in file {} is not of type np.ndarray".format(
                file_path))

    forward_mapped_data = utils.forward_map(raw_data, k)

    return forward_mapped_data, raw_data


def load_3d_synthetic_samples(nsamples, dim, k):
    images = 2 * gaussian_synthetic_data.generate_cubes( # Forward mapped
        nsamples=nsamples, cube_dim=dim) - 1.0
    raw_images = utils.backward_map(images)

    return Dataset_3d(images, spix=dim, shuffle=False, transform=None), raw_images


def load_2d_synthetic_samples(nsamples, dim, k):
    images = 2 * gaussian_synthetic_data.generate_squares( # Forward mapped
        nsamples=nsamples, square_dim=dim) - 1.0
    raw_images = utils.backward_map(images)

    return Dataset_2d(images, spix=dim, shuffle=False, transform=None)


def load_samples(nsamples=1000, shuffle=False, k=10, spix=256, map_scale=1., transform=None):
    ''' This function will be removed in the near futur'''
    pathfolder = path.data_path(spix)
    input_pattern = 'Box_70*snapshot_050'
    file_ext = '.dat'

    queue = []
    for file in os.listdir(pathfolder):
        if file.endswith(file_ext) and (np.all(
            [x in file for x in input_pattern.split("*")])):
            queue.append(os.path.join(pathfolder, file))
    if nsamples > len(queue):
        print('They are {} "{}" files.'.format(len(queue), file_ext))
        raise ValueError("The number of samples must be smaller "
                         "or equal to the number of files")
    else:
        print('Select {} samples out of {}.'.format(nsamples, len(queue)))
    raw_images = np.array(
        list(
            map(lambda i: np.fromfile(queue[i], dtype=np.float32),
                range(nsamples))))
    raw_images.resize([nsamples, spix, spix])

    if shuffle:
        p = np.random.permutation(nsamples)
        raw_images = raw_images[p]

    images = utils.forward_map(raw_images, k=k, scale=map_scale)

    dataset = Dataset(images, shuffle=False, transform=transform)
    return dataset


def load_samples_raw(nsamples=None, resolution=256, Mpch=70):
    ''' Load 2D or 3D raw images

    Arguments
    ---------
    * nsamples : desired number of samples (if None: all of them)
    * resolution : [256, 512]
    * Mpch : [70, 350]

    '''
    rootpath = path.root_path()
    input_pattern = '{}_nbody_{}Mpc'.format(resolution, Mpch)
    file_ext = '.h5'
    queue = []
    for file in os.listdir(rootpath):
        if file.endswith(file_ext) and input_pattern in file:
            queue.append(os.path.join(rootpath, file))

    if len(queue) == 0:
        ValueError('No file founds, check path and parameters')
    raw_images = []
    for file_path in queue:
        raw_images.append(
            utils.load_hdf5(
                filename=file_path, dataset_name='data', mode='r'))
        if type(raw_images[-1]) is not np.ndarray:
            raise ValueError(
                "Data stroed in file {} is not of type np.ndarray".format(
                    file_path))

    raw_images = np.array(raw_images).astype(np.float32)


    if nsamples is None:
        return raw_images
    else:
        if nsamples > len(raw_images):
            raise ValueError("Not enough sample")
        else:
            print('Select {} samples out of {}.'.format(
                nsamples, len(raw_images)))

        return raw_images[:nsamples]


def load_dataset(
        nsamples=None,
        resolution=256,
        Mpch=70,
        shuffle=True,
        forward_map = None,
        spix=128,
        augmentation=True,
        scaling=1,
        is_3d=False):
    ''' Load a 2D dataset object 

     Arguments
    ---------
    * nsamples : desired number of samples, if None => all of them (default None)
    * resolution : [256, 512] (default 256)
    * Mpch : [70, 350] (default 70)
    * shuffle: shuffle the data (default True)
    * foward : foward mapping use None for raw data (default None)
    * spix : resolution of the image (default 128)
    * augmentation : use data augmentation (default True)
    * scaling : downscale the image by a factor (default 1)
    * is_3d : load a 3d dataset (default False)
    '''

    # 1) Load raw images
    images = load_samples_raw(nsamples=nsamples, resolution=resolution, Mpch=Mpch)

    # 2) Apply forward map if necessary
    if forward_map:
        images = forward_map(images)

    # 2p) Apply downscaling if necessary
    if scaling>1:
        images = blocks.downsample(images, scaling)

    if augmentation:
        t = transformation.random_transformation_3d
    else:
        t = None
    
    # 5) Make a dataset
    if is_3d:
        dataset = Dataset_3d(images, spix=spix, shuffle=shuffle, transform=t)
    else:
        dataset = Dataset_2d(images, spix=spix, shuffle=shuffle, transform=t)

    return dataset



    

