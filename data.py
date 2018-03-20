import numpy as np
import os, random
import utils
from data import gaussian_synthetic_data
import socket
from ast import literal_eval as make_tuple
import tensorflow as tf
from tensorflow.contrib.data import Dataset
from tensorflow.python.framework import dtypes

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

def read_tfrecords_from_file(file_path, image_size, k=10.):
    '''
    read samples stored in a tfrecord file
    '''
    record_iterator = tf.python_io.tf_record_iterator(path=file_path)
    cubes = []
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)

        img_string = (example.features.feature['image']
                                      .bytes_list
                                      .value[0])

        cube = np.fromstring(img_string, dtype=np.float32)
        cube = cube.reshape(image_size)
        cubes.append(cube)

    data = np.array(cubes, dtype=np.float32)    
    forward_mapped_data = utils.forward_map(data, k)

    return forward_mapped_data, data

def read_tfrecords_from_dir(dir_path, image_size, k):
    '''
    read samples from all tfrecord files in a directory
    '''
    vstacked_forward = 0
    vstacked_raw = 0
    first = True

    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        forward_mapped_data, data = read_tfrecords_from_file(file_path, image_size, k)
        if first:
            vstacked_forward = forward_mapped_data
            vstacked_raw = data
        else:
            vstacked_forward = np.vstack((vstacked_forward, forward_mapped_data))
            vstacked_raw = np.vstack((vstacked_raw, data))

    return vstacked_forward, vstacked_raw


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


def make_smaller_samples(images, nx, ny=None):
    if ny is None:
        ny = nx
    nsamples = images.shape[0]
    spixx = images.shape[1]
    spixy = images.shape[1]
    cutx = spixx // nx
    cuty = spixy // ny
    img_small = np.zeros([nsamples * cutx * cuty, nx, ny])
    for i in range(cutx):
        for j in range(cuty):
            l = j + i * cuty
            img_small[l * nsamples:(
                l + 1) * nsamples, :, :] = images[:, i * nx:(
                    i + 1) * nx, j * ny:(j + 1) * ny]

    return img_small


def down_sample_images(images, scalings):
    from blocks import downsample
    down_sampled_images = []
    down_sampled_images.append(images)

    for scale in scalings:
        down_sampled_images.append(downsample(down_sampled_images[-1], scale))

    return down_sampled_images


def data_path(spix=256):
    # Check if we are on pizdaint
    if 'nid' in socket.gethostname():
        # Mhis to the store folder to be able to all use it?
        # For reading it is ok.
        return '/scratch/snx3000/nperraud/nati-gpu/data/size256_splits1000_n500x3/'
    else:
        # This should be done in a different way
        utils_module_path = os.path.dirname(__file__)
        return utils_module_path + '/../data/size{}_splits1000_n500x3/'.format(spix)


## Create input pipeline for reading in input from files

def create_input_pipeline(dir_paths, batch_size, k=10, shuffle=False, buffer_size=1000):
    sample_file_paths, num_samples, sample_dims = read_sample_file_paths(dir_paths)

    if shuffle:
        random.shuffle(sample_file_paths)

    dataset = tf.data.TFRecordDataset(sample_file_paths)

    def parser(serialized_example):
        """Parses a single tf.Example into image"""
        parsed_features = tf.parse_single_example(
            serialized_example,
            features={
                'image': tf.FixedLenFeature([], tf.string)
            })
        
        image = parsed_features['image']
        image = tf.decode_raw(image, tf.float32)
        image = tf.cast(image, tf.float32)
        image = tf.reshape(image, sample_dims)
        image = utils.forward_map(image, k) # forward map the raw images
        return image

    dataset = dataset.map(parser)
    dataset = dataset.batch(batch_size)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)

    return dataset, num_samples

def read_sample_file_paths(dir_paths):
    '''
    store paths of all sample files in a list
    '''
    num_samples = 0
    sample_dims = []
    sample_file_paths = []
    for dir_path in dir_paths:
        for file_name in os.listdir(dir_path):
            parts = file_name.split('_')
            num_samples += int(parts[1])
            dims = parts[2].split('.')[0]
            sample_dims = make_tuple(dims)

            file_path = os.path.join(dir_path, file_name)
            sample_file_paths.append(file_path)

    return sample_file_paths, num_samples, sample_dims
