import h5py
import numpy as np
import os
import shutil
import tensorflow as tf
import tensorflow.contrib.slim as slim
import pickle

import scipy


def test_resume(try_resume, params):
    ''' Try to load the parameters saved in `params['save_dir']+'params.pkl',` 

        Not sure we should implement this function that way.
    '''

    resume = False

    if try_resume:
        try:
            with open(params['save_dir']+'params.pkl', 'rb') as f:
                params = pickle.load(f)
            resume = True
            print('Resume, the training will start from the last iteration!')
        except:
            print('No resume, the training will start from the beginning!')

    return resume, params

def sample_latent(m, n, prior="uniform", normalize=False):
    if prior == "uniform":
        return np.random.uniform(-1., 1., size=[m, n])
    elif prior == "gaussian":
        z = np.random.normal(0, 1, size=[m, n])
        if normalize:
            # Sample on the sphere
            z = (z.T * np.sqrt(n / np.sum(z * z, axis=1))).T
        return z
    elif prior.startswith('student'):
        prior_ = prior.split('-')
        if len(prior_) == 2:
            df = int(prior_[1])
        else:
            df = 3
        return np.random.standard_t(df, size=[m, n])
    elif prior.startswith('chi2'):
        prior_ = prior.split('-')
        if len(prior_) >= 2:
            df = int(prior_[1])
            if len(prior_) >= 3:
                k = float(prior_[2])
            else:
                k = 7
        else:
            df, k = 3, 7
        return simple_numpy(np.random.chisquare(df, size=[m, n]), k)
    elif prior.startswith('gamma'):
        prior_ = prior.split('-')
        if len(prior_) >= 2:
            df = float(prior_[1])
            if len(prior_) >= 3:
                k = float(prior_[2])
            else:
                k = 4
        else:
            df, k = 1, 4
        return simple_numpy(np.random.gamma(df, size=[m, n]), k)
    else:
        raise ValueError(' [!] distribution not defined')


def forward_map(x, k=10., scale=1.):
    ''' maps real positive numbers to a [-scale, scale] range 

    Numpy version
    '''

    return scale * (2 * (x / (x + k)) - 1)


def backward_map(y, k=10., scale=1., real_max=1e8):
    ''' Inverse of the function forward map
        
    Numpy version
    '''

    simple_max = forward_map(real_max, k, scale)
    simple_min = forward_map(0, k, scale)

    y_clipped = np.clip(y, simple_min, simple_max)/scale
    return k * (y_clipped + 1) / (1 - y_clipped)


def pre_process(X_raw, k=10., scale=1.):
    ''' maps real positive numbers to a [-scale, scale] range 

    Tensorflow version
    '''

    k = tf.constant(k, dtype=tf.float32)
    X = tf.subtract(2.0 * (X_raw / tf.add(X_raw, k)), 1.0)*scale

    return X


def inv_pre_process(X, k=10., scale=1., real_max=1e8):
    ''' Inverse of the function forward map
        
    Tensorflow version
    '''

    simple_max = forward_map(real_max, k, scale)
    simple_min = forward_map(0, k, scale)

    X_clipped = tf.clip_by_value(X, simple_min, simple_max)/scale
    X_raw = tf.multiply((X_clipped + 1.0) / (1.0 - X_clipped), k)
    return X_raw


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def saferm(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
        print('Erase recursively directory: ' + path)
    if os.path.isfile(path):
        os.remove(path)
        print('Erase file: ' + path)


def makeit_square(x):
    sh = list(x.shape)
    nsamples, sx, sy = sh[0], sh[1], sh[2]
    if sx > sy:
        #         if mod(sx,sy):
        #             ValueError('Wrong size')
        cutx = sx // sy
        new_sh = sh
        new_sh[0] = nsamples * cutx
        new_sh[1] = sy
        new_x = np.zeros(new_sh)
        for i in range(cutx):
            new_x[i * nsamples:(i + 1) * nsamples, :, :] = x[:, i * sy:(
                i + 1) * sy, :]

    elif sy > sx:
        #         if mod(sy,sx):
        #             ValueError('Wrong size')
        cuty = sy // sx
        new_sh = sh
        new_sh[0] = nsamples * cuty
        new_sh[2] = sx
        new_x = np.zeros(new_sh)
        for i in range(cuty):
            new_x[i * nsamples:(i + 1) * nsamples, :, :] = x[:, :, i * sx:(
                i + 1) * sx]
    else:
        new_x = x
    return new_x

def get_tile_shape_from_3d_image(image_size):
    '''
    given a 3d image, tile it as a rectangle with slices of the 3d image,
    and return the shape of the rectangle
    '''
    x_dim, y_dim, z_dim = image_size
    num_images_in_each_row = num_images_each_row(x_dim)
    tile_shape = ( y_dim * (x_dim//num_images_in_each_row), z_dim * num_images_in_each_row)
    return tile_shape

def num_images_each_row(x_dim):
    num_images_in_each_row = int(x_dim**0.5)
    while x_dim % num_images_in_each_row != 0:#smallest number that is larger than square root of x_dim and divides x_dim
        num_images_in_each_row += 1    

    return num_images_in_each_row

def tile_cube_slices(cube):
    '''
    cube = [:, :, :]
    arrange cube as tile of squares
    '''
    x_dim = cube.shape[0]
    y_dim = cube.shape[1]
    z_dim = cube.shape[2]
    v_stacks = []
    num = 0
    num_images_in_each_row = num_images_each_row(x_dim)

    for i in range(x_dim//num_images_in_each_row):
        h_stacks = []
        for j in range(num_images_in_each_row): # show 'num_images_in_each_row' squares from the cube in one row
            h_stacks.append(cube[num, :, :])
            num += 1
        v_stacks.append( np.hstack(h_stacks) )

    tile = np.vstack(v_stacks)
    return tile.reshape([1, *(tile.shape), 1])

def get_3d_hists_dir_paths(path_3d_hists):
    dir_paths = []
    for item in os.listdir(path_3d_hists):
        dir_path = os.path.join(path_3d_hists, item)
        if os.path.isdir(dir_path) and item.endswith('hist'): # the directories where the 3d histograms are saved end with 'hist'
            dir_paths.append(dir_path)

    return dir_paths


# def make_batches(bs, *args):
#     """
#     Slide data with a size bs

#     Parameters
#     ----------
#     bs : batch size
#     *args : different pieces of data of the same size

#     """

#     ndata = len(args)
#     s0 = len(args[0])
#     for d in args:
#         if len(d) != s0:
#             raise ValueError("First dimensions differ!")

#     return itertools.zip_longest(
#         *(grouper(itertools.cycle(arg), bs) for arg in args))

# def grouper(iterable, n, fillvalue=None):
#     """
#     Collect data into fixed-length chunks or blocks. This function commes
#     from itertools
#     """
#     # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
#     args = [iter(iterable)] * n
#     return itertools.zip_longest(fillvalue=fillvalue, *args)


def save_hdf5(data, filename, dataset_name='data', mode='w'):
    h5f = h5py.File(filename, mode)
    h5f.create_dataset(dataset_name, data=data, dtype='float32')
    h5f.close()


def load_hdf5(filename, dataset_name='data', mode='r'):
    h5f = h5py.File(filename, mode)
    data = h5f[dataset_name][:]
    h5f.close()
    return data

def load_hdf5_all_datasets(filename, num=100):
    lst = []
    h5f = h5py.File(filename, 'r')
    for i in range(num):
        data = h5f['data' + str(i)][:]
        lst.append(data)

    h5f.close()
    return lst


def compose2(first,second):
    ''' Return the composed function `second(first(arg))` '''
    return lambda x: second(first(x))

def compose(*functions):
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)
