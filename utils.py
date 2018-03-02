import h5py
import numpy as np
import os
import shutil
import tensorflow as tf
import tensorflow.contrib.slim as slim
import pickle

import scipy


def test_resume(try_resume, params):
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


def forward_map(x, k=10.):
    return 2 * (x / (x + k)) - 1


def backward_map(y, k=10.):
    simple_max = forward_map(1e8, k)
    y_clipped = np.clip(y, -1.0, simple_max)
    return k * (y_clipped + 1) / (1 - y_clipped)


def pre_process(X_raw, k=10.):
    k = tf.constant(k, dtype=tf.float32)

    # maps real positive numbers to a [-1,1] range  2 * (x/(x+10)) - 1
    X = tf.subtract(2.0 * (X_raw / tf.add(X_raw, k)), 1.0)

    return X


def inv_pre_process(X, k=10.):
    simple_max = forward_map(
        1e8, k)  # clipping the values to a max of 1e10 particles
    X_clipped = tf.clip_by_value(X, -1.0, simple_max)
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


def tile_cube_slices(cube, epoch, batch, label, save_images=False):
        '''
        cube = [:, :, :]
        arrange cube as tile of squares
        '''
        x_dim = cube.shape[0]
        y_dim = cube.shape[1]
        z_dim = cube.shape[2]
        v_stacks = []
        num = 0
        for i in range(x_dim//8):
            h_stacks = []
            for j in range(8): # show 8 squares from the cube in one row
                h_stacks.append(cube[num, :, :])
                num += 1
            v_stacks.append( np.hstack(h_stacks) )

        tile = np.vstack(v_stacks)

        if save_images:
            dir_path = '../saved_result/Images/' + label
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            file_name = epoch + '_' + batch + '.jpg'
            scipy.misc.imsave(dir_path + '/' + file_name, tile)

        return tile.reshape([1, *(tile.shape), 1])


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


def save_hdf5(data, filename, dataset_name='data'):
    h5f = h5py.File(filename, 'w')
    h5f.create_dataset(dataset_name, data=data)
    h5f.close()


def load_hdf5(filename, dataset_name='data'):
    h5f = h5py.File(filename, 'r')
    data = h5f[dataset_name][:]
    h5f.close()
    return data