"""Utility functions."""
import os
import functools
import shutil
import pickle
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import h5py


def test_resume(try_resume, params):
    """ Try to load the parameters saved in `params['save_dir']+'params.pkl',`

        Not sure we should implement this function that way.
    """

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
    elif prior == "laplacian":
        return np.random.laplace(loc=0.0, scale=1.0, size=[m, n])
    elif prior == "one-sided-laplacian":
        return np.abs(np.random.laplace(loc=0.0, scale=1.0, size=[m, n]))
    elif prior == "2-2tanh2":
        margin = 10*np.finfo(np.float32).eps
        u = np.random.uniform(low=0.0, high=1.0-margin, size=[m, n])
        return np.arctanh(0.5*(u+1.0))
    # elif prior.startswith('chi2'):
    #     prior_ = prior.split('-')
    #     if len(prior_) >= 2:
    #         df = int(prior_[1])
    #         if len(prior_) >= 3:
    #             k = float(prior_[2])
    #         else:
    #             k = 7
    #     else:
    #         df, k = 3, 7
    #     return simple_numpy(np.random.chisquare(df, size=[m, n]), k)
    # elif prior.startswith('gamma'):
    #     prior_ = prior.split('-')
    #     if len(prior_) >= 2:
    #         df = float(prior_[1])
    #         if len(prior_) >= 3:
    #             k = float(prior_[2])
    #         else:
    #             k = 4
    #     else:
    #         df, k = 1, 4
    #     return simple_numpy(np.random.gamma(df, size=[m, n]), k)
    else:
        raise ValueError(' [!] distribution not defined')


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
    l = len(image_size)

    if l == 3:
        x_dim, y_dim, z_dim = image_size
    elif l == 4:
        x_dim, y_dim, z_dim, _ = image_size
    else:
        raise ValueError("image_size too large!!")

    num_images_in_each_row = num_images_each_row(x_dim)
    tile_shape = ( y_dim * (x_dim//num_images_in_each_row), z_dim * num_images_in_each_row)
    return tile_shape


def num_images_each_row(x_dim):
    num_images_in_each_row = int(x_dim**0.5)
    while x_dim % num_images_in_each_row != 0:#smallest number that is larger than square root of x_dim and divides x_dim
        num_images_in_each_row += 1    

    return num_images_in_each_row


def tile_cube_slices(cubes):
    """
    cubes = [:, :, :, :]
    arrange each cube in cubes, as tile of squares
    """
    x_dim = cubes.shape[1]
    y_dim = cubes.shape[2]
    z_dim = cubes.shape[3]
    num_images_in_each_row = num_images_each_row(x_dim)

    tiles = []
    for cube in cubes:
        num = 0
        v_stacks = []
        for i in range(x_dim//num_images_in_each_row):
            h_stacks = []
            for j in range(num_images_in_each_row): # show 'num_images_in_each_row' squares from the cube in one row
                h_stacks.append(cube[num, :, :])
                num += 1
            v_stacks.append( np.hstack(h_stacks) )

        tile = np.vstack(v_stacks)
        tiles.append(tile.reshape([*(tile.shape), 1]))

    return np.array(tiles)


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

def get_latent_dim(out_width, generator_params, is3d=False):
    """Calculate correct size for the latent dimension or fully connected layer
    before the first deconvolutional layer of a generator.

    Input arguments
    ---------------
    * out_width : output width of last deconvolutional layer of the network
    * generator_params  : parameter dict of the generator
    * is_3d  : whether 3d images should be generated by the network.

    Output: Required input layer size
    """
    w = out_width
    for stride in generator_params['stride']:
        w = w // stride
    if is3d:
        return w * w * w * generator_params['nfilter'][0]
    return w * w * generator_params['nfilter'][0]


def save_hdf5(data, filename, dataset_name='data', mode='w', dtype='float32'):
    h5f = h5py.File(filename, mode)
    h5f.create_dataset(dataset_name, data=data, dtype=dtype)
    h5f.close()


def load_hdf5(filename, dataset_name='data', mode='r'):
    h5f = h5py.File(filename, mode)
    data = h5f[dataset_name][:]
    h5f.close()
    return data


def load_dict_pickle(filename):
    with open(filename, 'rb') as infile:
        d = pickle.load(infile)
    return d


def save_dict_pickle(filename, dict_):
    with open(filename, 'wb') as outfile:
        pickle.dump(dict_, outfile)


def save_dict_for_humans(filename, dict_):
    """ Save dict in a pretty text format for humans. Cannot parse this back!
    Use save_dict_pickle for a load friendly version.
    """
    with open(filename, 'w') as outfile:

        outfile.write("All Params")
        outfile.write(str(dict_))

        if 'discriminator' in dict_:
            outfile.write("\nDiscriminator Params")
            outfile.write(str(dict_['discriminator']))

        if 'generator' in dict_:
            outfile.write("\nGenerator Params")
            outfile.write(str(dict_['generator']))

        if 'optimization' in dict_:
            outfile.write("\nOptimization Params")
            outfile.write(str(dict_['optimization']))

        if 'cosmology' in dict_:
            outfile.write("\nCosmology Params")
            outfile.write(str(dict_['cosmology']))

        if 'time' in dict_:
            outfile.write("\nTime Params")
            outfile.write(str(dict_['time']))


def compose2(first,second):
    """ Return the composed function `second(first(arg))` """
    return lambda x: second(first(x))


def compose(*functions):
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)


def print_params_to_py_style_output_helper(name, params):
    print("\n# {} Params".format(name.title()))
    d_name = "params_{}".format(name)
    print("{} = dict()".format(d_name))
    for key, value in params.items():
        print("{}['{}'] = {}".format(d_name, key, value))
    print("params['{}'] = {}".format(name, d_name))


def print_params_to_py_style_output(params):
    print("# General Params")
    print("params = dict()")
    for key, value in params.items():
        if not isinstance(value, dict):
            print("params['{}'] = {}".format(key, value))
    for key, value in params.items():
        if isinstance(value, dict):
            print_params_to_py_style_output_helper(key, value)


def print_sub_dict_params(d_name, params, indent = 0):
    indent_str = " " * indent
    print("\n{}{} params".format(indent_str, d_name).title())
    for key, value in params.items():
        if not isinstance(value, dict):
            print(" {}{}.{}: {}".format(indent_str, d_name, key, value))
    for key, value in params.items():
        if isinstance(value, dict):
            print_sub_dict_params(key, value, indent=indent+1)


def print_param_dict(params):
    print("General Params")
    for key, value in params.items():
        if not isinstance(value, dict):
            print(" {}: {}".format(key, value))
    for key, value in params.items():
        if isinstance(value, dict):
            print_sub_dict_params(key, value)