import numpy as np
import os
import utils
import gaussian_synthetic_data
import socket
import pickle

def load_3d_hists(path_3d_hists, k):
    raw_hists_3d = []
    for item in os.listdir(path_3d_hists):
        dir_path = os.path.join(path_3d_hists, item)
        if os.path.isdir(dir_path) and item.endswith('hist'): # the directories where the 3d histograms are saved end with 'hist'
            print("----------------------------------current directory {}".format(item))
            for file_name in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file_name)
                with open(file_path, 'rb') as pickle_file:
                    arr = pickle.load(pickle_file)
                    raw_hists_3d.append(arr)
                    
    raw_hists_3d = np.array(raw_hists_3d)
    hists_3d = utils.forward_map(raw_hists_3d, k)

    return hists_3d, raw_hists_3d


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
