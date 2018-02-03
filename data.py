import numpy as np
import os
import utils


def load_samples(nsamples=1000, permute=False, k=10, spix=256):
    path = '../data/size{}_splits1000_n500x3/'.format(spix)
    input_pattern = 'Box_70*snapshot_050'
    file_ext = '.dat'

    queue = []
    for file in os.listdir(path):
        if file.endswith(file_ext) and (np.all([x in file for x in input_pattern.split("*")])):
            queue.append(os.path.join(path, file))
    if nsamples > len(queue):
        print('They are {} "{}" files.'.format(len(queue), file_ext))
        raise ValueError("The number of samples must be smaller or equal to the number of files")
    else:
        print('Select {} samples out of {}.'.format(nsamples, len(queue)))

    raw_images = np.vstack(map(lambda i: np.fromfile(queue[i], dtype=np.float32), range(nsamples)))
    raw_images.resize([nsamples, spix, spix])

    images = utils.forward_map(raw_images, k)
    if permute:
        p = np.random.permutation(nsamples)
        images = images[p]

    return images, raw_images


def make_smaller_samples(images, nx, ny=None):
    if ny is None:
        ny = nx
    nsamples = images.shape[0]
    spixx = images.shape[1]
    spixy = images.shape[1]
    cutx = spixx//nx
    cuty = spixy//ny
    img_small = np.zeros([nsamples*cutx*cuty, nx, ny])
    for i in range(cutx):
        for j in range(cuty):
            l = j + i*cuty
            img_small[l*nsamples:(l+1)*nsamples, :, :] = images[:, i*nx:(i+1)*nx, j*ny:(j+1)*ny]

    return img_small


def down_sample_images(images, scalings):
    from blocks import downsample
    down_sampled_images = []
    down_sampled_images.append(images)

    for scale in scalings:
        down_sampled_images.append(downsample(down_sampled_images[-1], scale))

    return down_sampled_images
