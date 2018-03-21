import numpy as np
from blocks import downsample


def random_shift_2d(images):
    nx = images.shape[1]
    ny = images.shape[2]
    shiftx = np.random.randint(0, nx)
    shifty = np.random.randint(0, ny)
    out = np.roll(images, shift=shiftx, axis=1)
    out = np.roll(out, shift=shifty, axis=2)
    return out

def random_flip_2d(images):
    out = images
    if np.random.rand(1) > 0.5: 
        out = out[:,::-1,:]
    if np.random.rand(1) > 0.5: 
        out = out[:,:,::-1]
    return out

def random_transformation_2d(images):
    return random_flip_2d(random_shift_2d(images))



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
    down_sampled_images = []
    down_sampled_images.append(images)

    for scale in scalings:
        down_sampled_images.append(downsample(down_sampled_images[-1], scale))

    return down_sampled_images
