import numpy as np
from blocks import downsample


def random_shift_2d(images):
    ''' Apply a random circshift to 2d images'''
    nx = images.shape[1]
    ny = images.shape[2]
    shiftx = np.random.randint(0, nx)
    shifty = np.random.randint(0, ny)
    out = np.roll(images, shift=shiftx, axis=1)
    out = np.roll(out, shift=shifty, axis=2)
    return out


def random_flip_2d(images):
    ''' Apply a random flip to 2d images'''
    out = images
    if np.random.rand(1) > 0.5:
        out = np.flip(out, axis=1)
    if np.random.rand(1) > 0.5:
        out = np.flip(out, axis=2)
    return out


def random_transpose_2d(images):
    '''
    Apply a random transpose to 2d images
    '''

    # all possible transposes
    transposes = [(0, 1, 2, 3), (0, 2, 1, 3)]
    transpose = transposes[np.random.choice(len(transposes))]
    return np.transpose(images, axes=transpose)

def random_rotate_2d(images):
    '''
    random rotation of 2d images by multiple of 90 degree
    '''
    return random_transpose_2d(random_flip_2d(images))

def random_transformation_2d(images):
    return random_rotate_2d(random_shift_2d(images))


# def make_smaller_samples(images, nx, ny=None):
#     ''' This function will be deleted in the future '''
#     if ny is None:
#         ny = nx
#     nsamples = images.shape[0]
#     spixx = images.shape[1]
#     spixy = images.shape[1]
#     cutx = spixx // nx
#     cuty = spixy // ny
#     img_small = np.zeros([nsamples * cutx * cuty, nx, ny])
#     for i in range(cutx):
#         for j in range(cuty):
#             l = j + i * cuty
#             img_small[l * nsamples:(
#                 l + 1) * nsamples, :, :] = images[:, i * nx:(
#                     i + 1) * nx, j * ny:(j + 1) * ny]

#     return img_small

# def random_crop_2d(images, nx, ny=None):
#     ''' Crop randomly the image to the size nx, ny '''
#     if ny is None:
#         ny = nx
#     sx = images.shape[1]
#     sy = images.shape[2]
#     dx = np.random.randint(0,sx-nx)
#     dy = np.random.randint(0,sy-ny)
#     return images[:,dx:dx+nx,dy:dy+ny]

# def down_sample_images(images, scalings):
#     down_sampled_images = []
#     down_sampled_images.append(images)

#     for scale in scalings:
#         down_sampled_images.append(downsample(down_sampled_images[-1], scale))

#     return down_sampled_images


def random_flip_3d(images):
    ''' Apply a random flip to 3d images'''
    out = images
    if np.random.rand(1) > 0.5:
        out = np.flip(out, axis=1)
    if np.random.rand(1) > 0.5:
        out = np.flip(out, axis=2)
    if np.random.rand(1) > 0.5:
        out = np.flip(out, axis=3)
    return out


def random_transpose_3d(images):
    '''
    Apply a random transpose to 3d images
    '''

    # all possible transposes
    transposes = [(0, 1, 2, 3), (0, 1, 3, 2), (0, 2, 1, 3),
                  (0, 2, 3, 1), (0, 3, 1, 2), (0, 3, 2, 1)]
    transpose = transposes[np.random.choice(len(transposes))]
    return np.transpose(images, axes=transpose)


def random_rotate_3d(images):
    '''
    random rotation of 3d images by multiple of 90 degree
    '''
    return random_transpose_3d(random_flip_3d(images))


def random_translate_3d(images):
    '''
    random translation of 3d images along an axis by some pixels greater than shift_pix
    '''
    trans1 = np.random.choice(range(images.shape[1]))
    images = np.roll(images, trans1, 1)

    trans2 = np.random.choice(range(images.shape[2]))
    images = np.roll(images, trans2, 1)

    trans3 = np.random.choice(range(images.shape[3]))
    images = np.roll(images, trans3, 1)

    return images


def random_transformation_3d(images):
    return random_translate_3d(random_rotate_3d(images))
