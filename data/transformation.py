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
        out = out[:,::-1,:]
    if np.random.rand(1) > 0.5: 
        out = out[:,:,::-1]
    return out

def random_transformation_2d(images):
    return random_flip_2d(random_shift_2d(images))



def make_smaller_samples(images, nx, ny=None):
    ''' This function will be deleted in the future '''
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

def random_crop_2d(images, nx, ny=None):
    ''' Crop randomly the image to the size nx, ny '''
    if ny is None:
        ny = nx
    sx = images.shape[1]
    sy = images.shape[2]
    dx = np.random.randint(0,sx-nx)
    dy = np.random.randint(0,sy-ny)
    return images[:,dx:dx+nx,dy:dy+ny]


def down_sample_images(images, scalings):
    down_sampled_images = []
    down_sampled_images.append(images)

    for scale in scalings:
        down_sampled_images.append(downsample(down_sampled_images[-1], scale))

    return down_sampled_images

def rotate_3d(hist_3d):
    '''
    random rotation along a plane by multiple of 90 degree
    '''
    k = np.random.choice([0, 1, 2, 3]) # Number of times to rotate by 90 degree
    axes_rot = [(0,1), (0,2), (1,0), (1,2), (2,0), (2,1)] # plane of rotation
    axes_rot_choice = np.random.choice(len(axes_rot))
    axes_rot = axes_rot[axes_rot_choice]

    hist_3d_rot = np.rot90(hist_3d, k, axes_rot)
    return hist_3d_rot

def translate_3d(hist_3d, shift_pix=40):
    '''
    random translation along an axis by some pixels greater than 40
    '''
    trans = np.random.choice(range(shift_pix, hist_3d.shape[0])) # Magnitude of translation
    axis_trans = np.random.choice([0, 1, 2]) # Axis along which translation will be done 

    hist_3d_trans = np.roll(hist_3d, trans, axis_trans)
    return hist_3d_trans
