import numpy as np
import tensorflow as tf


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


def patch2img(patches, is_3d=False):
    if is_3d:
        imgs_down_left = np.concatenate([patches[:, :, :, :, 3], patches[:, :, :, :,2]], axis=2)
        imgs_down_right = np.concatenate([patches[:, :, :, :, 1], patches[:, :, :, :,0]], axis=2)
        imgs_down = np.concatenate([imgs_down_left, imgs_down_right], axis=3)
        imgs_up_left   = np.concatenate([patches[:, :, :, :, 7], patches[:, :, :, :, 6]], axis=2)
        imgs_up_right  = np.concatenate([patches[:, :, :, :, 5], patches[:, :, :, :, 4]], axis=2)
        imgs_up = np.concatenate([ imgs_up_left, imgs_up_right], axis=3)
        imgs = np.concatenate([imgs_up, imgs_down], axis=1)
    else:
        imgs_d = np.concatenate(
            [patches[:, :, :, 1], patches[:, :, :, 0]], axis=1)
        imgs_u = np.concatenate(
            [patches[:, :, :, 3], patches[:, :, :, 2]], axis=1)
        imgs = np.concatenate([imgs_u, imgs_d], axis=2)
    return imgs

def tf_patch2img_2d(dr, dl, ur, ul):

    imgs_d = tf.concat([dl, dr], axis=1)
    imgs_u = tf.concat([ul, ur], axis=1)
    imgs = tf.concat([imgs_u,  imgs_d], axis=2)
    return imgs

def tf_patch2img_3d(*args):
    imgs_down_left = tf.concat([args[3], args[2]], axis=2)
    imgs_down_right = tf.concat([args[1], args[0]], axis=2)
    imgs_down = tf.concat([imgs_down_left, imgs_down_right], axis=3)
    imgs_up_left   = tf.concat([args[7], args[6]], axis=2)
    imgs_up_right  = tf.concat([args[5], args[4]], axis=2)
    imgs_up = tf.concat([ imgs_up_left, imgs_up_right], axis=3)
    imgs = tf.concat([imgs_up, imgs_down], axis=1)
    return imgs


def flip_slices_2d(dl, ur, ul):
    flip_dl = np.flip(dl, axis=1)
    flip_ur = np.flip(ur, axis=2)    
    flip_ul = np.flip(np.flip(ul, axis=1), axis=2)
    return flip_dl, flip_ur, flip_ul

def tf_flip_slices_2d(dl, ur, ul):
    flip_dl = tf.reverse(dl, axis=[1])
    flip_ur = tf.reverse(ur, axis=[2])    
    flip_ul = tf.reverse(ul, axis=[1,2])
    return flip_dl, flip_ur, flip_ul

def tf_flip_slices(*args, size=2):
    if size==3:
        return tf_flip_slices_3d(*args)
    elif size==2:
        return tf_flip_slices_2d(*args)
    elif size==1:
        return tf.reverse(*args, axis=[1])
    else:
        raise ValueError("Size should be 1, 2 or 3")


def tf_patch2img(*args, size=2):
    if size==3:
        return tf_patch2img_3d(*args)
    elif size==2:
        return tf_patch2img_2d(*args)
    elif size==1:
        raise NotImplementedError("To be done and tested - should be trivial")
    else:
        raise ValueError("Size should be 1, 2 or 3")

def flip_slices_3d(*args):
    flip_d_above = np.flip(args[0], axis=2)
    flip_d_left = np.flip(args[1], axis=3)
    flip_d_corner = np.flip(np.flip(args[2], axis=2), axis=3)
    flip_up = np.flip(args[3], axis=1)
    flip_u_above = np.flip(np.flip(args[4], axis=1), axis=2)
    flip_u_left = np.flip(np.flip(args[5], axis=1), axis=3)
    flip_u_corner = np.flip(np.flip(np.flip(args[6], axis=1), axis=2), axis=3)
    return flip_d_above, flip_d_left, flip_d_corner, flip_up, flip_u_above, flip_u_left, flip_u_corner

def tf_flip_slices_3d(*args):
    flip_d_above = tf.reverse(args[0], axis=[2])
    flip_d_left = tf.reverse(args[1], axis=[3])
    flip_d_corner = tf.reverse(args[2], axis=[2, 3])
    flip_up = tf.reverse(args[3], axis=[1])
    flip_u_above = tf.reverse(args[4], axis=[1, 2])
    flip_u_left = tf.reverse(args[5], axis=[1, 3])
    flip_u_corner = tf.reverse(args[6], axis=[1, 2, 3])
    return flip_d_above, flip_d_left, flip_d_corner, flip_up, flip_u_above, flip_u_left, flip_u_corner




def slice_time(cubes, spix=64):
    '''
    slice each cube in cubes to smaller cubes,
    and return all the smaller cubes
    '''
    s = cubes.shape
    # The last dimension is used for the samples
    cubes = cubes.transpose([1, 2, 3, 0])

    # compute the number of slices (We assume square images)
    num_slices = cubes.shape[1] // spix

    # To ensure left over pixels in each dimension are ignored
    limit = num_slices * spix
    cubes = cubes[:, :limit, :limit]

    # split along first dimension
    sliced_dim1 = np.vstack(np.split(cubes, num_slices, axis=1))

    # split along second dimension
    sliced_dim2 = np.vstack(np.split(sliced_dim1, num_slices, axis=2))

    return sliced_dim2


def slice_2d(cubes, spix=64):
    '''
    slice each cube in cubes to smaller cubes,
    and return all the smaller cubes
    '''
    s = cubes.shape
    # The last dimension is used for the samples
    cubes = cubes.transpose([0, 3, 1, 2])

    cubes = cubes.reshape([s[0] * s[3], s[1], s[2]])

    # compute the number of slices (We assume square images)
    num_slices = s[2] // spix

    # To ensure left over pixels in each dimension are ignored
    limit = num_slices * spix
    cubes = cubes[:, :limit, :limit]

    # split along first dimension
    sliced_dim1 = np.vstack(np.split(cubes, num_slices, axis=1))

    # split along second dimension
    sliced_dim2 = np.vstack(np.split(sliced_dim1, num_slices, axis=2))

    return sliced_dim2


def slice_3d(cubes, spix=64):
    '''
    slice each cube in cubes to smaller cubes,
    and return all the smaller cubes
    '''
    num_slices_dim_1 = cubes.shape[1] // spix
    num_slices_dim_2 = cubes.shape[2] // spix
    num_slices_dim_3 = cubes.shape[3] // spix

    # To ensure left over pixels in each dimension are ignored
    limit_dim_1 = num_slices_dim_1 * spix
    limit_dim_2 = num_slices_dim_2 * spix
    limit_dim_3 = num_slices_dim_3 * spix

    cubes = cubes[:, :limit_dim_1, :limit_dim_2, :limit_dim_3]

    # split along first dimension
    cubes = np.vstack(np.split(cubes, num_slices_dim_1, axis=1))
    # split along second dimension
    cubes = np.vstack(np.split(cubes, num_slices_dim_2, axis=2))
    # split along third dimension
    cubes = np.vstack(np.split(cubes, num_slices_dim_3, axis=3))

    return cubes


def slice_2d_patch(img0, spix=64):

    # Handle the dimesnsions
    l = len(img0.shape)
    if l < 2:
        ValueError('Not enough dimensions')
    elif l == 2:
        img0 = img0.reshape([1, *img0.shape])
    elif l == 4:
        s = img0.shape
        img0 = img0.reshape([s[0] * s[1], s[2], s[3]])
    elif l > 4:
        ValueError('To many dimensions')
    _, sx, sy = img0.shape
    nx = sx // spix
    ny = sy // spix

    # 1) Create the different subparts
    img1 = np.roll(img0, spix, axis=1)
    img1[:, :spix, :] = 0

    img2 = np.roll(img0, spix, axis=2)
    img2[:, :, :spix] = 0

    img3 = np.roll(img1, spix, axis=2)
    img3[:, :, :spix] = 0

    # 2) Concatenate
    img = np.stack([img0, img1, img2, img3], axis=3)

    # 3) Slice the image
    img = np.vstack(np.split(img, nx, axis=1))
    img = np.vstack(np.split(img, ny, axis=2))

    return img


def slice_3d_patch(cubes, spix=32):
    '''
    cubes: the 3d histograms - [:, :, :, :]
    '''

    # Handle the dimesnsions
    l = len(cubes.shape)
    if l < 3:
        ValueError('Not enough dimensions')
    elif l == 3:
        cubes = cubes.reshape([1, *cubes.shape]) # add one extra dimension for number of cubes
    elif l > 4:
        ValueError('To many dimensions')

    _, sx, sy, sz = cubes.shape
    nx = sx // spix
    ny = sy // spix
    nz = sz // spix

    # 1) Create all 7 neighbors for each smaller cube
    img1 = np.roll(cubes, spix, axis=2)
    img1[:, :, :spix, :] = 0

    img2 = np.roll(cubes, spix, axis=3)
    img2[:, :, :, :spix] = 0
    
    img3 = np.roll(img1, spix, axis=3)
    img3[:, :, :, :spix] = 0
    
    img4 = np.roll(cubes, spix, axis=1) # extra for the 3D case
    img4[:, :spix, :, :] = 0
    
    img5 = np.roll(img4, spix, axis=2)
    img5[:, :, :spix, :] = 0
    
    img6 = np.roll(img4, spix, axis=3)
    img6[:, :, :, :spix] = 0
    
    img7 = np.roll(img5, spix, axis=3)
    img7[:, :, :, :spix] = 0
    

    # 2) Concatenate
    img_with_nbrs = np.stack([cubes, img1, img2, img3, img4, img5, img6, img7], axis=4) # 7 neighbors plus the original cube


    # 3) Slice the cubes
    img_with_nbrs = np.vstack(np.split(img_with_nbrs, nx, axis=1))
    img_with_nbrs = np.vstack(np.split(img_with_nbrs, ny, axis=2))
    img_with_nbrs = np.vstack(np.split(img_with_nbrs, nz, axis=3))

    return img_with_nbrs
