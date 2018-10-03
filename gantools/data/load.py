import numpy as np
import os
from gantools import utils
from gantools.data import gaussian_synthetic_data
from gantools.data import path
from gantools.data import transformation, fmap
from gantools.data.Dataset import Dataset_2d, Dataset_3d, Dataset_2d_patch, Dataset_3d_patch, Dataset_time, Dataset
from gantools.data import Dataset_medical
from skimage import io
from functools import partial


from gantools import blocks


def load_data_from_dir(dir_path, k=10):
    '''
    load training data, saved as ndarrays in files in directory dir_path
    '''
    raw_data = []
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        arr = utils.load_hdf5(
            filename=file_path, dataset_name='data', mode='r')
        raw_data.append(arr)

    raw_data = np.array(raw_data)
    forward_mapped_data = fmap.forward_map(raw_data, k)

    return forward_mapped_data, raw_data


def load_data_from_file(file_path, k=10):
    '''
    load training data, saved as ndarrays in file
    '''
    raw_data = []
    raw_data = utils.load_hdf5(
        filename=file_path, dataset_name='data', mode='r')
    if type(raw_data) is not np.ndarray:
        raise ValueError(
            "Data stroed in file {} is not of type np.ndarray".format(
                file_path))

    forward_mapped_data = fmap.forward_map(raw_data, k)

    return forward_mapped_data, raw_data


def load_3d_synthetic_samples(nsamples, dim, k):
    images = 2 * gaussian_synthetic_data.generate_cubes( # Forward mapped
        nsamples=nsamples, cube_dim=dim) - 1.0
    raw_images = fmap.backward_map(images)

    return Dataset_3d(images, spix=dim, shuffle=False, transform=None), raw_images


def load_2d_synthetic_samples(nsamples, dim, k):
    images = 2 * gaussian_synthetic_data.generate_squares( # Forward mapped
        nsamples=nsamples, square_dim=dim) - 1.0
    raw_images = fmap.backward_map(images)

    return Dataset_2d(images, spix=dim, shuffle=False, transform=None)


def load_samples_raw(nsamples=None, resolution=256, Mpch=70):
    ''' Load 2D or 3D raw images

    Arguments
    ---------
    * nsamples : desired number of samples (if None: all of them)
    * resolution : [256, 512]
    * Mpch : [70, 350]

    '''
    rootpath = path.root_path()
    input_pattern = '{}_nbody_{}Mpc'.format(resolution, Mpch)
    file_ext = '.h5'
    queue = []
    for file in os.listdir(rootpath):
        if file.endswith(file_ext) and input_pattern in file:
            queue.append(os.path.join(rootpath, file))
            # if len(queue) == 10:
            #     break

    if len(queue) == 0:
        raise LookupError('No file founds, check path and parameters')
    raw_images = []
    for file_path in queue:
        raw_images.append(
            utils.load_hdf5(
                filename=file_path, dataset_name='data', mode='r'))
        if type(raw_images[-1]) is not np.ndarray:
            raise ValueError(
                "Data stored in file {} is not of type np.ndarray".format(
                    file_path))

    raw_images = np.array(raw_images).astype(np.float32)


    if nsamples is None:
        return raw_images
    else:
        if nsamples > len(raw_images):
            raise ValueError("Not enough sample")
        else:
            print('Select {} samples out of {}.'.format(
                nsamples, len(raw_images)))

        return raw_images[:nsamples]


def load_dataset(
        nsamples=None,
        resolution=256,
        Mpch=70,
        shuffle=True,
        forward_map = None,
        spix=128,
        augmentation=True,
        scaling=1,
        is_3d=False,
        patch=False):

    ''' Load a 2D dataset object 

     Arguments
    ---------
    * nsamples : desired number of samples, if None => all of them (default None)
    * resolution : [256, 512] (default 256)
    * Mpch : [70, 350] (default 70)
    * shuffle: shuffle the data (default True)
    * foward : foward mapping use None for raw data (default None)
    * spix : resolution of the image (default 128)
    * augmentation : use data augmentation (default True)
    * scaling : downscale the image by a factor (default 1)
    * is_3d : load a 3d dataset (default False)
    * patch: experimental feature for patchgan
    '''

    # 1) Load raw images
    images = load_samples_raw(nsamples=nsamples, resolution=resolution, Mpch=Mpch)
    print("images shape = ", images.shape)

    # 2) Apply forward map if necessary
    if forward_map:
        images = forward_map(images)

    if (not is_3d):
        sh = images.shape
        images = images.reshape([sh[0]*sh[1], sh[2], sh[3]])

    # 2p) Apply downscaling if necessary
    if scaling>1:
        if is_3d:
            data_shape = 3
        else:
            data_shape = 2
        images = blocks.downsample(images, scaling, size=data_shape)

    if augmentation:
        # With the current implementation, 3d augmentation is not supported
        # for 2d scaling
        if scaling>1 and not is_3d:
            t = transformation.random_transformation_2d
        else:
            t = transformation.random_transformation_3d
    else:
        t = None
    
    # 5) Make a dataset
    if patch:
        if is_3d:
            dataset = Dataset_3d_patch(images, spix=spix, shuffle=shuffle, transform=t)
        else:
            dataset = Dataset_2d_patch(images, spix=spix, shuffle=shuffle, transform=t)

    else:
        if is_3d:
            dataset = Dataset_3d(images, spix=spix, shuffle=shuffle, transform=t)
        else:
            dataset = Dataset_2d(images, spix=spix, shuffle=shuffle, transform=t)

    return dataset


# def load_dataset_file(
#         nsamples=None,
#         resolution=256,
#         Mpch=70,
#         shuffle=True,
#         forward_map = None,
#         spix=128,
#         augmentation=True,
#         scaling=1,
#         is_3d=False,
#         patch=False):

#     ''' Load a 2D dataset object 

#      Arguments
#     ---------
#     * nsamples : desired number of samples, if None => all of them (default None)
#     * resolution : [256, 512] (default 256)
#     * Mpch : [70, 350] (default 70)
#     * shuffle: shuffle the data (default True)
#     * foward : foward mapping use None for raw data (default None)
#     * spix : resolution of the image (default 128)
#     * augmentation : use data augmentation (default True)
#     * scaling : downscale the image by a factor (default 1)
#     * is_3d : load a 3d dataset (default False)
#     * patch: experimental feature for patchgan
#     '''

#     if augmentation:
#         # With the current implementation, 3d augmentation is not supported
#         # for 2d scaling
#         if scaling>1 and not is_3d:
#             t = transformation.random_transformation_2d
#         else:
#             t = transformation.random_transformation_3d
#     else:
#         t = None
    
#     # 5) Make a dataset
#     if patch:
#         if is_3d:
#             dataset = Dataset_file_3d_patch(resolution=resolution, Mpch=Mpch,
#             forward_map = forward_map, scaling=scaling, 
#             spix=spix, shuffle=shuffle, transform=t)
#         else:
#             dataset = Dataset_file_2d_patch(resolution=resolution, Mpch=Mpch,
#             forward_map = forward_map, scaling=scaling, 
#             spix=spix, shuffle=shuffle, transform=t)

#     else:
#         if is_3d:
#             dataset = Dataset_file_3d(resolution=resolution, Mpch=Mpch,
#             forward_map = forward_map, scaling=scaling,
#             spix=spix, shuffle=shuffle, transform=t)
#         else:
#             dataset = Dataset_file_2d(resolution=resolution, Mpch=Mpch,
#             forward_map = forward_map, scaling=scaling,
#             spix=spix, shuffle=shuffle, transform=t)

#     return dataset



    
def load_time_dataset(
        resolution=256,
        Mpch=100,
        shuffle=True,
        forward_map = None,
        spix=128,
        augmentation=True):

    ''' Load a 2D dataset object 

     Arguments
    ---------
    * resolution : [256, 512] (default 256)
    * Mpch : [100, 500] (default 70)
    * shuffle: shuffle the data (default True)
    * foward : foward mapping use None for raw data (default None)
    * spix : resolution of the image (default 128)
    * augmentation : use data augmentation (default True)
    '''

    # 1) Load raw images
    images = load_time_cubes(resolution=resolution, Mpch=Mpch)
    # (ts, resolution, resolution, resolution)

    # 2) Apply forward map if necessary
    if forward_map:
        images = forward_map(images)
    if augmentation:
        t = transformation.random_transformation_3d
    else:
        t = None

    # 5) Make a dataset
    dataset = Dataset_time(X=images, shuffle=shuffle, slice_fn=slice_fn, transform=transform)

    return dataset


def load_medical_data():
    pathdata = os.path.join(path.medical_path(),'volumedata.tif')
    return np.array(io.imread(pathdata))

def do_nothing(x):
    return x

def load_medical_dataset(
        shuffle=True,
        forward_map=None,
        spix=32,
        augmentation=True,
        scaling=1,
        patch=False):

    ''' Load a 2D dataset object 

     Arguments
    ---------
    * shuffle: shuffle the data (default True)
    * forward : foward mapping use None for raw data (default None)
    * spix : resolution of the image (default 128)
    * augmentation : use data augmentation (default True)
    * scaling : downscale the image by a factor (default 1)
    * patch: experimental feature for patchgan
    '''
    images = load_medical_data()
    images = images.reshape([1, *images.shape])
    if scaling>1:
        images = blocks.downsample(images, scaling, 3)

    # 5) Make a dataset
    if patch:
        dataset = Dataset_medical.DatasetMedical(images, augmentation=augmentation,
        spix=spix, shuffle=shuffle, transform=forward_map)
    else:
        if augmentation:
            t = transformation.random_rotate_3d
        else:
            t = do_nothing
        if forward_map:
            images = forward_map(images)
        slice_fn = partial(Dataset_medical.slice_3d, spix=spix)
        dataset = Dataset(images, slice_fn=slice_fn,
        shuffle=shuffle, transform=t)

    return dataset


def load_nysnth_rawdata():
    pathdata = os.path.join(path.nsynth_path(), 'nsynth-valid.npz')
    return np.load(pathdata)['arr_0']


def load_nsynth_dataset(
        shuffle=True,
        scaling=1):

    ''' Load a Nsynth dataset object 

     Arguments
    ---------
    * shuffle: shuffle the data (default True)
    * scaling : downscale the image by a factor (default 1)
    '''
    sig = load_nysnth_rawdata()
    
    # 1) Transform the data
    def transform(x):
        x = x[:, :2**15]/(2**15)
        x = (0.99*x.T/np.max(np.abs(x), axis=1)).T
        return x
    sig = transform(sig)

    # 2) Downsample
    if scaling>1:
        sig = blocks.downsample(sig, scaling, 1)

    # 3) Make a dataset
    dataset = Dataset(sig, shuffle=shuffle)

    return dataset
