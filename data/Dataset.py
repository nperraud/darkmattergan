import itertools
import numpy as np
from utils import compose2
import functools


def do_noting(x):
    return x


class Dataset(object):
    ''' Dataset oject for GAN and CosmoGAN classes

        Transform should probably be False for a classical GAN.
    '''

    def __init__(self, X, shuffle=True, slice_fn=None, transform=None, dtype=np.float32):
        ''' Initialize a Dataset object

        Arguments
        ---------
        * X         : numpy array containing the data
        * shuffle   : True if the data should be shuffled
        * transform : Function to be applied ot each batch of the dataset
                      This allows extend the dataset.
        * slice_fn : Slicing function to cut the data into smaller parts
        '''
        X = X.astype(dtype)
        self._shuffle = shuffle
        if slice_fn:
            self._slice_fn = slice_fn
        else:
            self._slice_fn = do_noting
        if transform:
            self._transform = transform
        else:
            self._transform = do_noting

        self._data_process = compose2(self._transform, self._slice_fn)

        self._N = len(self._slice_fn(X))
        if shuffle:
            self._p = np.random.permutation(self._N)
        else:
            self._p = np.arange(self._N)
        self._X = X

    def get_all_data(self):
        ''' Return all the data (shuffled) '''
        return self._data_process(self._X)[self._p]

    def get_samples(self, N=100):
        ''' Get the `N` first samples '''
        return self._data_process(self._X)[self._p[:N]]

    def iter(self, batch_size=1):
        return self.__iter__(batch_size)

    def __iter__(self, batch_size=1):

        if batch_size > self.N:
            raise ValueError(
                'Batch size greater than total number of samples available!')

        # Reshuffle the data
        if self.shuffle:
            self._p = np.random.permutation(self._N)
        nel = (self._N // batch_size) * batch_size
        transformed_data = self._data_process(self._X)[self._p[range(nel)]]
        for data in grouper(transformed_data, batch_size):
            yield np.array(data)

    @property
    def shuffle(self):
        ''' Is the dataset suffled? '''
        return self._shuffle

    @property
    def N(self):
        ''' Number of element in the dataset '''
        return self._N


class Dataset_3d(Dataset):
    def __init__(self, X, spix=64, shuffle=True, transform=None):
        ''' Initialize a Dataset object
        Arguments
        ---------
        * X         : numpy array containing the data
        * shuffle   : True if the data should be shuffled
        * transform : Function to be applied to each bigger cube in the dataset
                      for data augmentation
        '''

        slice_fn = functools.partial(slice_3d, spix=spix)
        super().__init__(
            X=X, shuffle=shuffle, slice_fn=slice_fn, transform=transform)


class Dataset_2d(Dataset):
    def __init__(self, X, spix=128, shuffle=True, transform=None):
        ''' Initialize a Dataset object
        Arguments
        ---------
        * X         : numpy array containing the data
        * shuffle   : True if the data should be shuffled
        * transform : Function to be applied to each bigger cube in the dataset
                      for data augmentation
        '''

        slice_fn = functools.partial(slice_2d, spix=spix)
        super().__init__(
            X=X, shuffle=shuffle, slice_fn=slice_fn, transform=transform)

class Dataset_time(Dataset):
    def __init__(self, X, spix=128, shuffle=True, transform=None):
        ''' Initialize a Dataset object
        Arguments
        ---------
        * X         : numpy array containing the data
        * shuffle   : True if the data should be shuffled
        * transform : Function to be applied to each bigger cube in the dataset
                      for data augmentation
        '''

        slice_fn = functools.partial(slice_time, spix=spix)
        super().__init__(
            X=X, shuffle=shuffle, slice_fn=slice_fn, transform=transform)

class Dataset_2d_patch(Dataset):
    def __init__(self, X, spix=128, shuffle=True, transform=None):
        ''' Initialize a Dataset object for the 2d patch case
        Arguments
        ---------
        * X         : numpy array containing the data
        * shuffle   : True if the data should be shuffled
        * transform : Function to be applied to each bigger cube in the dataset
                      for data augmentation
        '''

        slice_fn = functools.partial(slice_2d_patch, spix=spix)
        super().__init__(
            X=X, shuffle=shuffle, slice_fn=slice_fn, transform=transform)

    def get_samples_full(self, N=100):
        X = self.get_samples(N=N)
        X_d = np.concatenate([X[:, :, :, 1], X[:, :, :, 0]], axis=1)
        X_u = np.concatenate([X[:, :, :, 3], X[:, :, :, 2]], axis=1)
        X_r = np.squeeze(np.concatenate([X_u, X_d], axis=2))
        return X_r


class Dataset_3d_patch(Dataset):
    def __init__(self, X, spix=32, shuffle=True, transform=None):
        ''' Initialize a Dataset object for the 3d patch case
        Arguments
        ---------
        * X         : numpy array containing the data
        * shuffle   : True if the data should be shuffled
        * transform : Function to be applied to each bigger cube in the dataset
                      for data augmentation
        '''

        slice_fn = functools.partial(slice_3d_patch, spix=spix)
        super().__init__(
            X=X, shuffle=shuffle, slice_fn=slice_fn, transform=transform)


def grouper(iterable, n, fillvalue=None):
    """
    Collect data into fixed-length chunks or blocks. This function commes
    from itertools
    """
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return itertools.zip_longest(fillvalue=fillvalue, *args)


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
    num_slices = cubes.shape[1] // spix

    # To ensure left over pixels in each dimension are ignored
    limit = num_slices * spix
    cubes = cubes[:, :limit, :limit, :limit]

    # split along first dimension
    sliced_dim1 = np.vstack(np.split(cubes, num_slices, axis=1))
    # split along second dimension
    sliced_dim2 = np.vstack(np.split(sliced_dim1, num_slices, axis=2))
    # split along third dimension
    sliced_dim3 = np.vstack(np.split(sliced_dim2, num_slices, axis=3))

    return sliced_dim3


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
    sliced_dim1 = np.vstack(np.split(img_with_nbrs, nx, axis=1))
    sliced_dim2 = np.vstack(np.split(sliced_dim1, ny, axis=2))
    sliced_dim3 = np.vstack(np.split(sliced_dim2, nz, axis=3))

    return sliced_dim3
