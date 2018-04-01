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

    def __init__(self, X, shuffle=True, slice_fn=None, transform=None):
        ''' Initialize a Dataset object

        Arguments
        ---------
        * X         : numpy array containing the data
        * shuffle   : True if the data should be shuffled
        * transform : Function to be applied ot each batch of the dataset
                      This allows extend the dataset.
        * slice_fn : Slicing function to cut the data into smaller parts
        '''


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

    def get_samples(self, N=100, transform=True):
        ''' Get the `N` first samples '''
        return self._data_process(self._X)[self._p[:N]]

    def iter(self, batch_size=1):
        return self.__iter__(batch_size)

    def __iter__(self, batch_size=1):

        if batch_size > self.N:
            raise ValueError('Batch size greater than total number of samples available!')

        # Reshuffle the data
        if self.shuffle:
            self._p = np.random.permutation(self._N)
        for data in grouper(self._data_process(self._X)[self._p], batch_size):
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
        super().__init__(X=X, shuffle=shuffle, slice_fn=slice_fn, transform=transform)

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
        super().__init__(X=X, shuffle=shuffle, slice_fn=slice_fn, transform=transform)



def grouper(iterable, n, fillvalue=None):
    """
    Collect data into fixed-length chunks or blocks. This function commes
    from itertools
    """
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return itertools.zip_longest(fillvalue=fillvalue, *args)




def slice_2d(cubes, spix=64):
    '''
    slice each cube in cubes to smaller cubes,
    and return all the smaller cubes
    '''
    s = cubes.shape

    cubes = cubes.reshape([s[0]*s[1],s[2],s[3]])
    num_slices = s[2] // spix

    # To ensure left over pixels in each dimension are ignored
    limit = num_slices * spix  
    cubes = cubes[:, :limit, :limit]
    
    # split along first dimension
    sliced_dim1 = np.vstack(np.split(cubes,       num_slices, axis=1)) 
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
    sliced_dim1 = np.vstack(np.split(cubes,       num_slices, axis=1)) 
    # split along second dimension
    sliced_dim2 = np.vstack(np.split(sliced_dim1, num_slices, axis=2)) 
    # split along third dimension
    sliced_dim3 = np.vstack(np.split(sliced_dim2, num_slices, axis=3)) 

    return sliced_dim3
