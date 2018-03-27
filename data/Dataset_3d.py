import numpy as np

def slice_3d(cubes, spix=64, shuffle=False):
    '''
    slice each cube in cubes to smaller cubes,
    and return all the smaller cubes
    '''
    num_slices = cubes[0].shape[1] // spix
    limit = num_slices * spix  # To ensure left over pixels in each dimension are ignored
    cubes = cubes[:, :limit, :limit, :limit]

    sliced_dim1 = np.vstack(np.split(cubes,       num_slices, axis=1)) # split along first dimension
    sliced_dim2 = np.vstack(np.split(sliced_dim1, num_slices, axis=2)) # split along second dimension
    sliced_dim3 = np.vstack(np.split(sliced_dim2, num_slices, axis=3)) # split along third dimension

    if shuffle:
        perm = np.random.permutation(len(sliced_dim3))
        sliced_dim3 = sliced_dim3[perm]

    return sliced_dim3


def get_batch(collection, batch_size=1):
    """
    Collect data into fixed-length chunks or blocks.
    Leave out the last few left over samples.
    """
    size = len(collection)

    for i in range(0, size, batch_size):
    	if i+batch_size <= size: # Make sure that left over samples are ignored
    		yield collection[i:i+batch_size, :]


class Dataset_3d(object):
    def __init__(self, X, spix=64, shuffle=True, transform=None, slice_fn=slice_3d):
        ''' Initialize a Dataset object
        Arguments
        ---------
        * X         : numpy array containing the data
        * shuffle   : True if the data should be shuffled
        * transform : Function to be applied to each bigger cube in the dataset
        '''
        self._X = X
        self._spix = spix
        self._shuffle = shuffle
        self._transform = transform
        self._slice_fn = slice_fn

        #calculate total number of samples
        self._N = len(self._slice_fn(self._X, self._spix, self._shuffle))

    def iter(self, batch_size=1):
        '''
        return an iterator to the smaller cube samples,
        where iterator corresponds to batch_size number of samples
        '''
        if batch_size > self.N:
            raise ValueError('Batch size greater than total number of samples available!')

        return self.__iter__(batch_size)

    def __iter__(self, batch_size=1):
        # 1) Augment data
        transformed_data = self._X
        if self._transform is not None:
        	transformed_data = self._transform(self._X)

        # 2) Cut data
        all_smaller_cubes = self._slice_fn(transformed_data, self._spix, self._shuffle)

        # 3) Iterate over batches
        for smaller_cube_batch in get_batch(all_smaller_cubes, batch_size):
            yield smaller_cube_batch

    def get_all_data(self):
        ''' Return all the data (shuffled) '''
        return self._slice_fn(self._X, self._spix)

    def get_samples(self, N=100, transform=True):
        ''' Get the 'N' first samples '''
        return self._slice_fn(self._X, self._spix)[:N]

    @property
    def N(self):
        ''' Number of element in the dataset '''
        return self._N