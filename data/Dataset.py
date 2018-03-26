import itertools
import numpy as np


class Dataset(object):
    ''' Dataset oject for GAN and CosmoGAN classes

        Transform should probably be False for a classical GAN.
    '''

    def __init__(self, X, shuffle=True, transform=None):
        ''' Initialize a Dataset object

        Arguments
        ---------
        * X         : numpy array containing the data
        * shuffle   : True if the data should be shuffled
        * transform : Function to be applied ot each batch of the dataset
                      This allows extend the dataset.

        '''

        self._shuffle = shuffle
        self._transform = transform
        self._N = len(X)
        if shuffle:
            self._p = np.random.permutation(self._N)
        else:
            self._p = np.arange(self._N)
        self._X = X[self._p]
        # self._batch_size = batch_size

    def get_all_data(self):
        ''' Return all the data (shuffled) '''
        return self._X

    def get_samples(self, N=100, transform=True):
        ''' Get the `N` first samples '''
        if self._transform and transform:
            return self._transform(self._X[:N])
        else:
            return self._X[:N]

    def iter(self, batch_size=1):
        return self.__iter__(batch_size)

    def __iter__(self, batch_size=1):
        for data in grouper(itertools.cycle(self._X), batch_size):
            if self._transform:
                yield self._transform(np.array(data))
            else:
                yield np.array(data)

    # @property
    # def batch_size(self):
    #     if self._batch_size is None:
    #         raise RuntimeError('Set the batch_size property')
    #     return self._batch_size

    # @batch_size.setter
    # def batch_size(self, value):
    #     ''' Set the batch_size 

    #     This function should be called before the `next` function.
    #     '''
    #     if not value or value < 0:
    #         raise ValueError('Should be a possitive number')
    #     self._batch_size = value

    @property
    def shuffle(self):
        ''' Is the dataset suffled? '''
        return self._shuffle

    @property
    def N(self):
        ''' Number of element in the dataset '''
        return self._N

    # def __getitem__(self, key):
    #     if isinstance(key, slice):
    #         return itertools.slice(self, key)


def grouper(iterable, n, fillvalue=None):
    """
    Collect data into fixed-length chunks or blocks. This function commes
    from itertools
    """
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return itertools.zip_longest(fillvalue=fillvalue, *args)


class Dataset_3d(Dataset):
    def __init__(self, X, spix=64, shuffle=True, transform=None):
        ''' Initialize a Dataset object

        Arguments
        ---------
        * X         : numpy array containing the data
        * shuffle   : True if the data should be shuffled
        * transform : Function to be applied ot each batch of the dataset
                      This allows extend the dataset.

        '''
        self._bigger_cubes = X
        self._spix = spix
        self._shuffle = shuffle
        self._transform = transform
        self.get_epoch_samples()


    def iter(self, batch_size=1):
        self.get_epoch_samples()
        return super().iter(batch_size)

    def __iter__(self, batch_size=1):
        for data in grouper(itertools.cycle(self._X), batch_size):
            yield np.array(data)

    def get_samples(self, N=100, transform=True):
        ''' Get the 'N' first samples '''
        return self._X[:N]

    def _slice_to_smaller_cubes(self):
        '''
        slice bigger cube to smaller cubes,
        and return the smaller cubes
        '''
        num_slices = self._bigger_cubes[0].shape[0] // self._spix
        limit = num_slices * self._spix
        
        smaller_cubes = []

        # iterate over each 3d histogram
        for i in range(len(self._bigger_cubes)):
            cube = self._bigger_cubes[i, :, :, :]

            if self._transform is not None:
                cube = self._transform(cube) # augment each 3d histogram
            
            cube = cube[:limit, :limit, :limit]

            splitted_0 = np.split(cube, num_slices, axis=0) # split the 3d histogram along 0th axis
            
            for j in range(len(splitted_0)):
                cube_0 = splitted_0[j]

                splitted_1 = np.split(cube_0, num_slices, axis=1) # split along the 1st axis

                for k in range(len(splitted_1)):
                    cube_1 = splitted_1[k]

                    splitted_2 = np.split(cube_1, num_slices, axis=2) # split along the 2nd axis
                    smaller_cubes.append(splitted_2)

        smaller_cubes = np.vstack(smaller_cubes).astype(np.float32)
        self._X = smaller_cubes


    def get_epoch_samples(self):
        '''
        Augment the bigger cubes first,
        then slice them into smaller cubes.
        This method is called at the start of every epoch
        '''
        self._slice_to_smaller_cubes()

        self._N = len(self._X)
        if self._shuffle:
            self._p = np.random.permutation(self._N)
        else:
            self._p = np.arange(self._N)

        self._X = self._X[self._p]
