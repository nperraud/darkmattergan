import itertools
from data import transformation
import numpy as np


class Dataset(object):
    ''' Dataset oject for GAN and CosmoGAN classes

        Transform should probably be False for a classical GAN.
    '''

    def __init__(self, X, shuffle=True, transform=None, batch_size=None):
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
        self._batch_size = batch_size

    @property
    def batch_size(self):
        if self._batch_size is None:
            raise RuntimeError('Set the batch_size property')
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        ''' Set the batch_size 

        This function should be called before the `next` function.
        '''
        if not value or value < 0:
            raise ValueError('Should be a possitive number')
        self._batch_size = value

    def get_dataset(self):
        ''' Return all the dataset (shuffled) '''
        return self._X

    def get_samples(self, N=100):
        ''' Get the `N` first samples '''
        return self._X[:N]

    def __iter__(self):
        for data in grouper(itertools.cycle(self._X), self.batch_size):
            if self._transform:
                yield self._transform(data)
            else:
                yield data

    @property
    def shuffle(self):
        ''' Is the dataset suffled? '''
        return self._shuffle

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