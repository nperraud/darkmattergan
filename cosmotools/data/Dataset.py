

class Dataset_parameters(Dataset):

    def __init__(self, X, params, shuffle=True, transform=None):
        ''' Initialize a Dataset object
        Arguments
        ---------
        * X         : numpy array containing the data
        * params    : numpy array containing the parameters that generated the data
        * shuffle   : True if the data should be shuffled
        * transform : transorm function to be applied
                      Note: the function should not augment the data samples
        '''

        # Shuffle data
        self._params = params
        super().__init__(X=X, shuffle=shuffle, transform=transform)

    def get_all_data(self):
        ''' Return all the data (shuffled) '''
        return [super().get_all_data(), self._params[self._p]]

    def get_samples(self, N=100):
        ''' Get the `N` first samples '''
        return [super().get_samples(N=N), self._params[self._p[:N]]]

    def __iter__(self, batch_size=1):

        if batch_size > self.N:
            raise ValueError(
                'Batch size greater than total number of samples available!')

        # Reshuffle the data
        if self.shuffle:
            self._p = np.random.permutation(self._N)
        nel = (self._N // batch_size) * batch_size
        transformed_data = self._data_process(self._X)[self._p[range(nel)]]
        transformed_params = self._params[self._p[range(nel)]]
        transformed_data_params = zip(transformed_data, transformed_params)
        for data in grouper(transformed_data_params, batch_size):
            yield np.array(data)


class Dataset_parameters_h5(Dataset_parameters):

    def __init__(self, filename, data_key, params_key, shape, batch=1000, transform=None):
        ''' Initialize a Dataset object
        Arguments
        ---------
        * filename  : name of the hdf5 file containing the data
        * batch     : maximum number of samples to be loaded in memory
        * data_key  : name of the dataset containing the actual data
        * params_key: name of the dataset containing the parameters
        * shape     : shape of the 2D images
        * transform : a transorm fucntion to be applied
        '''

        # Note: assume that the file is pre-shuffled
        self._batch = batch
        self._file = filename
        self._data_key = data_key
        self._params_key = params_key
        self._shape = shape
        if transform is None:
            self._transform = do_nothing
        else:
            self._transform = transform
        with h5py.File(self._file, 'r') as fh5:
            self._N = len(fh5[self._params_key])

        # Note: here data process is simply a transformation, otherwise if a slice function is applied
        # one would need to somehow replicate the parameters
        self._data_process = self._transform

    # Note: only the current batch is returned
    def get_all_data(self):
        return self.get_samples(self._batch)

    def load_batch(self, start, N=None):
        if N is None:
            N = self._batch
        with h5py.File(self._file, 'r') as fh5:
            train_maps = np.array(fh5[self._data_key][start:(start + N)])
            train_labels = np.array(fh5[self._params_key][start:(start + N)])
        return [train_maps.reshape((len(train_maps), self._shape[0], self._shape[1])), train_labels]

    def get_samples(self, N=100):
        ''' Get the `N` first samples '''
        if (N > self._batch):
            raise ValueError("N is bigger than the max amount of samples loadable in memory")
        X, params = self.load_batch(0)
        if len(X) < N:
            N = len(X)
        return [self._data_process(X[:N]), params[:N]]

    def __iter__(self, batch_size=1):

        if batch_size > self.N:
            raise ValueError('Batch size greater than total number of samples available!')
        if batch_size > self._batch:
            raise ValueError('Batch size greater than total number of samples loadable in memory!')

        # Number of batches to be loaded in memory
        nba = int(np.ceil(self._N / self._batch))

        # Check if the last batch contains at least batch_size elements
        if self._N - (nba - 1) * self._batch < batch_size:
            nba = nba - 1

        for b in range(nba):

            # Load batch into memory
            data, params = self.load_batch(b * self._batch)

            # Return iterator
            nel = (len(data) // batch_size) * batch_size
            data = self._data_process(data[range(nel)])
            params = params[range(nel)]
            transformed_data_params = zip(data, params)
            for data in grouper(transformed_data_params, batch_size):
                yield np.array(data)


# Assume that the parameters are sorted (grouped)
class Dataset_parameters_h5_sorted(Dataset_parameters_h5):

    def __init__(self, filename, data_key, params_key, shape, batch=1000, transform=None, shuffle=False):
        ''' Initialize a Dataset object
        Arguments
        ---------
        * filename  : name of the hdf5 file containing the data
        * batch     : maximum number of samples to be loaded in memory
        * data_key  : name of the dataset containing the actual data
        * params_key: name of the dataset containing the parameters
        * shape     : shape of the 2D images
        * transform : a transorm fucntion to be applied
        * shuffle   : whether the data should be shuffled during training (note that it might be slow)
        *             If set to true random batches of random parameter sets are sampled when iterating
        '''

        super().__init__(filename, data_key, params_key, shape, batch, transform)
        self._shuffle = shuffle

        # Load all params into memory
        with h5py.File(self._file, 'r') as fh5:
            self._params = np.array(fh5[self._params_key][:])

        # Precompute intervals of every parameter
        # Needed only if shuffle is True
        self._params_idx = dict()
        if self._shuffle:
            diff_params = self.get_different_params()
            for i in range(len(diff_params)):
                if i == 0:
                    self._params_idx[str(diff_params[i])] = self.get_params_range(diff_params[i], init=0)
                else:
                    self._params_idx[str(diff_params[i])] = self.get_params_range(diff_params[i], init=self._params_idx[str(diff_params[i - 1])][1])

    def get_different_params(self):
        return np.unique(self._params, axis=0)

    def get_params_range(self, params, init=0):

        # Check if there are precomputed values
        if str(params) in self._params_idx.keys():
            return self._params_idx[str(params)]

        # Compute from scratch
        start = -1
        end = -1
        for i in range(init, self._N):
            if start == -1 and str(self._params[i]) == str(params):
                start = i
            if start != -1 and end == -1 and not str(self._params[i]) == str(params):
                end = i
                break
        if start != -1 and end == -1:
            end = self._N
        end = end - 1
        return [start, end]

    def get_data_for_params(self, params, N=None):
        start, end = self.get_params_range(params)
        # assert(str(self._params[start]) == str(self._params[end]) and str(self._params[start]) != str(self._params[end + 1]))
        if start == -1 or end == -1:
            raise ValueError("Parameters not found")
        nsamples = end - start + 1
        if N is not None and nsamples > N:
            nsamples = N
        if nsamples > self._batch:
            print("WARNING: Can't load all the samples into memory. Returning " + str(self._batch) + " samples only")
            nsamples = self._batch
        data, params = self.load_batch(start, nsamples)
        data = data
        params = params
        return [data, params]

    def has_params(self, params):
        for p in self.get_different_params():
            if str(p) == str(params):
                return True
        return False

    # Get N elements from random parameter sets
    def get_random_data(self, N):

        # Estimated number of elements per param set
        assert(N <= self.N)
        N_set = N // len(self.get_different_params())

        # Get data
        data = []
        params = []
        old_shuffle = self._shuffle
        self._shuffle = True
        for batch in self.iter(N_set):
            for b in batch:
                data.append(b[0])
                params.append(b[1])
            if len(params) >= N:
                break
        self._shuffle = old_shuffle

        # Sort data
        data = np.array(data)
        params = np.array(params)
        perm = np.random.permutation(len(data))
        data = data[perm]
        params = params[perm]
        return data[:N], params[:N]
