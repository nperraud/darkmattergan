import itertools
import numpy as np
import tensorflow as tf
from gantools import utils, blocks
from . import path
from . import transformation, fmap
import functools, os
from gantools.utils import compose2


def do_nothing(x):
    return x



class DatasetMedical(object):
    ''' Dataset object for GAN and CosmoGAN classes

        Transform should probably be False for a classical GAN.
    '''

    def __init__(self, X, spix=32, shuffle=True, augmentation=True, transform=None, dtype=np.uint8):
        ''' Initialize a Dataset object

        Arguments
        ---------
        * X         : numpy array containing the data
        * shuffle   : True if the data should be shuffled
        * augmentation : Function to be applied ot each batch of the dataset
                      This allows extend the dataset.
        * slice_fn : Slicing function to cut the data into smaller parts
        '''
        X = X.astype(dtype)
        self._shuffle = shuffle
        # if slice_fn:
        # else:
        #     self._slice_fn = do_noting
        slice_fn = functools.partial(slice_shift_3d_patch, spix=spix)
        if augmentation:
            self._slice_fn = compose2(transformation.random_rotate_3d, slice_fn)
        else:
            self._slice_fn = slice_fn

        if transform:
            self._transform = transform
        else:
            self._transform = do_nothing
        self._spix = spix

        self._N = len(self._slice_fn(X))
        if shuffle:
            self._p = np.random.permutation(self._N)
        else:
            self._p = np.arange(self._N)
        self._X = X

    def get_all_data(self):
        ''' Return all the data (shuffled) '''
        return self._transform(slice_3d(self._X, self.spix)[self._p])

    def get_samples(self, N=100):
        ''' Get the `N` first samples '''
        return self._transform(self._slice_fn(self._X)[self._p[:N]])

    # TODO: kwargs to be removed
    def iter(self, batch_size=1, **kwargs):
        return self.__iter__(batch_size, **kwargs)

    # TODO: kwargs to be removed
    def __iter__(self, batch_size=1, **kwargs):

        if batch_size > self.N:
            raise ValueError(
                'Batch size greater than total number of samples available!')

        # Reshuffle the data
        if self.shuffle:
            self._p = np.random.permutation(self._N)
        nel = (self._N // batch_size) * batch_size
        transformed_data = self._slice_fn(self._X)[self._p[range(nel)]]
        for data in grouper(transformed_data, batch_size):
            yield self._transform(np.array(data))

    @property
    def shuffle(self):
        ''' Is the dataset suffled? '''
        return self._shuffle

    @property
    def N(self):
        ''' Number of element in the dataset '''
        return self._N

    @property
    def spix(self):
        return self._spix
    

def grouper(iterable, n, fillvalue=None):
    """
    Collect data into fixed-length chunks or blocks. This function commes
    from itertools
    """
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return itertools.zip_longest(fillvalue=fillvalue, *args)


def slice_shift_3d_patch(bbox, spix=32, limit_size=True):
    '''
    cubes: the 3d histograms - [:, :, :, :]
    '''

    # Handle the dimesnsions
    l = len(bbox.shape)
    if l < 3:
        ValueError('Not enough dimensions')
    elif l == 3:
        bbox = bbox.reshape([1, *bbox.shape]) # add one extra dimension for number of cubes
    elif l > 4:
        ValueError('To many dimensions')

    _, sx, sy, sz = bbox.shape
    nx = (sx // spix) - 1
    ny = (sy // spix) - 1
    nz = (sz // spix) - 1
    if limit_size:
        if nx>10:
            nx=10    
        if ny>10:
            ny=10
        if nz>10:
            nz=10
    lx = sx - nx*spix
    ly = sy - ny*spix
    lz = sz - nz*spix
    
    # 0) Select a subpart of the images
    rx = np.random.randint(0, lx)
    ry = np.random.randint(0, ly)
    rz = np.random.randint(0, lz)
    bbox = bbox[:, rx:rx+nx*spix, ry:ry+ny*spix, rz:rz+nz*spix]

    # 1) Create all 7 neighbors for each smaller cube
    img1 = np.roll(bbox, spix, axis=2)
    img1[:, :, :spix, :] = 0

    img2 = np.roll(bbox, spix, axis=3)
    img2[:, :, :, :spix] = 0
    
    img3 = np.roll(img1, spix, axis=3)
    img3[:, :, :, :spix] = 0
    
    img4 = np.roll(bbox, spix, axis=1) # extra for the 3D case
    img4[:, :spix, :, :] = 0
    
    img5 = np.roll(img4, spix, axis=2)
    img5[:, :, :spix, :] = 0
    
    img6 = np.roll(img4, spix, axis=3)
    img6[:, :, :, :spix] = 0
    
    img7 = np.roll(img5, spix, axis=3)
    img7[:, :, :, :spix] = 0

    # 2) Concatenate
    img_with_nbrs = np.stack([bbox, img1, img2, img3, img4, img5, img6, img7], axis=4) # 7 neighbors plus the original cube

    # Clear variable to gain some RAM
    del img1, img2, img3, img4, img5, img6, img7
    
    # 3) Slice the cubes
    img_with_nbrs = np.vstack(np.split(img_with_nbrs, nx, axis=1))
    img_with_nbrs = np.vstack(np.split(img_with_nbrs, ny, axis=2))
    img_with_nbrs = np.vstack(np.split(img_with_nbrs, nz, axis=3))
      
    return img_with_nbrs

def slice_3d(cubes, spix=64):
    '''
    slice each cube in cubes to smaller cubes,
    and return all the smaller cubes
    '''


    _, sx, sy, sz = cubes.shape
    nx = (sx // spix) - 1
    ny = (sy // spix) - 1
    nz = (sz // spix) - 1
    lx = sx - nx*spix
    ly = sy - ny*spix
    lz = sz - nz*spix
    
    # 0) Select a subpart of the images
    rx = np.random.randint(0, lx)
    ry = np.random.randint(0, ly)
    rz = np.random.randint(0, lz)
    cubes = cubes[:, rx:rx+nx*spix, ry:ry+ny*spix, rz:rz+nz*spix]

    # split along first dimension
    cubes = np.vstack(np.split(cubes, nx, axis=1))
    # split along second dimension
    cubes = np.vstack(np.split(cubes, ny, axis=2))
    # split along third dimension
    cubes = np.vstack(np.split(cubes, nz, axis=3))

    return cubes


# class Dataset_medical(object):
#     ''' Dataset oject for GAN and CosmoGAN classes

#         Transform should probably be False for a classical GAN.
#     '''

#     def __init__(self, resolution=256,
#             forward_map = None, scaling=1,
#             shuffle=True, slice_fn=None, transform=None):
#         ''' Initialize a Dataset object

#         Arguments
#         ---------
#         * resolution : [256, 512] (default 256)
#         * foward_map : foward mapping use None for raw data (default None)
#         * scaling : downscale the image by a factor (default 1)
#         * shuffle   : True if the data should be shuffled
#         * transform : Function to be applied ot each batch of the dataset
#                       This allows extend the dataset.
#         * slice_fn : Slicing function to cut the data into smaller parts
#         '''

#         self._resolution = resolution
#         self._forward_map = forward_map
#         self._scaling = scaling
#         if self._scaling>1:
#             self._scaling_sess = tf.Session()
    
#         self._shuffle = shuffle

#         if slice_fn:
#             self._slice_fn = slice_fn
#         else:
#             self._slice_fn = do_nothing

#         if transform:
#             self._transform = transform
#         else:
#             self._transform = do_nothing

#         self._hist_paths = self._get_hist_paths()
#         self._num_hists = len(self._hist_paths)
#         print("Number of histograms={}".format(self._num_hists))
#         self._hists = None
#         self._N = None
#         self._get_total_num_samples()

#     def _get_total_num_samples(self):
#         '''
#         Calculate total number of samples
#         '''
#         if self.N is None:
#             total_num_samples = 0
#             for batch_real in self.iter(name='_get_total_num_samples'):
#                 total_num_samples += 1

#             self._N = total_num_samples

#     def _get_hist_paths(self):
#         '''
#         Accumulate file paths of all histograms is a numpy array
#         '''
#         rootpath = path.medical_path()
#         #input_pattern = '{}_nbody_{}Mpc'.format(self._resolution)
#         file_ext = '.h5'
#         queue = []
#         for file in os.listdir(rootpath):
#             if file.endswith(file_ext): #and input_pattern in file:
#                 queue.append(os.path.join(rootpath, file))

#         if len(queue) == 0:
#             raise LookupError('No file founds, check path and parameters')

#         # shuffle file paths
#         if self.shuffle:
#             self._perm_hists = np.random.permutation(len(queue))
#         else:
#             self._perm_hists = np.arange(len(queue))

#         return np.array(queue)[self._perm_hists]

#     def _load_hists(self, num_hists_at_once=5, indices=None):

#         ''' 
#         Load 3d histograms, augment and scale them.
#         num_hists_at_once: number of histograms to be loaded at once in memory
#         indices: indices of the histograms to be loaded
#         '''

#         # 1) Load raw images
#         images = self._load_hists_raw(num_hists_at_once, indices)

#         # 2) Apply forward map if necessary
#         if self._forward_map:
#             images = self._forward_map(images)

#         return images 

#     def _load_hists_raw(self, num_hists_at_once, indices):
#         ''' 
#         Load 3D histograms
#         num_hists_at_once: number of histograms to be loaded at once in memory
#         indices: indices of the histograms to be loaded
#         '''

#         if indices is None:
#             indices = np.arange(num_hists_at_once)

#         raw_images = []
#         for file_path in self._hist_paths[indices]:
#             raw_image = utils.load_hdf5(filename=file_path, dataset_name='data', mode='r')
#             #mean = np.mean(raw_image)
#             #var = np.var(raw_image)
#             #raw_image = (raw_image-mean)/var # normalize
#             raw_images.append(raw_image)
#             if type(raw_images[-1]) is not np.ndarray:
#                 raise ValueError(
#                     "Data stored in file {} is not of type np.ndarray".format(
#                         file_path))

#         raw_images = np.array(raw_images).astype(np.float32)

#         return raw_images

#     def get_all_data(self):
#         ''' 
#         Return at max 10 histograms
#         '''

#         if self.num_hists > 10:
#             hists = self._load_hists(num_hists_at_once=10)
#         else:
#             hists = self._load_hists(num_hists_at_once=self.num_hists)

#         # Apply augmentation and slicing
#         samples = self._data_process(hists)
#         if self._shuffle:
#             perm_samples = np.random.permutation(len(samples))
#         else:
#             perm_samples = np.arange(len(samples))

#         return samples[perm_samples]

#     def get_samples(self, N=100):
#         ''' Get the first 'N' shuffled samples '''
#         return get_all_data()[:N]

#     def iter(self, batch_size=1, num_hists_at_once=5, name=None):
#         '''
#         Iterate through dataset
#         batch_size: number of samples to return at once
#         num_hists_at_once: number of histograms to be loaded at once in memory
#         '''

#         #if num_hists_at_once > 10:
#             #raise ValueError('Load less than 10 histograms at a time due to memory concerns!')

#         if num_hists_at_once > self.num_hists:
#             num_hists_at_once = self.num_hists

#         return self.__iter__(batch_size, num_hists_at_once, name)

#     def __iter__(self, batch_size, num_hists_at_once, name):

#         # Reshuffle the data
#         if self.shuffle:
#             perm_hists = np.random.permutation(self.num_hists)
#         else:
#             perm_hists = np.arange(self.num_hists)

#         # Load 'num_hists_at_once' histograms at a time
#         for i in range(0, self.num_hists, num_hists_at_once):
#             curr_inds = range(i, i+num_hists_at_once)
#             curr_inds = perm_hists[curr_inds]

#             if num_hists_at_once == self.num_hists:
#                 if self._hists is None: # Load all histograms only once in the beginning
#                     self._hists = self._load_hists(num_hists_at_once=None, indices=curr_inds)
#                     print("Loaded all {} histograms only once in the beginning!".format(num_hists_at_once))
                
#                 transformed_samples = self._data_process(self._hists)

#             else:
#                 hists = self._load_hists(num_hists_at_once=None, indices=curr_inds)
#                 transformed_samples = self._data_process(hists)
#                 print("Loaded {} histograms for iterator {}!".format(num_hists_at_once, name))
                
#             num_samples = len(transformed_samples)

#             if self.shuffle:
#                 perm_samples = np.random.permutation(num_samples)
#             else:
#                 perm_samples = np.arange(num_samples)

#             nel = (num_samples // batch_size) * batch_size
#             if nel == 0:
#                 print('batch_size={} greater than num_samples={} loaded at once. Resisizing batch_size to {}.'.format(batch_size, num_samples, num_samples))
#                 batch_size = num_samples
#                 nel = num_samples

#             transformed_samples = transformed_samples[perm_samples[range(nel)]]
#             for data in grouper(transformed_samples, batch_size):
#                 yield np.array(data)

#     def _data_process(self, hists):
#         samples = self._transform(hists)
        
#         # Apply downscaling if necessary
#         if self._scaling>1:
#             samples = blocks.downsample(samples, self._scaling, is_3d=True, sess=self._scaling_sess)

#         # if self._forward_map:
#         #     samples = self._forward_map(samples)

#         samples = self._slice_fn(samples)

#         return samples


#     @property
#     def shuffle(self):
#         ''' Is the dataset suffled? '''
#         return self._shuffle

#     @property
#     def N(self):
#         ''' Number of element in the dataset '''
#         return self._N

#     @property
#     def num_hists(self):
#         ''' Number of histograms in the dataset '''
#         return self._num_hists

#     @property
#     def resolution(self):
#         ''' Resolution of the bigger histogram'''
#         return self._resolution

#     @property
#     def scaling(self):
#         ''' Scaling down factor of the bigger histogram'''
#         return self._scaling

#     def get_big_dataset(self):
#         '''
#         Get the bigger dataset before it was sliced
#         using the 'self._slice_fn' function.
#         '''
#         _class = self.__class__ # get the appropriate derived class
#         return _class(resolution   =self._resolution, 
#                         forward_map=self._forward_map, 
#                         scaling    =self._scaling,
#                         spix       =self._resolution // self._scaling, #The bigger cube should not be sliced into smaller cubes
#                         shuffle    =self._shuffle, 
#                         transform  =self._transform)


# class Dataset_medical_3d(Dataset_medical):
#     def __init__(self, resolution=256,
#             forward_map = None, scaling=1,
#             spix=32, shuffle=True, transform=None):
#         ''' Initialize a Dataset object
#         Arguments
#         ---------
#         * resolution : [256, 512] (default 256)
#         * foward_map : foward mapping use None for raw data (default None)
#         * scaling : downscale the image by a factor (default 1)
#         * spix : resolution of the image (default 32)
#         * shuffle   : True if the data should be shuffled
#         * transform : Function to be applied ot each batch of the dataset
#                       This allows extend the dataset.
#         '''

#         slice_fn = functools.partial(slice_3d, spix=spix)
#         super().__init__(resolution=resolution,
#             forward_map = forward_map, scaling=scaling,
#             shuffle=shuffle, slice_fn=slice_fn, transform=transform)


# class Dataset_medical_3d_patch(Dataset_medical):
#     def __init__(self, resolution=256,
#             forward_map = None, scaling=1,
#             spix=32, shuffle=True, transform=None):
#         ''' Initialize a Dataset object for the 3d patch case
#         Arguments
#         ---------
#         * resolution : [256, 512] (default 256)
#         * foward_map : foward mapping use None for raw data (default None)
#         * scaling : downscale the image by a factor (default 1)
#         * spix : resolution of the image (default 32)
#         * shuffle   : True if the data should be shuffled
#         * transform : Function to be applied to each bigger cube in the dataset
#                       for data augmentation
#         '''

#         slice_fn = functools.partial(slice_3d_patch, spix=spix)
#         super().__init__(resolution=resolution,
#             forward_map = forward_map, scaling=scaling,
#             shuffle=shuffle, slice_fn=slice_fn, transform=transform)


# def grouper(iterable, n, fillvalue=None):
#     """
#     Collect data into fixed-length chunks or blocks. This function comes
#     from itertools
#     """
#     # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
#     args = [iter(iterable)] * n
#     return itertools.zip_longest(fillvalue=fillvalue, *args)



# def slice_3d(cubes, spix=64):
#     '''
#     slice each cube in cubes to smaller cubes,
#     and return all the smaller cubes
#     '''
#     num_slices = cubes.shape[1] // spix

#     # To ensure left over pixels in each dimension are ignored
#     limit = num_slices * spix
#     cubes = cubes[:, :limit, :limit, :limit]

#     # split along first dimension
#     cubes = np.vstack(np.split(cubes, num_slices, axis=1))
#     # split along second dimension
#     cubes = np.vstack(np.split(cubes, num_slices, axis=2))
#     # split along third dimension
#     cubes = np.vstack(np.split(cubes, num_slices, axis=3))

#     return cubes



# def slice_3d_patch(cubes, spix=32):
#     '''
#     cubes: the 3d histograms - [:, :, :, :]
#     '''

#     # Handle the dimesnsions
#     l = len(cubes.shape)
#     if l < 3:
#         ValueError('Not enough dimensions')
#     elif l == 3:
#         cubes = cubes.reshape([1, *cubes.shape]) # add one extra dimension for number of cubes
#     elif l > 4:
#         ValueError('To many dimensions')

#     _, sx, sy, sz = cubes.shape
#     nx = sx // spix
#     ny = sy // spix
#     nz = sz // spix

#     # 1) Create all 7 neighbors for each smaller cube
#     img1 = np.roll(cubes, spix, axis=2)
#     img1[:, :, :spix, :] = 0

#     img2 = np.roll(cubes, spix, axis=3)
#     img2[:, :, :, :spix] = 0
    
#     img3 = np.roll(img1, spix, axis=3)
#     img3[:, :, :, :spix] = 0
    
#     img4 = np.roll(cubes, spix, axis=1) # extra for the 3D case
#     img4[:, :spix, :, :] = 0
    
#     img5 = np.roll(img4, spix, axis=2)
#     img5[:, :, :spix, :] = 0
    
#     img6 = np.roll(img4, spix, axis=3)
#     img6[:, :, :, :spix] = 0
    
#     img7 = np.roll(img5, spix, axis=3)
#     img7[:, :, :, :spix] = 0
    

#     # 2) Concatenate
#     img_with_nbrs = np.stack([cubes, img1, img2, img3, img4, img5, img6, img7], axis=4) # 7 neighbors plus the original cube


#     # 3) Slice the cubes
#     img_with_nbrs = np.vstack(np.split(img_with_nbrs, nx, axis=1))
#     img_with_nbrs = np.vstack(np.split(img_with_nbrs, ny, axis=2))
#     img_with_nbrs = np.vstack(np.split(img_with_nbrs, nz, axis=3))

#     return img_with_nbrs