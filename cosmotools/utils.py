import os
import sys
if sys.version_info[0] > 2:
    from urllib.request import urlretrieve
else:
    from urllib import urlretrieve
import hashlib
import zipfile
import h5py
import numpy as np

def printt(s=''):

    global PRINT_TEMP_STR_LEN
    try:
        PRINT_TEMP_STR_LEN
    except:
        PRINT_TEMP_STR_LEN=0
    print('\r'+PRINT_TEMP_STR_LEN*' '+'\r'+s, end='')
    PRINT_TEMP_STR_LEN=len(s)


def read_pickle(filename):

    import pickle
    with open(filename, 'rb') as f:
        return pickle.load(f)


def write_pickle(filename, obj):

    import pickle
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)
    print('wrote {}'.format(filename))

def require_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return None


def download(url, target_dir, filename=None):
    require_dir(target_dir)
    if filename is None:
        filename = url_filename(url)
    filepath = os.path.join(target_dir, filename)
    urlretrieve(url, filepath)
    return filepath


def url_filename(url):
    return url.split('/')[-1].split('#')[0].split('?')[0]


def check_md5(file_name, orginal_md5):
    # Open,close, read file and calculate MD5 on its contents
    with open(file_name, 'rb') as f:
        hasher = hashlib.md5()  # Make empty hasher to update piecemeal
        while True:
            block = f.read(64 * (
                1 << 20))  # Read 64 MB at a time; big, but not memory busting
            if not block:  # Reached EOF
                break
            hasher.update(block)  # Update with new block
    md5_returned = hasher.hexdigest()
    # Finally compare original MD5 with freshly calculated
    if orginal_md5 == md5_returned:
        print('MD5 verified.')
        return True
    else:
        print('MD5 verification failed!')
        return False


def unzip(file, targetdir):
    with zipfile.ZipFile(file, "r") as zip_ref:
        zip_ref.extractall(targetdir)


# Append to a h5 file
def append_h5(file, X, params=None, X_key="train_maps", params_key="train_labels", overwrite=False):
    if overwrite:
        X_shape = list(X.shape)
        X_shape[0] = None
        if params is not None:
            params_shape = list(params.shape)
            params_shape[0] = None
        with h5py.File(file, 'w') as f:
            f.create_dataset(X_key, data=X, maxshape=tuple(X_shape))
            if params is not None:
                f.create_dataset(params_key, data=params, maxshape=tuple(params_shape))
    else:
        with h5py.File(file, 'a') as f:
            f[X_key].resize((f[X_key].shape[0] + X.shape[0]), axis = 0)
            f[X_key][-X.shape[0]:] = X
            if params is not None:
                f[params_key].resize((f[params_key].shape[0] + params.shape[0]), axis = 0)
                f[params_key][-params.shape[0]:] = params
                
                
# Note: can be memory expensive
def shuffle_h5(filein, fileout, key_data="train_maps", key_params="train_labels"):

    # Load file
    with h5py.File(filein, 'r') as f:
        data = np.array(f[key_data][:])
        params = np.array(f[key_params][:])

    # Shuffle data
    perm = np.random.permutation(len(params))
    data = data[perm]
    params = params[perm]

    # Write file
    with h5py.File(fileout, 'w') as f:
        f.create_dataset(key_data, data=data)
        f.create_dataset(key_params, data=params)


# Compute the maximum and minimum of a big dataset
# Done in a memory efficient way
def find_minmax_large(dataset):
    vmin = None
    vmax = None
    for it in dataset:
        pixels = it[:, 0][0]
        curr_min = np.min(pixels)
        curr_max = np.max(pixels)
        if vmin is None or curr_min < vmin:
            vmin = curr_min
        if vmax is None or curr_max > vmax:
            vmax = curr_max
    return vmin, vmax


# Produce histogram of pixel intensities given a dataset
# This is performed in an efficient way in term of memory
def histogram_large(dataset, bins=100, lim=None, density=False):

    # Compute min and max
    if lim is None:
        lim = find_minmax(dataset)

    # Compute histograms batch by batch
    count = np.zeros(bins)
    for it in dataset:
        pixels = it[:, 0][0]
        histo, x = np.histogram(pixels, bins=bins, range=lim)
        count = count + histo
    if density:
        delta = x[1] - x[0]
        count = count / (delta * np.sum(count))
    return count, x


