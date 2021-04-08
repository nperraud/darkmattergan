import sys
sys.path.insert(0, '../')

import h5py
import os
from cosmotools.utils import append_h5, require_dir, find_minmax_large, histogram_large, shuffle_h5
from cosmotools.data.load import load_params_dataset
from cosmotools.data.path import root_path_kids
import numpy as np

def get_params(filename):
    words = filename.split('_')
    return [float(words[3]), float(words[4])]

if __name__== "__main__":

    # 1. Set paths
    path_dataset = root_path_kids()
    require_dir(path_dataset)


    
    # 2. Make a big file with all the data  
    print("1. Make full dataset")
    folder_out = path_dataset
    fileout = os.path.join(folder_out,'kids.h5')
#     files = os.listdir(path_dataset)
#     files.sort()
#     first = True
#     for file in files:
#         if file[-4:]=='.npy':
#             maps = np.load(path_dataset + file) # Load maps
#             params = get_params(file) # Parse parameters
#             params = np.tile(np.array(params), [len(maps), 1])
#             append_h5(fileout, maps, params=params, overwrite=first)
#             first = False
    
    dataset = load_params_dataset('kids.h5', batch=12000, sorted=True, shape=[128, 128])
    assert(dataset.N==684000)
    diff_params = dataset.get_different_params()
    
    # 3. Divide into test and training set
    print("2. Divide train/testing set")
    
    test_params = [[0.137, 1.23],
                   [0.196, 1.225], # extr
                   [0.127, 0.836], # extr
                   [0.25, 0.658],
                   [0.311, 0.842],
                   [0.199, 0.87],
                   [0.254, 0.852],
                   [0.312, 0.664],
                   [0.356, 0.614],
                   [0.421, 0.628],
                   [0.487, 0.643]] # extr
    test_params = np.array(test_params)
    
    params_map = dict()
    for i in range(len(diff_params)):
        params_map[str(diff_params[i])] = i

    test_dic = dict()
    for p in test_params:
        if str(p) in params_map.keys():
            test_dic[params_map[str(p)]] = True


    test_params = []
    train_params = []
    for i in range(len(diff_params)):
        if i in test_dic.keys():
            test_params.append(diff_params[i])
        else:
            train_params.append(diff_params[i])
    test_params = np.array(test_params)
    train_params = np.array(train_params)

    with h5py.File(os.path.join(folder_out, 'train_test_params_kids.h5'), 'w') as f:
        f.create_dataset('train', data=train_params)
        f.create_dataset('test', data=test_params)

    dataset = load_params_dataset('kids.h5', batch=12000, sorted=True, shape=[128, 128])

    first = True
    for p in test_params:
        X, par = dataset.get_data_for_params(p)
        append_h5(os.path.join(folder_out,'kids_test.h5'), X, par, overwrite=first)
        first = False

    first = True
    for p in train_params:
        X, par = dataset.get_data_for_params(p)
        append_h5(os.path.join(folder_out, 'kids_train.h5'), X, par, overwrite=first)
        first = False

    # 4. Shuffle dataset
    print("3. Shuffle training set")

    shuffle_h5(os.path.join(folder_out,'kids_train.h5'), os.path.join(folder_out, 'kids_train_shuffled.h5'))
    
    # 5. Regressor
    print("4. Build dataset set for the regressor")    
    dataset = load_params_dataset('kids_train_shuffled.h5', batch=12000, shape=[128, 128])
    
    batch_size = 12000
    test_prob = 0.2
    
    train_file = os.path.join(folder_out,'kids_reg_train.h5')
    test_file = os.path.join(folder_out,'kids_reg_test.h5')
    
    first = True
    X_test = []
    p_test = []
    X_train = []
    p_train = []
    idx = 0
    for b in dataset:
        if np.random.rand() < test_prob:
            X_test.append(b[0, 0])
            p_test.append(b[0, 1])
        else:
            X_train.append(b[0, 0])
            p_train.append(b[0, 1])
        idx = idx + 1
        if idx % batch_size == 0:
            append_h5(test_file, np.array(X_test), np.array(p_test), overwrite=first)
            append_h5(train_file, np.array(X_train), np.array(p_train), overwrite=first)
            first = False
            X_test = []
            p_test = []
            X_train = []
            p_train = []
    if len(X_test) > 0:
        append_h5(test_file, np.array(X_test), np.array(p_test), overwrite=first)
    if len(X_train) > 0:
        append_h5(train_file, np.array(X_train), np.array(p_train), overwrite=first)
        
    print("=== All done! ===")    
