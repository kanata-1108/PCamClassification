import os
import h5py
import numpy as np

def h5_to_npy(h5_path, xy_flag, train_flag):

    if train_flag == True:
        if xy_flag == True:
            with h5py.File(h5_path, 'r') as f:
                # Camelyon2024.ipynbにあった[...]は、numpyにおいて全ての次元をスライスという意味。pythonの標準リストには使用できない
                x = f['x'][:50000]
                np.save(h5_path[:-3], x)

        else:
            with h5py.File(h5_path, 'r') as f:
                y = f['y'][:50000, :, 0, 0]
                np.save(h5_path[:-3], y)

    else:
        if xy_flag == True:
            with h5py.File(h5_path, 'r') as f:
                x = f['x']
                np.save(h5_path[:-3], x)
        
        else:
            with h5py.File(h5_path, 'r') as f:
                y = f['y'][:, :, 0, 0]
                np.save(h5_path[:-3], y)

if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))

    x_train_path = "./data/camelyonpatch_level_2_split_train_x.h5"
    y_train_path = "./data/camelyonpatch_level_2_split_train_y.h5"
    x_valid_path = "./data/camelyonpatch_level_2_split_valid_x.h5"
    y_valid_path = "./data/camelyonpatch_level_2_split_valid_y.h5"
    x_test_path = "./data/camelyonpatch_level_2_split_test_x.h5"

    h5_to_npy(h5_path = x_train_path, xy_flag = True, train_flag = True)
    h5_to_npy(h5_path = y_train_path, xy_flag = False, train_flag = True)
    h5_to_npy(h5_path = x_valid_path, xy_flag = True, train_flag = False)
    h5_to_npy(h5_path = y_valid_path, xy_flag = False, train_flag = False)
    h5_to_npy(h5_path = x_test_path, xy_flag = True, train_flag = False)