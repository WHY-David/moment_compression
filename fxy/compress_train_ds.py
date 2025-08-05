import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from data_gen import generate_train_data

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from compressor import Compressor

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


def compress_train_ds(train_ds, return_list, k=1, tol=1e-12):
    # Compress with Compressor
    cp = Compressor(train_ds, tol=tol, index_type='flat')
    weight_dict = cp.compress_weights(k, return_at=return_list, overquery=0)

    # # Extract alive images from original dataset
    # all_images = train_ds.data.numpy().reshape(len(train_ds), -1)
    # alive_images = all_images[alive]

    # return weights[alive], alive_images
    return weight_dict


# If run as script, run compression and print result
if __name__ == '__main__':
    seed = 0
    d = 100_000
    k = 2
    print(f"k={k}")

    train_ds = generate_train_data(d, noise=0.5, seed=seed)

    lst = list(range(100, 1000, 100))
    return_list = lst + [10*d for d in lst] + [100*d for d in lst] + [1000*d for d in lst]
    weight_dict = compress_train_ds(train_ds, return_list, k=k)

    to_save = {f"d_{d}": c_ for d, c_ in weight_dict.items()}

    np.savez_compressed(f"weights_k{k}.npz", **to_save)