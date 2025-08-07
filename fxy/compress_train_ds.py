import numpy as np
from data_gen import generate_train_data

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from compressor import Compressor

# set BLAS/OpenMP threads before libs import
p = (os.cpu_count() or 8) // 2 or 1
print(f"p = {p}")
os.environ.setdefault("OMP_NUM_THREADS", str(p))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(p))
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(p))
os.environ.setdefault("NUMEXPR_MAX_THREADS", str(p))
os.environ.setdefault("FAISS_NUM_THREADS", str(p))

import faiss
try: 
    faiss.omp_set_num_threads(p)
except Exception: 
    print("faiss.omp_set_num_threads(p) failed")



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
    k = 3
    print(f"k = {k}")

    train_ds = generate_train_data(d, noise=0.5, seed=seed)

    lst = list(range(100, 1000, 100))
    return_list = lst + [10*d for d in lst] + [100*d for d in lst] + [1000*d for d in lst]
    weight_dict = compress_train_ds(train_ds, return_list, k=k)

    to_save = {f"d_{d}": c_ for d, c_ in weight_dict.items()}

    np.savez_compressed(f"weights_k{k}.npz", **to_save)