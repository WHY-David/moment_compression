import numpy as np
import faiss

from compressor_gpu import Compressor

def f(x, y):
    """Compute f(x, y) = exp(sin(pi * x) + y^2)."""
    return np.exp(np.sin(np.pi * x) + y**2)

def generate_train_data(size, func=f, noise=0., seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, size)
    y = rng.uniform(-1, 1, size)
    fv = func(x, y)
    noise_vals = rng.normal(loc=0.0, scale=noise, size=size)
    fv_noisy = fv + noise_vals
    return np.column_stack((x, y, fv_noisy))



def compress_train_ds(train_ds, return_list, k=1, tol=1e-12):
    # Compress with Compressor
    cp = Compressor(train_ds, tol=tol, index_type='flat')
    weight_dict = cp.compress_weights(k, return_at=return_list, max_candidates=10000, overquery=0, print_progress=True)

    # # Extract alive images from original dataset
    # all_images = train_ds.data.numpy().reshape(len(train_ds), -1)
    # alive_images = all_images[alive]

    # return weights[alive], alive_images
    return weight_dict


# If run as script, run compression and print result
if __name__ == '__main__':
    assert hasattr(faiss, 'StandardGpuResources'), "Install `faiss-gpu`, not CPU-only FAISS!"

    seed = 0
    d = 100_000
    k = 1
    print(f"k = {k}")

    train_ds = generate_train_data(d, noise=0.5, seed=seed)

    lst = list(range(100, 1000, 100))
    return_list = lst + [10*d for d in lst] + [100*d for d in lst] + [1000*d for d in lst]
    weight_dict = compress_train_ds(train_ds, return_list, k=k)

    to_save = {f"d_{d}": c_ for d, c_ in weight_dict.items()}

    np.savez_compressed(f"weights_k{k}.npz", **to_save)