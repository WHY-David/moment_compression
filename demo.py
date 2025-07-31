import numpy as np
import matplotlib.pyplot as plt

# from moment_matching import compress, compress_naive, multi_exponents, all_moments
from compressor import Compressor, multi_exponents, all_moments

def demo_2d(d=1000, k=2, seed=0, **kwargs):
    """
    Demonstration of compress_moments_nd on 2D data:
      1. Generate `d` random points in R^2 (standard normal).
      2. Compress to at most binom(2+k, k) atoms matching moments up to order k.
      3. Plot original data (left) and compressed support (right), using marker area ∝ weight c_j.
      4. Print tensor moments M_l for l=0..k and the maximum absolute error.
    """

    # generate data
    np.random.seed(seed)
    data = np.random.randn(d, 2)

    # compress
    compressor = Compressor(data, tol=1e-12, index_type='flat')
    c_, w_ = compressor.compress(k, **kwargs)

    # Compute tensors and calculate error
    exps = multi_exponents(2, k)
    moment_original = sum(all_moments(data[j,:], exps) for j in range(d))
    moment_compressed = sum(c_[j]*all_moments(w_[j,:], exps) for j in range(c_.size))
    max_err = np.max(np.abs(moment_original - moment_compressed))

    # plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.scatter(data[:, 0], data[:, 1], s=5, alpha=0.6)
    ax1.set_title("Original data ({} points)".format(d))

    coords = w_
    weights = c_
    # marker area `s` equal to weight so radius ∝ sqrt(c_j)
    ax2.scatter(coords[:, 0], coords[:, 1], s=weights*10, alpha=0.6)
    ax2.set_title(f"Compressed to {c_.size} atoms, moment error = {max_err:.1e}")

    # enforce same axis ranges on both subplots
    x_all = np.concatenate([data[:, 0], coords[:, 0]])
    y_all = np.concatenate([data[:, 1], coords[:, 1]])
    x_min, x_max = x_all.min(), x_all.max()
    y_min, y_max = y_all.min(), y_all.max()
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.show()
    


def demo_3d(d=500, k=2, seed=0, **kwargs):
    """
    Demonstration of compress_moments on 3D data:
      1. Generate `d` random points in R^3 (standard normal).
      2. Compress to at most binom(3+k, k) atoms matching moments up to order k.
      3. Plot original data (left) and compressed support (right) as 3D scatter plots,
         using marker area ∝ weight c_j.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # Generate data
    np.random.seed(seed)
    data = np.random.rand(d, 3)

    # Compress
    compressor = Compressor(data, tol=1e-12)
    c_, w_ = compressor.compress(k, **kwargs)

    # Compute tensors and calculate error
    exps = multi_exponents(3, k)
    moment_original = sum(all_moments(data[j,:], exps) for j in range(d))
    moment_compressed = sum(c_[j]*all_moments(w_[j,:], exps) for j in range(c_.size))
    max_err = np.max(np.abs(moment_original - moment_compressed))

    # Plotting
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(data[:, 0], data[:, 1], data[:, 2], s=5, alpha=0.6)
    ax1.set_title(f"Original 3D data ({d} points)")

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(w_[:, 0], w_[:, 1], w_[:, 2], s=c_, alpha=0.6)
    ax2.set_title(f"Compressed to {c_.size} atoms. k={k}, error={max_err:.1e}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # run demo with default parameters
    demo_2d(d=100000, k=2, dstop=1000, verbose=True)
    # demo_3d(d=1000, k=2, seed=0)