import numpy as np
import matplotlib.pyplot as plt

def demo_2d(d=1000, k=2, seed=0, plot=False):
    """
    Demonstration of compress_moments_nd on 2D data:
      1. Generate `d` random points in R^2 (standard normal).
      2. Compress to at most binom(2+k, k) atoms matching moments up to order k.
      3. Plot original data (left) and compressed support (right), using marker area ∝ weight c_j.
      4. Print tensor moments M_l for l=0..k and the maximum absolute error.
    """

    # Step 1: generate data
    np.random.seed(seed)
    data = np.random.randn(d, 2)

    # Step 2: compress
    c_, w_ = compress(data, k, tol=1e-13)

    # Step 3: compute and print tensor moments up to order k
    def _compute_moment(arr, l):
        if l == 0:
            return arr.shape[0]
        d0, m0 = arr.shape
        M = np.zeros((m0,)*l)
        for w in arr:
            outer = w
            for _ in range(l-1):
                outer = np.multiply.outer(outer, w)
            M += outer
        return M
    
    print(f"\nMoments up to order {k}:")
    max_err = 0.0
    for l in range(k+1):
        M_o = _compute_moment(data, l)
        if l == 0:
            M_c = c_.sum()
        else:
            M_c = None
            for cj, wj in zip(c_, w_):
                outer = wj
                for _ in range(l-1):
                    outer = np.multiply.outer(outer, wj)
                if M_c is None:
                    M_c = cj * outer
                else:
                    M_c = M_c + cj * outer
        print(f"l={l}: \noriginal = {M_o}\ncompressed = {M_c}")
        err = abs(M_o - M_c) if l == 0 else np.max(np.abs(M_o - M_c))
        max_err = max(max_err, err)
    print("Max tensor-moment error:", max_err)

    # Step 4: plotting
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.scatter(data[:, 0], data[:, 1], s=5, alpha=0.6)
        ax1.set_title("Original data ({} points)".format(d))

        coords = w_
        weights = c_
        # marker area `s` equal to weight so radius ∝ sqrt(c_j)
        ax2.scatter(coords[:, 0], coords[:, 1], s=weights, alpha=0.6)
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
    


def demo_3d(d=500, k=2, seed=0):
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
    data = np.random.randn(d, 3)

    # Compress
    c_, w_ = compress(data, k)

    # Compute tensors and calculate error
    exps = _multi_exponents(3, k)
    moment_original = sum(_all_moments(data[j,:], exps) for j in range(d))
    moment_compressed = sum(c_[j]*_all_moments(w_[j,:], exps) for j in range(c_.size))
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
    demo_2d(d=1000, k=3, plot=True)
    # demo_3d(d=5000, k=3, seed=42)