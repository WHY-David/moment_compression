import numpy as np
import itertools
import matplotlib.pyplot as plt
# from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.linalg import null_space


# def null_vector(A, tol=1e-12, maxiter=1000, operator=False):
#     """
#     A: sparse (D, D+1) matrix
#     returns: dense array of length D+1
#     """
#     if operator:
#         # build the linear operator for M = A^T A
#         n = A.shape[1]
#         def matvec(x):
#             return A.T.dot(A.dot(x))
#         M = LinearOperator((n, n), matvec, dtype=A.dtype)
#     else:
#         M = A.T @ A

#     eigvals, eigvecs = eigsh(M, k=1, sigma=-1., tol=tol, maxiter=maxiter)
#     if eigvals[0] > tol:
#         raise RuntimeError('No null vector found')
#     return eigvecs[:, 0]


def _multi_exponents(m, k):
    """
    Generate all exponent-tuples e = (e_0,...,e_{m-1}) of nonnegative
    integers with sum(e) <= k, ordered by increasing total degree.
    The total count is binom(m+k, k).
    """
    exps = []
    def gen(curr, remaining, idx):
        if idx == m:
            if remaining == 0:
                exps.append(tuple(curr))
            return
        if idx == m-1:
            # last coordinate must absorb all remaining
            curr.append(remaining)
            gen(curr, 0, idx+1)
            curr.pop()
        else:
            for take in range(remaining+1):
                curr.append(take)
                gen(curr, remaining - take, idx+1)
                curr.pop()
    for total in range(k+1):
        gen([], total, 0)
    return exps

# compute the moment‐feature vector of a single point
def _all_moments(w, exps):
    return np.array([np.prod(w**e) for e in exps], dtype=float)


def compress_moments(data, k, tol=1e-12):
    """
    Compress a dataset of d points in R^m down to at most
      binom(m+k, k)
    atoms matching moments up to order k.

    Parameters
    ----------
    data : array_like, shape (d, m)
        Original samples w_i in R^m.
    k : int
        Maximum moment order to match.
    tol : float
        Threshold for dropping tiny weights.

    Returns
    -------
    pruned : list of (c_j, w_j)
        c_j > 0 and w_j is one of the original data points.  The
        list length N <= binom(m+k, k) and
          sum_j c_j * (w_j)**l  = sum_i (w_i)**l
        (interpreted componentwise in the tensor sense) for l=0,…,k.
        Here (l=0) enforces sum_j c_j = d.
    """
    w = np.asarray(data, dtype=float)
    if w.ndim != 2:
        raise ValueError("`data` must be a 2D array of shape (d, m)")
    d, m = w.shape

    # build exponent list and feature dimension D
    exps = _multi_exponents(m, k)
    D = len(exps)                # = binom(m+k, k)

    # trivial if already small
    if d <= D:
        # trivial case: exactly d atoms with weight 1 each
        c_ = np.ones(d, dtype=float)
        w_ = w.copy()
        return c_, w_

    # initial uniform weights lambda_i summing to 1
    lambda_ = np.full(d, 1.0/d)
    I = set(range(d))  # active support indices

    # iteratively peel off points until support ≤ D
    while len(I) > D:
        # take any D+1 active indices
        subset = list(itertools.islice(I, D+1))

        # build the D×(D+1) moment matrix A
        A = np.empty((D, D+1), dtype=float)
        for col, j in enumerate(subset):
            A[:, col] = _all_moments(w[j], exps)

        alpha = null_space(A, rcond=1e-12)[:, 0]
        # ensure alpha has some positive entries; if not, flip its sign
        if not np.any(alpha > 0):
            alpha = -alpha

        # find the largest step t so lam[j] - t*α_j ≥ 0 for all j
        # (at least one will hit zero)
        t = min((lambda_[j]/aj) for aj, j in zip(alpha, subset) if aj > 0)

        # move weights and drop zeros
        for aj, j in zip(alpha, subset):
            lambda_[j] -= t * aj
            if lambda_[j] <= tol:
                lambda_[j] = 0.0
                I.remove(j)

    # build final arrays of weights and support points
    idx = [j for j in sorted(I) if lambda_[j] > tol]
    c_ = np.array([lambda_[j] * d for j in idx], dtype=float)
    w_ = w[idx, :].copy()
    return c_, w_


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
    c_, w_ = compress_moments(data, k, tol=1e-13)

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
    c_, w_ = compress_moments(data, k)

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
    # demo_2d(d=10000, k=8, plot=True)
    demo_3d(d=5000, k=4, seed=42)