# benchmarking two ways to compute null vectors. my sparse svd version has much smaller constant. But their scaling are the same. And python's arpack is not very stable. 

import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.linalg import null_space
from scipy.sparse.linalg import LinearOperator, eigsh

def null_vector(A, tol=1e-12, maxiter=1000, operator=False):
    """
    A: sparse (D, D+1) matrix
    returns: dense array of length D+1
    """
    if operator:
        # build the linear operator for M = A^T A
        n = A.shape[1]
        def matvec(x):
            return A.T.dot(A.dot(x))
        M = LinearOperator((n, n), matvec, dtype=A.dtype)
    else:
        M = A.T @ A

    eigvals, eigvecs = eigsh(M, k=1, sigma=-1., tol=tol, maxiter=maxiter)
    if eigvals[0] > tol:
        raise RuntimeError('No null vector found')
    return eigvecs[:, 0]

if __name__ == "__main__":
    np.random.seed(0)

    # choose a range of matrix sizes
    D_list = [100, 200, 400, 800, 1600, 3200, 6400]

    times_null_space = []
    times_eigsh      = []

    for D in D_list:
        A = np.random.randn(D, D+1)

        # --- method 1: scipy.linalg.null_space ---
        t0 = time.perf_counter()
        v1 = null_space(A, rcond=1e-12)[:, 0]
        t1 = time.perf_counter()
        times_null_space.append(t1 - t0)

        # --- method 2: eigsh on A^T A ---
        t0 = time.perf_counter()
        v2 = null_vector(A, tol=1e-12, maxiter=1000, operator=False)
        t1 = time.perf_counter()
        times_eigsh.append(t1 - t0)

        # verify correctness (optional)
        err1 = np.linalg.norm(A @ v1)
        err2 = np.linalg.norm(A @ v2)
        print(f"D={D:5d}  ‖A·v₁‖={err1:.2e},  ‖A·v₂‖={err2:.2e}")

    # --- plot scaling ---
    plt.figure(figsize=(6,4))
    plt.loglog(D_list, times_null_space, "o-", label="null_space")
    plt.loglog(D_list, times_eigsh,      "s-", label="eigsh")
    plt.xlabel("D")
    plt.ylabel("runtime (s)")
    plt.title("Null-vector computation vs. matrix size")
    plt.legend()
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.tight_layout()
    plt.show()