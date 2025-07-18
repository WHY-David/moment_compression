# only works for m=1 (1D data)

import numpy as np
# from scipy.linalg import svd 
import itertools
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.special import comb

def null_vector(A, tol=1e-14, maxiter=200, operator=False):
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


def reduce1(w_, c_, k=2, tol=1e-12, check=False):
    """
    data is a list of pairs [(c_j, w_j)]
    drop one of them and adjust the weights
    """
    d = len(c_)

    if check:
        m = 1
        Nmk = comb(m+k, m, exact=True)
        if d != Nmk+1:
            raise ValueError('len(data) must be comb(m+k,m)+1')
        if w_.shape[1] != d:
            raise ValueError('moments.shape[1] != len(weights)')

    # build (k+1)*(k+2) moment matrix A = [w^0; w^1; ...; w^k]
    A = np.vstack([w[subset]**l for l in range(k+1)])
    alpha = null_vector(A, tol=tol)

    # choose largest t>0 so lambda_[j] - t*alpha_j >= 0
    t = min((lambda_[j] / alpha_k)
            for alpha_k, j in zip(alpha, subset)
            if alpha_k > 0)

    # update weights and drop zeroed entries
    for alpha_k, j in zip(alpha, subset):
        lambda_[j] -= t * alpha_k
        if lambda_[j] <= tol:
            lambda_[j] = 0.0
            I.remove(j)




def reduce_moment_matching(data, k=1, tol=1e-12):
    """
    Given a sequence `data` of length d, returns a list of at most 3 pairs
    (c_j, w_j) with w_j in data and c_j > 0 such that
        sum_j c_j * (w_j)**k == (1/d) * sum_i data[i]**k * d   for k=0,1,2.
    Runs in O(d) time by iteratively reducing the support.
    """
    w = np.asarray(data, dtype=float)
    d = w.size

    # trivial if already small
    if d <= 3:
        # uniform mass 1 placed on each of up to 3 points
        return [(1.0, float(x)) for x in w]

    # initial uniform weights lambda_i summing to 1
    lambda_ = np.full(d, 1.0/d)
    I = set(range(d))  # active support indices

    # iteratively remove points until at most k+1 remain
    while len(I) > k+1:
        # pick any k+2 active indices
        subset = list(itertools.islice(I, k+2))
        # build (k+1)*(k+2) moment matrix A = [w^0; w^1; ...; w^k]
        A = np.vstack([w[subset]**l for l in range(k+1)])
        alpha = null_vector(A, tol=tol)

        # choose largest t>0 so lambda_[j] - t*alpha_j >= 0
        t = min((lambda_[j] / alpha_k)
                for alpha_k, j in zip(alpha, subset)
                if alpha_k > 0)

        # update weights and drop zeroed entries
        for alpha_k, j in zip(alpha, subset):
            lambda_[j] -= t * alpha_k
            if lambda_[j] <= tol:
                lambda_[j] = 0.0
                I.remove(j)

    # convert to (c_j, w_j) with c_j = lambda_j * d
    return [(float(lambda_[j] * d), float(w[j])) for j in I if lambda_[j] > tol]



if __name__ == '__main__':
    # generate data
    d = 10000
    data = np.random.randn(d)
    k = 3

    # original moments
    p = [float(np.mean(data**i)) for i in range(k+1)]

    # reduce
    pruned = reduce_moment_matching(data, k=k)
    c = np.array([cj for cj, wj in pruned])
    wj = np.array([wj for cj, wj in pruned])
    N = len(pruned)

    # pruned moments
    q = [float(np.sum(c * wj**i) / d) for i in range(k+1)]

    print(f"Original moments (p0 to p{k}):", p)
    print(f"\nPruned moments (q0 to q{k}):", q)
    print("\nPruned distribution (c_j, w_j):")
    for c, w in pruned:
        print(f"  (c = {c:.4f}, w = {w:.4f})")
