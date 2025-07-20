import numpy as np
import itertools
from scipy.linalg import null_space
import faiss


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


def multi_exponents(m, k):
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
def all_moments(w, exps):
    return np.array([np.prod(w**e) for e in exps], dtype=float)

def compress_naive(data, k, tol=1e-12):
    """
    Compress a dataset of d points in R^m down to at most
      binom(m+k, k)
    atoms matching moments up to order k.
    Naive version that chooses subset to be compressed by lexicographic order. 

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
    exps = multi_exponents(m, k)
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
            A[:, col] = all_moments(w[j], exps)

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



def compress(
    data,
    k,
    tol=1e-12,
    index_type='flat',          # 'ivf' or 'flat'
    nlist=None,
    nprobe=32,
    candidate_fraction=0.1,    # fraction of alive points used as candidate centers
    max_candidates=10000,
    overquery=5,               # extra neighbors to fetch beyond D+1
    refine=True,               # apply simple pruning refinement inside candidate cluster
    random_state=0,
    rebuild_fraction=0.30,     # retrain / rebuild when this fraction of ORIGINAL points removed
    verbose=False
):
    """
    Moment compression with diameter-aware Carathéodory peeling.

    Strategy:
      * Maintain a boolean 'alive' mask over the original points.
      * Build a FAISS index ONCE (train for IVF) on the full dataset (or Flat).
      * At each iteration:
          - Find a (D+1)-subset of alive indices with small diameter by:
              + Sampling candidate centers (subset of alive points).
              + For each, query k = D+1 + overquery neighbors.
              + Filter out dead ones; keep first (D+1) alive.
              + Optionally refine by pruning farthest-average points if we over-fetched.
              + Track subset with smallest diameter.
          - Form the D x (D+1) moment matrix of that subset.
          - Compute a null vector alpha and adjust weights (Carathéodory step) dropping at least one atom.
      * Rebuild the index only after every ~rebuild_fraction * original_d removals to avoid fragmentation.

    Parameters
    ----------
    data : (d,m) array
    k : int, max total degree of moments to match
    tol : float, weight drop tolerance
    index_type : {'ivf','flat'}
    nlist : int or None, IVF list count (auto if None)
    nprobe : int, IVF probe parameter
    candidate_fraction : float in (0,1], fraction of alive points used as candidate centers
    max_candidates : int, upper bound on candidate centers
    overquery : int, extra neighbors fetched beyond D+1
    refine : bool, whether to prune to D+1 by iterative removal (if we fetched more)
    random_state : int
    rebuild_fraction : float, fraction of ORIGINAL points removed between rebuilds
    verbose : bool
    """
    rng = np.random.default_rng(random_state)
    w_ = np.asarray(data, dtype=float)
    d, m = w_.shape

    # Build exponent list and feature dimension D
    exps = multi_exponents(m, k)
    D = len(exps)

    # Trivial case
    if d <= D:
        c_ = np.ones(d, dtype=float)
        return c_, w_.copy()

    # Initial weights (lambda_i sum to 1)
    lambda_ = np.full(d, 1.0 / d)
    alive = np.ones(d, dtype=bool)
    removed_total = 0
    next_rebuild_threshold = rebuild_fraction * d

    # ---- FAISS index helpers ----
    def build_index(points):
        """Build or rebuild FAISS index on ALL alive points (or full set at start)."""
        pts = points.astype(np.float32)
        if index_type == 'flat':
            idx = faiss.IndexFlatL2(m)
            idx.add(pts)
            return idx
        elif index_type == 'ivf':
            _nlist = nlist
            if _nlist is None:
                _nlist = max(32, int(min(4 * np.sqrt(points.shape[0]), 8192)))
            quant = faiss.IndexFlatL2(m)
            idx = faiss.IndexIVFFlat(quant, m, _nlist, faiss.METRIC_L2)
            # Training sample (up to 100k or all alive)
            train_sample = min(100_000, points.shape[0])
            sample_idx = rng.choice(points.shape[0], size=train_sample, replace=False)
            idx.train(pts[sample_idx])
            idx.add(pts)
            idx.nprobe = nprobe
            if verbose:
                print(f"[FAISS] Built IVF index: nlist={_nlist}, nprobe={nprobe}, alive={points.shape[0]}")
            return idx
        else:
            raise ValueError("index_type must be 'ivf' or 'flat'.")

    # Maintain mapping between FAISS internal order and original indices.
    # We rebuild index on ALL *alive* points in their current order when rebuilding.
    # We'll store an array 'alive_indices' giving original indices corresponding to FAISS IDs.
    alive_indices = np.arange(d)
    index = build_index(w_)

    def rebuild_if_needed():
        nonlocal index, alive_indices, removed_total, next_rebuild_threshold
        if removed_total >= next_rebuild_threshold:
            # Gather alive points
            alive_indices = np.where(alive)[0]
            index = build_index(w_[alive])
            # Set next threshold
            next_rebuild_threshold += rebuild_fraction * d

    # Diameter utilities
    def diameter(idx_subset):
        Y = w_[idx_subset]
        norms = np.sum(Y * Y, axis=1, keepdims=True)
        D2 = norms + norms.T - 2.0 * (Y @ Y.T)
        np.fill_diagonal(D2, 0.0)
        return float(np.sqrt(np.max(D2)))

    def refine_prune(indices, target_size):
        """Greedy remove point with largest average distance until target_size."""
        idxs = list(indices)
        while len(idxs) > target_size:
            Y = w_[idxs]
            norms = np.sum(Y * Y, axis=1, keepdims=True)
            D2 = norms + norms.T - 2.0 * (Y @ Y.T)
            np.fill_diagonal(D2, 0.0)
            avg = np.mean(np.sqrt(D2), axis=1)
            remove_pos = int(np.argmax(avg))
            del idxs[remove_pos]
        return np.array(idxs, dtype=int)

    # Candidate center selection produces a subset of *alive* original indices.
    def choose_candidate_centers():
        alive_idx = np.where(alive)[0]
        ccount = int(min(max(1, candidate_fraction * alive_idx.size), max_candidates))
        if ccount >= alive_idx.size:
            return alive_idx
        return rng.choice(alive_idx, size=ccount, replace=False)

    # Fetch neighbors for a given original index (center)
    def center_neighbors(center_orig_idx, need):
        """
        Return up to 'need' alive neighbor original indices INCLUDING the center.
        Uses a loop with growing k_query if dead points occupy slots.
        """
        if index_type == 'flat':
            # Flat index order is original order; we can query directly
            query_vec = w_[center_orig_idx].astype(np.float32)[None, :]
            k_query = min(w_.shape[0], need + overquery)
            while k_query <= w_.shape[0]:
                _, neigh = index.search(query_vec, k_query)
                # Map FAISS IDs to original indices (flat: same)
                neigh_ids = neigh[0]
                alive_neigh = [i for i in neigh_ids if alive[i]]
                if center_orig_idx not in alive_neigh:
                    alive_neigh.insert(0, center_orig_idx)
                # Uniquify while preserving order
                seen = set()
                filtered = []
                for v in alive_neigh:
                    if v not in seen:
                        seen.add(v)
                        filtered.append(v)
                    if len(filtered) >= need:
                        break
                if len(filtered) >= need or k_query == w_.shape[0]:
                    return np.array(filtered[:need], dtype=int)
                k_query = min(w_.shape[0], int(k_query * 1.5) + 1)
            return np.array(filtered, dtype=int)
        else:
            # IVF: current index is built only on alive points; FAISS IDs map through alive_indices
            query_vec = w_[center_orig_idx].astype(np.float32)[None, :]
            alive_count = alive_indices.size
            k_query = min(alive_count, need + overquery)
            while k_query <= alive_count:
                _, neigh = index.search(query_vec, k_query)
                mapped = alive_indices[neigh[0]]
                alive_neigh = [i for i in mapped if alive[i]]
                if center_orig_idx not in alive_neigh:
                    alive_neigh.insert(0, center_orig_idx)
                seen = set()
                filtered = []
                for v in alive_neigh:
                    if v not in seen:
                        seen.add(v)
                        filtered.append(v)
                    if len(filtered) >= need:
                        break
                if len(filtered) >= need or k_query == alive_count:
                    return np.array(filtered[:need], dtype=int)
                k_query = min(alive_count, int(k_query * 1.5) + 1)
            return np.array(filtered, dtype=int)

    # ---- Main iterative Carathéodory peeling ----
    active_count = np.count_nonzero(alive)
    iteration = 0
    while active_count > D:
        iteration += 1
        # Select a (D+1) subset with small diameter
        target_size = D + 1
        best_subset = None
        best_diam = np.inf

        candidates = choose_candidate_centers()
        for cidx in candidates:
            neigh = center_neighbors(cidx, target_size + overquery if refine else target_size)
            if neigh.size < target_size:
                continue
            if refine and neigh.size > target_size:
                neigh = refine_prune(neigh, target_size)
            diam_val = diameter(neigh)
            if diam_val < best_diam:
                best_diam = diam_val
                best_subset = neigh
        if best_subset is None:
            raise RuntimeError("Failed to find a viable subset of size D+1 among alive points.")

        # Build D x (D+1) moment matrix for best_subset
        A = np.empty((D, target_size), dtype=float)
        for col, j in enumerate(best_subset):
            A[:, col] = all_moments(w_[j], exps)

        # Null-space direction
        alpha = null_space(A, rcond=1e-12)[:, 0]
        if not np.any(alpha > 0):
            alpha = -alpha

        # Determine maximal step t
        t = min((lambda_[j] / a_j) for a_j, j in zip(alpha, best_subset) if a_j > 0)

        # Update weights; identify indices to drop (weight hits zero)
        dropped = []
        for a_j, j in zip(alpha, best_subset):
            lambda_[j] -= t * a_j
            if lambda_[j] <= tol:
                if alive[j]:
                    alive[j] = False
                    dropped.append(j)
                    lambda_[j] = 0.0

        removed_total += len(dropped)
        active_count -= len(dropped)

        if verbose:
            print(f"[Iter {iteration}] diameter={best_diam:.4e}, dropped={len(dropped)}, active={active_count}")

        # Rebuild if enough removals accumulated (only for IVF; Flat need not but we can for symmetry)
        if (index_type == 'ivf' or verbose) and removed_total >= next_rebuild_threshold:
            rebuild_if_needed()
            # After rebuild, alive_indices updated automatically

    # Collect final support
    final_idx = [j for j in range(d) if alive[j] and lambda_[j] > tol]
    c_ = np.array([lambda_[j] * d for j in final_idx], dtype=float)
    w_final = w_[final_idx, :].copy()
    return c_, w_final