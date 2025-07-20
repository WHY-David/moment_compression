import faiss
import numpy as np

def min_diameter_cluster_faiss(points, N):
    d, m = points.shape
    index = faiss.IndexFlatL2(m)  # Exact L2; swap for approximate if needed
    index.add(points)
    _, indices = index.search(points, N)
    min_diam = np.inf
    min_cluster = None
    for i in range(d):
        cluster = points[indices[i]]
        dist_matrix = np.linalg.norm(cluster[:, None, :] - cluster[None, :, :], axis=-1)
        diam = np.max(dist_matrix)
        if diam < min_diam:
            min_diam = diam
            min_cluster = indices[i]
    return min_cluster, min_diam

def min_diameter_subset_faiss(
    X,
    N,
    reduced_dim=None,
    use_pca=True,
    candidate_fraction=0.1,
    max_candidates=10000,
    index_type='ivf',          # 'flat' or 'ivf'
    nlist=None,
    nprobe=16,
    refine_extra=0,
    random_state=0,
    verbose=False
):
    """
    Approximate minimum-diameter subset of size N using FAISS kNN retrieval.

    Parameters
    ----------
    X : ndarray, shape (d, m)
        Data matrix (float32 or convertible).
    N : int
        Target subset size (2 <= N).
    reduced_dim : int or None
        If not None and reduced_dim < m, perform dimensionality reduction
        (PCA if use_pca else random Gaussian projection) for neighbor search only.
    use_pca : bool
        Use PCA (True) or random projection (False) when reducing dimension.
    candidate_fraction : float in (0,1]
        Fraction of points to treat as candidate centers.
    max_candidates : int
        Upper bound on number of candidate centers.
    index_type : {'flat', 'ivf'}
        FAISS index backend.
    nlist : int or None
        Number of IVF lists (ignored if index_type='flat'). If None, auto-set.
    nprobe : int
        IVF probe parameter (trade accuracy/speed).
    refine_extra : int
        Retrieve N + refine_extra neighbors and then pick best N-subset
        by pruning farthest points iteratively (simple refinement).
    random_state : int
        RNG seed for reproducibility.
    verbose : bool
        Print progress information.

    Returns
    -------
    best_indices : ndarray, shape (N,)
        Indices (w.r.t original X) of selected subset.
    best_diameter : float
        Diameter (Euclidean) of that subset in original space.
    info : dict
        Auxiliary info: {'evaluated_centers': int, 'reduced_dim': int or None}.
    """
    rng = np.random.default_rng(random_state)
    X = np.asarray(X)
    d, m = X.shape
    if N > d:
        raise ValueError("N cannot exceed number of points.")
    if X.dtype != np.float32:
        Xf = X.astype(np.float32)
    else:
        Xf = X

    # ---- Dimensionality Reduction (for search space) ----
    if reduced_dim is not None and reduced_dim < m:
        if use_pca:
            # PCA via FAISS
            mat = faiss.PCAMatrix(m, reduced_dim)
            mat.train(Xf)
            Xr = mat.apply_py(Xf)
        else:
            # Random Gaussian projection
            G = rng.normal(size=(m, reduced_dim)).astype(np.float32)
            G /= np.sqrt(np.sum(G**2, axis=0, keepdims=True) + 1e-12)
            Xr = Xf @ G
        search_vectors = Xr
        search_dim = search_vectors.shape[1]
    else:
        search_vectors = Xf
        search_dim = m
        reduced_dim = None  # normalize output field

    # ---- Build FAISS Index ----
    if index_type == 'flat':
        index = faiss.IndexFlatL2(search_dim)
        index.add(search_vectors)
    elif index_type == 'ivf':
        if nlist is None:
            nlist = int(min(4 * np.sqrt(d), 8192))
            nlist = max(nlist, 32)
        quantizer = faiss.IndexFlatL2(search_dim)
        index = faiss.IndexIVFFlat(quantizer, search_dim, nlist, faiss.METRIC_L2)
        # Train
        # Sample for training if very large
        train_sample = min(100_000, d)
        sample_idx = rng.choice(d, size=train_sample, replace=False)
        index.train(search_vectors[sample_idx])
        index.add(search_vectors)
        index.nprobe = nprobe
    else:
        raise ValueError("Unsupported index_type.")

    # ---- Select candidate centers ----
    cand_count = int(min(max(1, candidate_fraction * d), max_candidates))
    if cand_count == d:
        candidate_centers = np.arange(d)
    else:
        candidate_centers = rng.choice(d, size=cand_count, replace=False)

    # ---- Helper: diameter computation ----
    def diameter_indices(idxs):
        # compute pairwise distances (N x N) efficiently
        Y = Xf[idxs]  # (k, m)
        # Use (y_i - y_j)^2 = ||y_i||^2 + ||y_j||^2 - 2 y_i·y_j
        norms = np.sum(Y * Y, axis=1, keepdims=True)
        D2 = norms + norms.T - 2.0 * (Y @ Y.T)
        np.fill_diagonal(D2, 0.0)
        return float(np.sqrt(np.max(D2)))

    # ---- Optional refinement: pick best N out of N+R by greedy pruning ----
    def refine_subset(candidates, target_size):
        # candidates: indices length M >= target_size
        # Greedy: while len > target_size, remove the point with largest distance to its nearest neighbor cluster tightening heuristic
        idxs = list(candidates)
        while len(idxs) > target_size:
            Y = Xf[idxs]
            norms = np.sum(Y * Y, axis=1, keepdims=True)
            D2 = norms + norms.T - 2.0 * (Y @ Y.T)
            np.fill_diagonal(D2, np.inf)
            # For each point, record its *farthest* distance inside cluster (or could use max-min)
            # Strategy: remove point with largest *average* distance (alternative: largest max distance)
            avg_dist = np.mean(np.sqrt(D2), axis=1)
            remove_pos = int(np.argmax(avg_dist))
            del idxs[remove_pos]
        return np.array(idxs, dtype=int)

    best_diameter = np.inf
    best_indices = None

    # Batch query: we only need the (N + refine_extra) nearest for each center
    k_query = N + refine_extra
    if k_query > d:
        k_query = d

    # Prepare center queries
    centers_vecs = search_vectors[candidate_centers]
    # FAISS search
    _, neigh_idx = index.search(centers_vecs, k_query)

    for cpos, center in enumerate(candidate_centers):
        nn = neigh_idx[cpos]  # includes the center itself
        if refine_extra > 0 and len(nn) > N:
            subset = refine_subset(nn, N)
        else:
            subset = nn[:N]
        # Ensure uniqueness
        subset = np.unique(subset)
        if subset.shape[0] < N:
            # If duplicates due to extremely small dataset or something odd, skip
            continue
        diam = diameter_indices(subset)
        if diam < best_diameter:
            best_diameter = diam
            best_indices = subset
        if verbose and (cpos+1) % max(1, cand_count // 10) == 0:
            print(f"[{cpos+1}/{cand_count}] current best diameter = {best_diameter:.6f}")

    info = {
        'evaluated_centers': int(len(candidate_centers)),
        'reduced_dim': reduced_dim,
        'index_type': index_type,
        'nlist': nlist if index_type == 'ivf' else None,
        'nprobe': nprobe if index_type == 'ivf' else None
    }

    return best_indices, best_diameter, info



X = np.random.randn(100_000, 1).astype(np.float32)
subset_idx, diam, info = min_diameter_subset_faiss(
    X, N=10,
    reduced_dim=64,
    candidate_fraction=0.1,
    refine_extra=5,
    index_type='flat',
    nprobe=32,
    verbose=True
)
print("Subset indices:", subset_idx)
print("Diameter:", diam)
print("Info:", info)