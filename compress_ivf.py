import faiss
import numpy as np
from pathlib import Path

# ----------------------------
# Utilities: alive bitset
# ----------------------------
class AliveMask:
    def __init__(self, size, packed=True, backing_file=None):
        self.size = size
        self.packed = packed
        if backing_file:
            self._arr = np.memmap(backing_file, mode='w+', dtype=np.uint8,
                                  shape=( (size + 7) // 8, )) if packed \
                        else np.memmap(backing_file, mode='w+', dtype=np.bool_, shape=(size,))
            self._arr[:] = 0xFF if packed else True
        else:
            self._arr = np.ones(( (size + 7) // 8, ), dtype=np.uint8) if packed \
                        else np.ones(size, dtype=np.bool_)

    def is_alive(self, ids: np.ndarray) -> np.ndarray:
        if self.packed:
            byte = ids // 8
            bit  = ids % 8
            return ((self._arr[byte] >> bit) & 1).astype(bool)
        else:
            return self._arr[ids]

    def kill(self, id_: int):
        if self.packed:
            self._arr[id_ // 8] &= ~(1 << (id_ % 8))
        else:
            self._arr[id_] = False

    def fraction_alive(self):
        if self.packed:
            # popcount
            return (np.unpackbits(self._arr).sum() / self.size)
        else:
            return float(self._arr.sum()) / self.size

# ----------------------------
# Min enclosing ball (Welzl)
# ----------------------------
def meb_welzl(P, R=None):
    # P: (n, d), R: list of boundary points (<= d+1)
    # returns (c, r)
    if R is None: R = []
    if len(P) == 0 or len(R) == P.shape[1] + 1:
        return _ball_from_boundary(np.array(R))
    p = P[-1]
    c, r = meb_welzl(P[:-1], R)
    if r is not None and np.linalg.norm(p - c) <= r + 1e-12:
        return c, r
    return meb_welzl(P[:-1], R + [p])

def _ball_from_boundary(B):
    # B: (k, d), k=0..d+1
    k = len(B)
    if k == 0:
        return np.zeros(0), 0.0
    if k == 1:
        return B[0], 0.0
    if k == 2:
        c = (B[0] + B[1]) / 2.0
        r = np.linalg.norm(B[0] - c)
        return c, r
    # Solve the unique sphere through up to d+1 points
    # (A)(c) = b  with A = 2*(p_i - p_0), b = ||p_i||^2 - ||p_0||^2
    p0 = B[0]
    A = 2.0 * (B[1:] - p0)
    b = (np.sum(B[1:]**2, axis=1) - np.sum(p0**2))
    c = np.linalg.lstsq(A, b, rcond=None)[0]
    c = c  # already coordinates in R^d
    # translate back: solve gave c in standard form; recover center:
    # Actually the above solves for center directly (see derivation).
    # If shapes mismatch, fall back to pseudo-inverse:
    if c.shape[0] != p0.shape[0]:
        c = np.linalg.pinv(A) @ b
    r = np.linalg.norm(B[0] - c)
    return c, r

def min_enclosing_ball(points, iters=3, seed=None):
    # Robustify Welzl with small shuffling and restarts
    rng = np.random.default_rng(seed)
    best = (None, np.inf)
    for _ in range(iters):
        P = points.copy()
        rng.shuffle(P)
        c, r = meb_welzl(P, [])
        if r < best[1]: best = (c, r)
    return best

# ----------------------------
# Exact diameter
# ----------------------------
def exact_diameter(X):
    # X: (N, d)
    diffs = X[:, None, :] - X[None, :, :]
    D2 = np.einsum('ijk,ijk->ij', diffs, diffs, optimize=True)
    i, j = np.unravel_index(np.argmax(D2), D2.shape)
    return float(np.sqrt(D2[i, j])), (i, j)

# ----------------------------
# FAISS session
# ----------------------------
class DenseSubsetFinder:
    def __init__(self, d_dim, nlist=80000, pq_M=32, pq_nbits=8,
                 use_opq=True, nprobe=128):
        self.d_dim = d_dim
        self.nprobe = nprobe
        # coarse quantizer
        self.coarse = faiss.IndexFlatL2(d_dim)
        self.index = faiss.IndexIVFPQ(self.coarse, d_dim, nlist, pq_M, pq_nbits)
        if use_opq:
            opq = faiss.OPQMatrix(d_dim, pq_M)
            self.pre = faiss.IndexPreTransform(opq, self.index)
            self.faiss_index = self.pre
        else:
            self.faiss_index = self.index
        # search params
        faiss.omp_set_num_threads(max(1, faiss.omp_get_max_threads()))
        self.index.nprobe = nprobe

    def train(self, sample):
        # sample: (n_train, d)
        self.faiss_index.train(sample.astype(np.float32))

    def add_with_ids(self, xb, ids):
        self.faiss_index.add_with_ids(xb.astype(np.float32), ids.astype(np.int64))

    def set_nprobe(self, nprobe):
        self.index.nprobe = nprobe

    def search(self, queries, k):
        D, I = self.faiss_index.search(queries.astype(np.float32), k)
        return D, I

# ----------------------------
# Iterative driver
# ----------------------------
class IterativeReducer:
    def __init__(self, memmap_path, vec_dim, index: DenseSubsetFinder,
                 alive: AliveMask, dtype=np.float32):
        self.X = np.memmap(memmap_path, mode='r', dtype=dtype, shape=(-1, vec_dim))
        self.index = index
        self.alive = alive
        self.d = self.X.shape[0]
        self.vec_dim = vec_dim

    def _filter_alive(self, ids):
        ids = ids[ids >= 0]
        mask = self.alive.is_alive(ids)
        return ids[mask]

    def nearest_alive_ids(self, center, target_N, k_factor=8, max_tries=3):
        # Oversample by k_factor, filter dead; expand if needed
        k = max(target_N * k_factor, target_N + 32)
        q = center.reshape(1, -1)
        nprobe0 = self.index.index.nprobe
        for t in range(max_tries):
            D, I = self.index.search(q, k)
            ids = self._filter_alive(I[0])
            if ids.size >= target_N:
                return ids[:target_N], ids  # (exact N, plus the soup)
            # expand search for this query only
            self.index.set_nprobe(min(4 * nprobe0, max(64, 2 * self.index.index.nprobe)))
            k = int(k * 1.5)
        # fallback: return whatever we have
        return ids, ids

    def refine_subset(self, seed_vec, N, soup_factor=5, meb_iters=2):
        # 1) Get a soup of alive candidates near the seed
        idsN, idsSoup = self.nearest_alive_ids(seed_vec, N, k_factor=soup_factor)
        if idsSoup.size == 0:
            return None
        soup = self.X[idsSoup]
        # 2) Initialize: take N nearest (by exact distance to seed)
        dists = np.linalg.norm(soup - seed_vec[None, :], axis=1)
        order = np.argsort(dists)
        take = order[:min(N, order.size)]
        S_ids = idsSoup[take]
        S = soup[take]

        center = seed_vec.copy()
        best = (np.inf, None, None, None)  # (radius, center, S_ids, S_points)

        for _ in range(max(1, meb_iters)):
            # MEB on current S
            c, r = min_enclosing_ball(S, iters=2)
            # Pull N nearest alive to c within soup (exact re-rank)
            dists = np.linalg.norm(soup - c[None, :], axis=1)
            order = np.argsort(dists)
            S_ids = idsSoup[order[:min(N, order.size)]]
            S = soup[order[:min(N, order.size)]]
            if r < best[0]:
                best = (r, c, S_ids.copy(), S.copy())
            center = c

        # Exact diameter on final S
        diam, (i, j) = exact_diameter(best[3])
        return {
            "radius": best[0],
            "center": best[1],
            "subset_ids": best[2],
            "subset_points": best[3],
            "diameter": diam,
        }

    def iterate(self, seeds, N, reduce1, max_iters=None, repack_trigger=0.6):
        """
        seeds: array of seed centers (k, d) or seed IDs to fetch centers from X
        reduce1: function(ids: np.ndarray) -> int  (returns the ID to remove)
        """
        it = 0
        results = []
        while True:
            if max_iters is not None and it >= max_iters:
                break
            # Choose a seed (you can cycle or keep a small heap by rN)
            for seed in seeds:
                seed_vec = self.X[seed] if np.issubdtype(type(seed), np.integer) else np.asarray(seed, np.float32)
                cand = self.refine_subset(seed_vec, N)
                if cand is None or cand["subset_ids"].size < N:
                    continue
                # Ask user policy to pick which ID to remove
                kill_id = reduce1(cand["subset_ids"])
                self.alive.kill(int(kill_id))

                results.append({
                    "iter": it,
                    "kill_id": int(kill_id),
                    "diameter": cand["diameter"],
                    "radius": cand["radius"],
                    "subset_ids": cand["subset_ids"],
                })
                it += 1
                # Periodic repack trigger (optional)
                if self.alive.fraction_alive() < repack_trigger:
                    print("Alive fraction low; consider rebuilding the index in the background.")
                break  # move to next iteration after 1 removal
            else:
                # No seed yielded a valid subset (exhausted or too many dead points nearby)
                print("No valid subset found with current seeds; try refreshing seeds or increasing nprobe/k.")
                break
        return results