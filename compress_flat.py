import faiss
import numpy as np

# ----------------------------
# Alive bitset (packed)
# ----------------------------
class AliveMask:
    def __init__(self, size, packed=True):
        self.size = size
        self.packed = packed
        self.arr = np.ones(((size + 7) // 8,), dtype=np.uint8) if packed \
                   else np.ones(size, dtype=np.bool_)

    def is_alive(self, ids: np.ndarray) -> np.ndarray:
        ids = ids.astype(np.int64)
        if self.packed:
            byte = ids // 8
            bit  = ids % 8
            return ((self.arr[byte] >> bit) & 1).astype(bool)
        else:
            return self.arr[ids]

    def kill(self, id_: int):
        if self.packed:
            self.arr[id_ // 8] &= ~(1 << (id_ % 8))
        else:
            self.arr[id_] = False

# ----------------------------
# Exact diameter (for N<=200)
# ----------------------------
def exact_diameter(X):
    # X: (N, m)
    diffs = X[:, None, :] - X[None, :, :]
    D2 = np.einsum('ijk,ijk->ij', diffs, diffs, optimize=True)
    i, j = np.unravel_index(np.argmax(D2), D2.shape)
    return float(np.sqrt(D2[i, j])), (i, j)

# ----------------------------
# Cluster refine (Flat index)
# ----------------------------
class FlatFinder:
    def __init__(self, xb: np.ndarray, ids: np.ndarray):
        """
        xb: (d, m) float32 vectors
        ids: (d,) int64 ids in [0..d-1]
        """
        assert xb.dtype == np.float32
        self.xb = xb
        self.ids = ids.astype(np.int64)
        self.d, self.m = xb.shape

        base = faiss.IndexFlatL2(self.m)             # exact L2
        self.index = faiss.IndexIDMap2(base)         # keep external ids
        self.index.add_with_ids(self.xb, self.ids)   # add once

        faiss.omp_set_num_threads(max(1, faiss.omp_get_max_threads()))

    def search_alive(self, q: np.ndarray, want: int, alive: AliveMask,
                     k_factor=8, max_tries=3):
        """
        q: (m,) query vector
        want: target alive neighbors to return (>= N)
        """
        q = q.reshape(1, -1).astype(np.float32)
        k = max(want * k_factor, want + 32)
        for _ in range(max_tries):
            D, I = self.index.search(q, k)
            I = I[0]
            I = I[I >= 0]
            mask = alive.is_alive(I)
            alive_ids = I[mask]
            if alive_ids.size >= want:
                return alive_ids[:want], alive_ids  # exact N, plus a soup
            k = int(k * 1.5)  # oversample more if needed
        return alive_ids, alive_ids  # whatever we could get

    def refine_near_seed(self, seed_vec: np.ndarray, N: int, alive: AliveMask,
                         soup_factor=5, iters=3):
        """
        1) pull ~5N exact neighbors (post-filtered alive)
        2) select N nearest to current center
        3) recenter (mean) and repeat
        """
        idsN, idsSoup = self.search_alive(seed_vec, N, alive, k_factor=soup_factor)
        if idsSoup.size < N:
            return None  # not enough alive points nearby

        soup = self.xb[idsSoup]
        center = seed_vec.astype(np.float32)

        best = (np.inf, None, None)
        for _ in range(max(1, iters)):
            # select N nearest to the current center (exact distances in RAM)
            dists = np.linalg.norm(soup - center[None, :], axis=1)
            order = np.argsort(dists)
            take = order[:N]
            S_ids = idsSoup[take]
            S = soup[take]

            # compute diameter (exact)
            diam, (i, j) = exact_diameter(S)
            if diam < best[0]:
                best = (diam, S_ids.copy(), S.copy())

            # recenter (mean). For tighter balls, replace with MEB center.
            center = S.mean(axis=0).astype(np.float32)

        return {"diameter": best[0], "subset_ids": best[1], "subset_points": best[2]}

# ----------------------------
# Example driver
# ----------------------------
def example():
    # Toy data (replace with your mmapped array)
    d, m, N = 50000, 10, 100
    rng = np.random.default_rng(0)
    xb = rng.standard_normal((d, m), dtype=np.float32)
    ids = np.arange(d, dtype=np.int64)

    alive = AliveMask(d, packed=True)
    finder = FlatFinder(xb, ids)

    # Make some seeds: a few random ids or any centroids you like
    seed_ids = rng.choice(d, size=100, replace=False)
    seed_vecs = xb[seed_ids]

    def reduce1(subset_ids: np.ndarray) -> int:
        # Example policy: remove one endpoint of current diameter
        S = xb[subset_ids]
        diam, (i, j) = exact_diameter(S)
        return int(subset_ids[i])  # your custom choice

    results = []
    # iterate a few times: find N-set → reduce1 → mark dead
    for t in range(10000):
        best = None
        for s in seed_vecs[:20]:  # try a handful of seeds each round
            cand = finder.refine_near_seed(s, N, alive, soup_factor=5, iters=3)
            if cand is None: 
                continue
            if (best is None) or (cand["diameter"] < best["diameter"]):
                best = cand
        if best is None:
            print("No valid subset found; broaden seeds or increase k_factor.")
            break
        kill_id = reduce1(best["subset_ids"])
        alive.kill(kill_id)
        if t%500 ==0:
            results.append({"iter": t, "kill_id": kill_id, "diameter": best["diameter"]})
            print(f"[iter {t}] kill {kill_id}, diameter={best['diameter']:.4f}")

    return results

if __name__ == "__main__":
    example()