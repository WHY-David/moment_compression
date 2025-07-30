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

    def count_alive(self) -> int:
        """Return the number of alive ids."""
        if self.packed:
            if self.arr.size == 0:
                return 0
            bits = np.unpackbits(self.arr)[: self.size]
            return int(bits.sum())
        else:
            return int(self.arr.sum())

    def alive_fraction(self) -> float:
        """Return fraction of alive ids in [0,1]."""
        cnt = self.count_alive()
        return cnt / float(self.size) if self.size > 0 else 0.0

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
# Utilities: sample alive ids
# ----------------------------
def sample_alive_ids(alive: AliveMask, n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Try to sample n alive ids efficiently. First do a few rounds of rejection
    sampling; if not enough, fall back to a full bit-unpack once.
    Returns an array of unique ids (length <= n if fewer alive).
    """
    out = []
    target = int(n)
    # Fast rejection sampling first
    tries = 0
    max_tries = max(10 * target, 100)
    while len(out) < target and tries < max_tries:
        batch = min(target - len(out), 4096)
        cand = rng.integers(0, alive.size, size=batch, endpoint=False)
        mask = alive.is_alive(cand)
        if mask.any():
            out.extend(list(cand[mask]))
        tries += 1
    # If still short, slow path: unpack all bits once
    if len(out) < target:
        bits = np.unpackbits(alive.arr)[: alive.size]
        ids_all = np.flatnonzero(bits)
        if ids_all.size == 0:
            return np.empty((0,), dtype=np.int64)
        rng.shuffle(ids_all)
        need = target - len(out)
        out.extend(list(ids_all[:need]))
    # Dedup and trim
    if not out:
        return np.empty((0,), dtype=np.int64)
    out = np.array(out, dtype=np.int64)
    if out.size > target:
        out = out[:target]
    return out

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
                     k_factor=8, max_tries=5, max_k: int | None = None):
        """
        q: (m,) query vector
        want: target alive neighbors to return (>= N)
        Strategy: geometrically increase k; as a last resort, set k to the
        maximum allowed (all points). Returns (top_alive[:want], all_alive_in_k).
        """
        q = q.reshape(1, -1).astype(np.float32)
        if max_k is None:
            max_k = self.d
        k = max(int(want * k_factor), want + 32)
        k = min(k, max_k)
        alive_ids = np.empty((0,), dtype=np.int64)
        for _ in range(max_tries):
            D, I = self.index.search(q, k)
            I = I[0]
            I = I[I >= 0]
            mask = alive.is_alive(I)
            alive_ids = I[mask]
            if alive_ids.size >= want:
                return alive_ids[:want], alive_ids
            if k >= max_k:
                break
            # grow k geometrically up to max_k
            k = min(max_k, int(k * 1.5) + 1)
        # Final attempt: if we haven't reached max_k yet, pull max_k exactly once
        if k < max_k:
            D, I = self.index.search(q, max_k)
            I = I[0]
            I = I[I >= 0]
            mask = alive.is_alive(I)
            alive_ids = I[mask]
        # Return whatever we have (may be < want) so caller can infer termination
        return alive_ids[:min(want, alive_ids.size)], alive_ids

    def refine_near_seed(self, seed_vec: np.ndarray, N: int, alive: AliveMask,
                         soup_factor=5, iters=3):
        """
        1) pull ~soup_factor*N exact neighbors (post-filtered alive), expanding k as needed
        2) select N nearest to current center
        3) recenter (mean) and repeat
        """
        # If not enough alive points overall, signal termination
        if alive.count_alive() < N:
            return None

        idsN, idsSoup = self.search_alive(seed_vec, N, alive, k_factor=soup_factor)
        if idsSoup.size < N:
            return None  # not enough alive points nearby (or in total)

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

    # Seed management: start with some alive ids and refresh as needed
    def get_seeds(num=200):
        return sample_alive_ids(alive, num, rng)

    seed_ids = get_seeds(500)

    def reduce1(subset_ids: np.ndarray) -> int:
        # Example policy: remove one endpoint of current diameter
        S = xb[subset_ids]
        diam, (i, j) = exact_diameter(S)
        return int(subset_ids[i])  # your custom choice

    results = []
    max_iters = 10000
    soup_factor = 5
    for t in range(max_iters):
        # Stop if not enough alive remain to form a subset
        if alive.count_alive() < N:
            print(f"Stopped: only {alive.count_alive()} alive < N={N}.")
            break

        best = None
        found = False

        # Try multiple random alive seeds each round
        # If existing seed pool is small or stale, refresh it
        if seed_ids.size < 50:
            seed_ids = get_seeds(500)
        rng.shuffle(seed_ids)
        trial_seeds = seed_ids[:64]

        # Attempt with current soup_factor; if not found, escalate once
        for attempt in range(2):
            for sid in trial_seeds:
                s = xb[int(sid)]
                cand = finder.refine_near_seed(s, N, alive, soup_factor=soup_factor, iters=3)
                if cand is None:
                    continue
                found = True
                if (best is None) or (cand["diameter"] < best["diameter"]):
                    best = cand
            if found:
                break
            # Escalate neighborhood size for a second attempt
            soup_factor = min(soup_factor * 2, 64)

        if not found or best is None:
            # As a last resort, fully refresh seeds and try once more
            seed_ids = get_seeds(1000)
            for sid in seed_ids[:128]:
                s = xb[int(sid)]
                cand = finder.refine_near_seed(s, N, alive, soup_factor=soup_factor, iters=3)
                if cand is None:
                    continue
                found = True
                if (best is None) or (cand["diameter"] < best["diameter"]):
                    best = cand

        if not found or best is None:
            # Could not find a valid subset even after escalation; likely because
            # the local neighborhoods are exhausted. Continue to next iteration
            # after refreshing seeds; if this persists, loop will stop once alive < N.
            print("No valid subset this round; refreshing seeds and continuing.")
            seed_ids = get_seeds(1000)
            continue

        kill_id = reduce1(best["subset_ids"])
        alive.kill(kill_id)
        if t % 500 == 0:
            results.append({"iter": t, "kill_id": kill_id, "diameter": best["diameter"]})
            print(f"[iter {t}] kill {kill_id}, diameter={best['diameter']:.4f}, alive={alive.count_alive()}")

    return results

if __name__ == "__main__":
    example()