import numpy as np
import itertools
from scipy.linalg import null_space
import faiss
import warnings

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


class Compressor:
    """
    Moment compression with diameter-aware Carathéodory peeling.

    Usage:
        comp = Compressor(data, k, ...)
        c_, w_ = comp.run()
    """
    def __init__(self, data, weights=None, tol=1e-12, random_state=0, index_type='flat'):
        self.w_ = np.asarray(data, dtype=float)
        if self.w_.ndim != 2:
            raise ValueError("`data` must be a 2D array of shape (d, m)")
        self.d, self.m = self.w_.shape

        self.index_type = index_type
        self.tol = tol

        # self.candidate_fraction = candidate_fraction
        # self.max_candidates = max_candidates
        # self.overquery = overquery
        # self.refine = refine
        # self.rebuild_fraction = rebuild_fraction

        self.rng = np.random.default_rng(random_state)


        if weights is None:
            self.c_ = np.full(self.d, 1.0/self.d)
        else:
            self.c_ = np.asarray(weights, dtype=float)
            assert len(self.c_) == self.d
            self.c_ /= np.sum(self.c_)
        self.alive = (self.c_ > self.tol)
        self.next_rebuild_threshold = self.rebuild_fraction * self.d

        # Mapping between FAISS IDs and original indices
        self.alive_indices = np.arange(self.d)[self.alive]

        # Build initial index
        self.index = faiss.IndexFlatL2(self.m)
        self._build_index(self.w_[self.alive])


    # ------------------------ Internal helpers ------------------------
    def _build_index(self, points: np.ndarray):
        """Build or rebuild FAISS index on provided points."""
        pts = points.astype(np.float32)
        if self.index_type == 'flat':
            self.index.add(pts)
        elif self.index_type == 'ivf':
            # automatic params
            _nlist = max(32, int(min(4 * np.sqrt(points.shape[0]), 8192)))
            _nprobe = 32
            _max_sample = 10000

            self.index = faiss.IndexIVFFlat(self.index, self.m, _nlist, faiss.METRIC_L2)
            # Training sample (up to 10k or all alive)
            if pts.shape[0] <= 10000:
                self.index.train(pts)
            else:
                sample_idx = self.rng.choice(pts.shape[0], size=_max_sample, replace=False)
                self.index.train(pts[sample_idx])
            self.index.add(pts)
            self.index.nprobe = _nprobe
        else:
            raise ValueError("index_type must be 'ivf' or 'flat'.")

    def _rebuild_if_needed(self):
        """Rebuild the index on current alive points when enough removals have accumulated."""
        if self.removed_total >= self.next_rebuild_threshold:
            self.alive_indices = np.where(self.alive)[0]
            if self.index_type == 'ivf':
                self.index = self._build_index(self.w_[self.alive])
            else:
                # For Flat, rebuilding is optional; keep behavior consistent with verbose monitoring
                self.index = self._build_index(self.w_)
            self.next_rebuild_threshold += self.rebuild_fraction * self.d

    def _diameter(self, idx_subset: np.ndarray) -> float:
        Y = self.w_[idx_subset]
        norms = np.sum(Y * Y, axis=1, keepdims=True)
        D2 = norms + norms.T - 2.0 * (Y @ Y.T)
        np.fill_diagonal(D2, 0.0)
        return float(np.sqrt(np.max(D2)))

    def _refine_prune(self, indices: np.ndarray, target_size: int) -> np.ndarray:
        """Greedy remove the point with largest average distance until the target size is reached."""
        idxs = list(indices)
        while len(idxs) > target_size:
            Y = self.w_[idxs]
            norms = np.sum(Y * Y, axis=1, keepdims=True)
            D2 = norms + norms.T - 2.0 * (Y @ Y.T)
            np.fill_diagonal(D2, 0.0)
            avg = np.mean(np.sqrt(D2), axis=1)
            remove_pos = int(np.argmax(avg))
            del idxs[remove_pos]
        return np.array(idxs, dtype=int)

    def _choose_candidate_centers(self) -> np.ndarray:
        """Return a random subset of alive original indices to serve as candidate centers."""
        alive_idx = np.where(self.alive)[0]
        ccount = int(min(max(1, self.candidate_fraction * alive_idx.size), self.max_candidates))
        if ccount >= alive_idx.size:
            return alive_idx
        return self.rng.choice(alive_idx, size=ccount, replace=False)

    def _center_neighbors(self, center_orig_idx: int, need: int) -> np.ndarray:
        """
        Return up to `need` alive neighbor original indices INCLUDING the center.
        Uses a loop with growing k_query if dead points occupy slots.
        """
        if self.index_type == 'flat':
            query_vec = self.w_[center_orig_idx].astype(np.float32)[None, :]
            k_query = min(self.w_.shape[0], need + self.overquery)
            while k_query <= self.w_.shape[0]:
                _, neigh = self.index.search(query_vec, k_query)
                neigh_ids = neigh[0]
                alive_neigh = [i for i in neigh_ids if self.alive[i]]
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
                if len(filtered) >= need or k_query == self.w_.shape[0]:
                    return np.array(filtered[:need], dtype=int)
                k_query = min(self.w_.shape[0], int(k_query * 1.5) + 1)
            return np.array(filtered, dtype=int)
        else:
            query_vec = self.w_[center_orig_idx].astype(np.float32)[None, :]
            alive_count = self.alive_indices.size
            k_query = min(alive_count, need + self.overquery)
            while k_query <= alive_count:
                _, neigh = self.index.search(query_vec, k_query)
                mapped = self.alive_indices[neigh[0]]
                alive_neigh = [i for i in mapped if self.alive[i]]
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
        
    def _reduce1(self, subset, self.tol=1e-12):
        '''
        Caratheodory peeling
        Ensure len(subset) = Nmk+1
        '''
        # Build D x (D+1) moment matrix for best_subset
        A = np.empty((Nmk, len(subset)), dtype=float)
        for col, j in enumerate(subset):
            A[:, col] = all_moments(self.w_[j], self.exps)

        # Null-space direction
        alpha = null_space(A, rcond=self.tol)[:, 0]
        if not np.any(alpha > 0):
            alpha = -alpha

        # Determine maximal step t
        t = min((self.c_[j] / a_j) for a_j, j in zip(alpha, subset) if a_j > 0)

        # Update weights; mark indices to drop (weight hits zero)
        dropped = []
        for aj, j in zip(alpha, subset):
            self.c_[j] -= t * aj
            if self.c_[j] <= self.tol:
                self.alive[j] = False
                self.c_[j] = 0.
        #         if self.alive[j]:
        #             self.alive[j] = False
        #             dropped.append(j)
        #             self.c_[j] = 0.0

        # self.removed_total += len(dropped)
        # active_count -= len(dropped)


    # ------------------------ Public API ------------------------
    def compress_weights(self,         k: int,
        dstop=None,                 # stop when d <= dstop; None means dstop = binom(m+k, k)
        candidate_fraction=0.1,     # fraction of alive points used as candidate centers
        max_candidates=10000,
        overquery=2,                # extra neighbors to fetch beyond D+1
        refine=True,                # apply simple pruning refinement inside candidate cluster
        rebuild_fraction=0.30,      # retrain / rebuild when this fraction of ORIGINAL points removed
        verbose=False):
        """
        Execute the Carathéodory peeling until the active set size ≤ dstop.
        Returns
        -------
        c_ : np.ndarray, shape (N,)
            Positive weights scaled to sum to 1.
        """
        # Build exponent list and feature dimension D
        self.exps = multi_exponents(self.m, k)
        Nmk = len(self.exps)

        # Determine stopping threshold
        if dstop is None:
            dstop = Nmk
        elif dstop < Nmk:
            warnings.warn("dstop can't be smaller than binom(m+k, k); setting dstop = binom(m+k, k)")
            dstop = Nmk
        else:
            dstop = dstop

        active_count = int(np.count_nonzero(self.alive))

        while active_count > dstop:
            target_size = Nmk + 1
            best_subset = None
            best_diam = np.inf

            candidates = self._choose_candidate_centers()
            for cidx in candidates:
                neigh = self._center_neighbors(cidx, target_size + self.overquery if self.refine else target_size)
                if neigh.size < target_size:
                    continue
                if self.refine and neigh.size > target_size:
                    neigh = self._refine_prune(neigh, target_size)
                diam_val = self._diameter(neigh)
                if diam_val < best_diam:
                    best_diam = diam_val
                    best_subset = neigh

            if best_subset is None:
                raise RuntimeError("Failed to find a viable subset of size D+1 among alive points.")

            # Build D x (D+1) moment matrix for best_subset
            A = np.empty((Nmk, target_size), dtype=float)
            for col, j in enumerate(best_subset):
                A[:, col] = all_moments(self.w_[j], self.exps)

            # Null-space direction
            alpha = null_space(A, rcond=self.self.tol)[:, 0]
            if not np.any(alpha > 0):
                alpha = -alpha

            # Determine maximal step t
            t = min((self.c_[j] / a_j) for a_j, j in zip(alpha, best_subset) if a_j > 0)

            # Update weights; mark indices to drop (weight hits zero)
            dropped = []
            for a_j, j in zip(alpha, best_subset):
                self.c_[j] -= t * a_j
                if self.c_[j] <= self.self.tol:
                    if self.alive[j]:
                        self.alive[j] = False
                        dropped.append(j)
                        self.c_[j] = 0.0

            self.removed_total += len(dropped)
            active_count -= len(dropped)

            if verbose:
                print(f"diameter={best_diam:.4e}, dropped={len(dropped)}, active={active_count}")

            # Rebuild if enough removals accumulated (IVF primarily; Flat optional)
            if (self.index_type == 'ivf' or self.verbose) and self.removed_total >= self.next_rebuild_threshold:
                self._rebuild_if_needed()

        # Collect final support
        final_idx = [j for j in range(self.d) if self.alive[j] and self.c_[j] > self.tol]
        c_ = np.array([self.c_[j] * self.d for j in final_idx], dtype=float)
        w_final = self.w_[final_idx, :].copy()
        return c_, w_final

