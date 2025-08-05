import numpy as np
from scipy.linalg import null_space
import faiss
import warnings
from copy import copy
from tqdm import tqdm
import os

# Try GPU; fall back to CPU automatically
def _has_cuda():
    try:
        _ = faiss.get_num_gpus()
        return faiss.get_num_gpus() > 0
    except Exception:
        return False

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


class Compressor:
    """
    Moment compression with diameter-aware Carathéodory peeling.
    """
    def __init__(self, data, weights=None, tol=1e-12, random_state=0,
                 use_gpu=True, gpu_device=0, multi_gpu=False,
                 rebuild_dead_ratio=0.2, rebuild_dead_min=100_000):
        self.w_ = np.asarray(data, dtype=float)
        if self.w_.ndim != 2:
            raise ValueError("`data` must be a 2D array of shape (d, m)")
        self.d, self.m = self.w_.shape

        self.tol = tol
        self.rng = np.random.default_rng(random_state)

        if weights is None:
            self.c_ = np.full(self.d, 1.0)
        else:
            self.c_ = np.asarray(weights, dtype=float)
            assert len(self.c_) == self.d   
        self.alive = np.nonzero(self.c_ > self.tol)[0]

        # alive bookkeeping and rebuild policy
        self.alive_mask = np.zeros(self.d, dtype=bool)
        self.alive_mask[self.alive] = True
        self.removed_since_rebuild = 0
        self.rebuild_dead_ratio = float(rebuild_dead_ratio)
        self.rebuild_dead_min = int(rebuild_dead_min)

        # GPU options
        self.use_gpu = bool(use_gpu)
        self.gpu_device = int(gpu_device)
        self.multi_gpu = bool(multi_gpu)
        self.gpu_res = None  # faiss.StandardGpuResources when on GPU

        # Build initial (Flat) index on CPU, then optionally move to GPU
        self.index = self._build_flat_index()


    # # ------------------------ Internal helpers ------------------------
    def _build_flat_index(self):
        """(Re)build a FlatL2 index with ID mapping; place on GPU if available/desired.
        We avoid per-iteration remove_ids on the device by maintaining an alive_mask
        and rebuilding only when dead fraction is large.
        """
        # Base CPU index with ID mapping so we preserve original ids
        base = faiss.IndexIDMap2(faiss.IndexFlatL2(self.m))
        alive_ids = self.alive.copy().astype('int64')
        base.add_with_ids(np.ascontiguousarray(self.w_[alive_ids].astype(np.float32)), alive_ids)

        # Track how many vectors are physically in the index (for dead-ratio checks)
        self._indexed_count = int(alive_ids.size)
        self.removed_since_rebuild = 0

        # GPU placement (optional)
        place_on_gpu = self.use_gpu and _has_cuda()
        if place_on_gpu:
            try:
                if self.multi_gpu:
                    idx = faiss.index_cpu_to_all_gpus(base)
                else:
                    if self.gpu_res is None:
                        self.gpu_res = faiss.StandardGpuResources()
                    idx = faiss.index_cpu_to_gpu(self.gpu_res, self.gpu_device, base)
                return idx
            except Exception as e:
                warnings.warn(f"FAISS GPU unavailable or failed ({e}); falling back to CPU.")
        return base

    def _diameter(self, idx_subset: np.ndarray) -> float:
        Y = self.w_[idx_subset]
        norms = np.sum(Y * Y, axis=1, keepdims=True)
        D2 = norms + norms.T - 2.0 * (Y @ Y.T)
        np.fill_diagonal(D2, 0.0)
        return float(np.sqrt(np.max(D2)))

    def _refine_prune(self, indices, target_size: int) -> np.ndarray:
        """Greedy remove the point with largest average distance until the target size is reached."""
        idxs = [int(x) for x in indices if int(x) >= 0]
        if len(idxs) <= target_size:
            return np.array(idxs, dtype=int)
        while len(idxs) > target_size:
            Y = self.w_[idxs]
            norms = np.sum(Y * Y, axis=1, keepdims=True)
            D2 = norms + norms.T - 2.0 * (Y @ Y.T)
            np.fill_diagonal(D2, 0.0)
            avg = np.mean(np.sqrt(D2), axis=1)
            remove_pos = int(np.argmax(avg))
            del idxs[remove_pos]
        return np.array(idxs, dtype=int)

    def _f32(self, X):
        return np.ascontiguousarray(X.astype(np.float32))

    def _choose_candidate_centers(self, candidate_fraction, max_candidates) -> np.ndarray:
        """Return a random subset of alive original indices to serve as candidate centers."""
        ccount = int(min(max(1, candidate_fraction * self.alive.size), max_candidates))
        if ccount >= self.alive.size:
            return self.alive
        return self.rng.choice(self.alive, size=ccount, replace=False)
    
    def _find_best_subset(self, target_size: int, overquery: int, candidate_fraction, max_candidates):
        n_neigh = target_size + overquery
        if target_size <= self.alive.size < n_neigh:
            n_neigh = self.alive.size

        best_subset = None
        best_diam = np.inf

        candidates = self._choose_candidate_centers(candidate_fraction, max_candidates)
        center_vecs = self._f32(self.w_[candidates, :])
        _, I = self.index.search(center_vecs, n_neigh)
        # If not found, I will be padded by -1. I.shape = nq*m
        for subset in I:
            # filter out invalid ids and dead points, then refine/prune
            subset = [int(x) for x in subset if int(x) >= 0 and self.alive_mask[int(x)]]
            if not subset:
                continue
            subset = self._refine_prune(subset, target_size)
            if len(subset) < target_size:
                continue
            diam = self._diameter(subset)
            if diam < best_diam:
                best_diam = diam
                best_subset = subset

        if best_subset is None:
            raise RuntimeError(
                "Failed to find a viable subset of the requested size among alive points; "
            )
        
        return best_diam, best_subset
        
    def _reduce1(self, subset):
        '''
        Caratheodory peeling
        Ensure len(subset) = Nmk+1
        '''
        # Build D x (D+1) moment matrix for best_subset
        A = np.empty((len(self.exps), len(subset)), dtype=float)
        for col, j in enumerate(subset):
            A[:, col] = all_moments(self.w_[j], self.exps)

        # Null-space direction
        alpha = null_space(A, rcond=self.tol)[:, 0]
        if not np.any(alpha > 0):
            alpha = -alpha

        # Determine maximal step t
        t = min((self.c_[j] / a_j) for a_j, j in zip(alpha, subset) if a_j > 0)

        # Update weights; update index
        remove = []
        for aj, j in zip(alpha, subset):
            self.c_[j] -= t * aj
            if self.c_[j] <= self.tol:
                self.c_[j] = 0.
                remove.append(j)

        # mark removed in mask and recompute alive list
        if remove:
            remove = np.array(remove, dtype=int)
            self.alive_mask[remove] = False
            self.alive = np.nonzero(self.alive_mask & (self.c_ > self.tol))[0]
            self.removed_since_rebuild += int(remove.size)

            # Rebuild the physical index only when too many dead entries accumulate
            dead_fraction = self.removed_since_rebuild / max(1, self._indexed_count)
            if (self.removed_since_rebuild >= self.rebuild_dead_min) or (dead_fraction >= self.rebuild_dead_ratio):
                self.index = self._build_flat_index()


    # ------------------------ Public API ------------------------
    def compress_weights(self,
        k: int,
        dstop=None,                 # stop when d <= dstop; None means dstop = binom(m+k, k)
        candidate_fraction=0.1,     # fraction of alive points used as candidate centers
        max_candidates: int=5000,
        overquery: int=5,                # extra neighbors to fetch beyond D+1
        return_at=None  # None or list; list ordered from small to large
        ) -> None | dict:
        """
        Execute the Carathéodory peeling until the active set size ≤ dstop.
        Returns
        -------
        c_ : np.ndarray, shape (N,)
            Positive weights scaled to sum to 1.
        """
        if return_at is not None:
            outputs = dict()
            return_list = sorted(list(return_at))

        self.exps = multi_exponents(self.m, k)
        Nmk = len(self.exps)

        # Determine stopping threshold
        if return_at is not None:
            dstop = return_list[0]
        if dstop is None:
            dstop = Nmk
        elif dstop < Nmk:
            warnings.warn("dstop can't be smaller than binom(m+k, k); setting dstop = binom(m+k, k)")
            dstop = Nmk

        # # progress bar setup
        # pbar = tqdm(total=self.d, desc="Compressing", unit="pt")
        # prev_alive = self.alive.size

        # main loop
        while self.alive.size > dstop:
            best_diam, best_subset = self._find_best_subset(Nmk+1, overquery, candidate_fraction, max_candidates)
            self._reduce1(best_subset)

            if return_at is not None:
                if self.alive.size in return_list:
                    outputs[self.alive.size] = self.c_.copy()
                    print(f'Compress progress: {self.alive.size}/{self.d}')

        #     # update progress bar at end of iteration
        #     removed = prev_alive - self.alive.size
        #     if removed > 0:
        #         pbar.update(removed)
        #         prev_alive = self.alive.size
        #     pbar.set_postfix(best_diam=f"{best_diam:.2e}")
        # pbar.close()
        
        if return_at is not None:
            return outputs


    def compress(self, k: int, **kwargs):
        self.compress_weights(k, **kwargs)
        c_ = self.c_[self.alive].copy()
        w_ = self.w_[self.alive, :].copy()
        return c_, w_
