import numpy as np
import itertools
from scipy.linalg import null_space
import faiss
import warnings
from copy import copy

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
    def __init__(self, data, weights=None, tol=1e-12, random_state=0, index_type='flat'):
        self.w_ = np.asarray(data, dtype=float)
        if self.w_.ndim != 2:
            raise ValueError("`data` must be a 2D array of shape (d, m)")
        self.d, self.m = self.w_.shape

        self.index_type = index_type
        self.tol = tol
        self.rng = np.random.default_rng(random_state)

        if weights is None:
            self.c_ = np.full(self.d, 1.0)
        else:
            self.c_ = np.asarray(weights, dtype=float)
            assert len(self.c_) == self.d   
        self.alive = np.nonzero(self.c_ > self.tol)[0]

        # Build initial index
        if index_type == 'flat':
            self.index = faiss.IndexIDMap2(faiss.IndexFlatL2(self.m))
            self.index.add_with_ids(self.w_[self.alive].astype(np.float32), self.alive)
        elif index_type == 'ivf':
            self.index = self._build_ivf_index()
        else:
            raise ValueError("index_type must be 'ivf' or 'flat'.")


    # ------------------------ Internal helpers ------------------------
    def _build_ivf_index(self):
        """(re)build ivf data from scratch with robust nlist/nprobe and training size"""
        alive_n = int(self.alive.size)
        # choose nlist so avg list occupancy ~100 and satisfy FAISS training rule 39*nlist <= alive_n
        target_occ = 100
        nlist = max(32, min(8192, alive_n // max(1, target_occ)))
        nlist = min(nlist, max(32, alive_n // 39))  # ensure enough train data; shrink nlist if dataset is small
        if nlist < 32:
            nlist = 32

        # choose nprobe ~5% of nlist (bounded)
        nprobe = max(8, min(512, int(0.05 * nlist)))

        # build quantizer + IVF
        quant = faiss.IndexFlatL2(self.m)
        ivf = faiss.IndexIVFFlat(quant, self.m, int(nlist), faiss.METRIC_L2)

        # training sample: at least min(alive_n, 39*nlist)
        train_sz = min(alive_n, 39 * int(nlist))
        if alive_n <= train_sz:
            train_ids = self.alive
        else:
            train_ids = self.rng.choice(self.alive, size=train_sz, replace=False)
        ivf.train(self.w_[train_ids].astype(np.float32))

        # wrap in IDMap2 and add alive vectors/ids
        idmap = faiss.IndexIDMap2(ivf)
        idmap.add_with_ids(self.w_[self.alive].astype(np.float32), self.alive)

        # print(f"Rebuilt ivf index: nlist={nlist}, nprobe={nprobe}, alive={alive_n}, train_sz={train_sz}")
        return idmap


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
        center_vecs = self.w_[candidates, :]
        _, I = self.index.search(center_vecs, n_neigh)
        # If not found, I will be padded by -1. I.shape = nq*m
        for subset in I:
            subset = self._refine_prune(subset, target_size)
            if len(subset) < target_size:
                continue
            diam = self._diameter(subset)
            if diam < best_diam:
                best_diam = diam
                best_subset = subset

        if best_subset is None:
            if self.index_type == 'ivf':
                print("[fallback] IVF search failed to find a viable subset; switching to Flat (IndexFlatL2) and retrying once.")
                self.index_type = 'flat'
                self.index = faiss.IndexIDMap2(faiss.IndexFlatL2(self.m))
                self.index.add_with_ids(self.w_[self.alive].astype(np.float32), self.alive)
                return self._find_best_subset(target_size, overquery, candidate_fraction, max_candidates)
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
        self.alive = self.alive[~np.isin(self.alive, remove)]
        self.index.remove_ids(np.array(remove, dtype='int64'))


    # ------------------------ Public API ------------------------
    def compress_weights(self,
        k: int,
        dstop=None,                 # stop when d <= dstop; None means dstop = binom(m+k, k)
        candidate_fraction=0.1,     # fraction of alive points used as candidate centers
        max_candidates: int=5000,
        overquery: int=5,                # extra neighbors to fetch beyond D+1
        rebuild_interval: int=2,      # retrain / rebuild when this fraction of ORIGINAL points removed
        verbose: bool=False, 
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
            return_list = list(return_at)
        if self.index_type == 'ivf':    
            rebuild_threshold = self.d-rebuild_interval

        self.exps = multi_exponents(self.m, k)
        Nmk = len(self.exps)

        # Determine stopping threshold
        if dstop is None:
            dstop = Nmk
        elif dstop < Nmk:
            warnings.warn("dstop can't be smaller than binom(m+k, k); setting dstop = binom(m+k, k)")
            dstop = Nmk

        # main loop
        while self.alive.size > dstop:
            best_diam, best_subset = self._find_best_subset(Nmk+1, overquery, candidate_fraction, max_candidates)
            self._reduce1(best_subset)

            if return_at is not None:
                if self.alive.size <= return_list[-1]:
                    outputs[self.alive.size] = self.c_.copy()
                    return_list.pop()

            if verbose:
                print(f"diameter={best_diam:.4e}, #alive={self.alive.size}")
            
            if self.index_type == 'ivf':
                # switch to flat when alive becomes small
                if self.alive.size <= 20000:
                    self.index_type = 'flat'
                    self.index = faiss.IndexIDMap2(faiss.IndexFlatL2(self.m))
                    self.index.add_with_ids(self.w_[self.alive].astype(np.float32), self.alive)
                    continue
                # Rebuild if enough removals accumulated (IVF only)
                if self.alive.size <= rebuild_threshold:
                    self.index = self._build_ivf_index()
                    rebuild_threshold -= rebuild_interval
        
        if return_at is not None:
            return outputs


    def compress(self, k: int, **kwargs):
        self.compress_weights(k, **kwargs)
        c_ = self.c_[self.alive].copy()
        w_ = self.w_[self.alive, :].copy()
        return c_, w_

