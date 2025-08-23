import numpy as np
# import itertools
# from scipy.linalg import null_space
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering
import faiss
import warnings
# from tqdm import tqdm
from typing import Union, Optional
from joblib import Parallel, delayed

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

def find_null_vec(A, rng, max_trials=5, ridge = 0., tol=1e-12):
    z = None
    if A.shape[1] > 1.5*A.shape[0]:
        for _ in range(max_trials):
            # Random probe and projection onto Null(AS):
            # z = (I - AS^T (AS AS^T + λI)^{-1} AS) r
            r = rng.standard_normal(A.shape[1])
            Ar = A @ r
            G = A @ A.T
            G_reg = G + ridge * np.eye(G.shape[0])
            try:
                y = np.linalg.solve(G_reg, Ar)
            except np.linalg.LinAlgError:
                # extremely ill-conditioned: fall back to least squares on the regularized system
                y = np.linalg.lstsq(G_reg, Ar, rcond=None)[0]
            z_candidate = r - A.T @ y
            if not np.any(z_candidate > 0):
                z_candidate = -z_candidate
            resid = np.linalg.norm(A @ z_candidate)
            # accept if we got a reasonably good null vector with some positive entries
            if np.any(z_candidate > 1e-5) and resid <= tol * (1.0 + np.linalg.norm(Ar)):
                z = z_candidate
                break

    if z is None:
        # Fallback: use SVD to get a null vector
        _, _, Vh = np.linalg.svd(A, full_matrices=True)
        z = Vh[-1, :]
        if not np.any(z > 0):
            z = -z

    return z


class Compressor:
    """
    Moment compression with diameter-aware Carathéodory peeling.
    """
    def __init__(self, data, weights=None, tol=1e-12, random_state=0):
        self.w_ = np.asarray(data, dtype=float)
        if self.w_.ndim != 2:
            raise ValueError("`data` must be a 2D array of shape (d, m)")
        self.d, self.m = self.w_.shape

        # self.index_type = index_type
        self.tol = tol
        self.rng = np.random.default_rng(random_state)

        if weights is None:
            self.c_ = np.full(self.d, 1.0)
        else:
            self.c_ = np.asarray(weights, dtype=float)
            assert len(self.c_) == self.d   
        self.alive = np.nonzero(self.c_ > self.tol)[0]

    def _build_index(self):
        self.index = faiss.IndexIDMap2(faiss.IndexFlatL2(self.m))
        self.index.add_with_ids(self.w_[self.alive].astype(np.float32), self.alive)


    def _diameter(self, idx_subset) -> float:
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

    def _reduce(self, subset):
        """
        This updates index as well. Used in greedy
        """
        _, c = self._reduce_compute(subset)
        self.c_[subset] = c
        remove = subset[c < self.tol]
        if len(remove):
            self.alive = self.alive[~np.isin(self.alive, remove)]
            self.index.remove_ids(np.array(remove, dtype=int))

    def _reduce_compute(self, subset):
        """
        Pure computation version of _reduce: does not mutate self.c_ or self.alive.
        Returns (subset, c_updated, remove_indices).
        Used in parallel
        """
        # subset = np.asarray(subset, dtype=int)
        A = self.all_moments[:, subset]
        c = self.c_[subset].copy()
        if np.any(c < -self.tol):
            raise ValueError("c must be nonnegative (within tolerance).")
        b = A@c
        target = A.shape[0]

        S = np.flatnonzero(c > self.tol).tolist()
        while len(S) > target:
            AS = A[:, S]
            z = find_null_vec(AS, self.rng, tol=self.tol)
            # update c
            zpos = z > 0
            t = np.min((c[S][zpos]) / z[zpos])
            c[S] -= t * z
            S = [idx for idx in S if c[idx] > self.tol]

        # sanity check
        c[c < self.tol] = 0.0
        diff = np.linalg.norm(A @ c - b)
        if diff > 1e-8 * (1.0 + np.linalg.norm(b)):
            warnings.warn(f"||A@c - b|| = {diff:.2e}. Consider increasing accuracy. ")

        return subset, c


    def compress(self, 
                 k:int, # moment matching order
                 dstop: Optional[int] = None,
                 greedy_threshold: int = 2000,
                 print_progress=False
                 ):
        exps = multi_exponents(self.m, k)
        Nmk = len(exps)
        self.all_moments = np.stack([all_moments(w, exps) for w in self.w_], axis=-1)
        assert self.all_moments.shape == (Nmk, self.d)

        # determing dstop
        if dstop is None:
            dstop = Nmk
        elif dstop < Nmk:
            warnings.warn("dstop can't be smaller than binom(m+k, k); setting dstop = binom(m+k, k)")
            dstop = Nmk

        method = 'kmeans' if self.alive.size>dstop+greedy_threshold else 'greedy'
        if method == 'greedy':
            self._build_index()

        while self.alive.size > dstop:
            prev_alive_size = self.alive.size
            # choose kmeans or greedy automatically
            if method == 'kmeans':
                n_clusters = int(min(0.98*self.alive.size / Nmk, 100*dstop/Nmk))
                mbk = MiniBatchKMeans(n_clusters=n_clusters, max_iter=200, batch_size=4096, random_state=0)
                labels = mbk.fit_predict(self.w_[self.alive], sample_weight=self.c_[self.alive])
                # indices within mask
                clusters = [np.where(labels==j)[0].tolist() for j in range(n_clusters)]
                # indicies in the original labeling
                tasks = [self.alive[subset] for subset in clusters if len(subset) > Nmk]

                diam = max(self._diameter(subset) for subset in tasks)

                # Parallel loop to compress each subset
                results = Parallel(n_jobs=-1, prefer='threads')(
                    delayed(self._reduce_compute)(subset) for subset in tasks
                )
                # Commit phase: write back weights and update alive/index once
                to_remove = []
                for subset, c_new in results:
                    self.c_[subset] = c_new
                    remove = subset[c_new < self.tol]
                    to_remove.extend(remove.tolist())
                if len(to_remove) != 0:
                    self.alive = self.alive[~np.isin(self.alive, to_remove)]

                if print_progress:
                    print(f"KMeans round: #alive={self.alive.size}/{self.d}, #removed={prev_alive_size-self.alive.size}, #clusters={n_clusters}, diam={diam:.2e}")
                if self.alive.size < dstop+greedy_threshold or n_clusters < 50:
                    method = 'greedy'
                    self._build_index()
            elif method == 'greedy':
                best_diam, best_subset = self._find_best_subset(Nmk+1, overquery=0, candidate_fraction=0.1, max_candidates=5000)
                assert len(best_subset) == Nmk+1
                self._reduce(best_subset) # update c and remove pts from index
                if print_progress and self.alive.size%500==0:
                    print(f"Greedy round: #alive={self.alive.size}/{self.d}, #removed={prev_alive_size-self.alive.size}, best diam={best_diam:.2e}")
            else:
                raise RuntimeError("Undefined reduction method")

            if self.alive.size == prev_alive_size:
                warnings.warn("A round reduced 0 points")
                if method == 'kmeans':
                    print("Fall back to sequential greedy reduction")
                    method = 'greedy'
                    self._build_index()
                

        return self.c_[self.alive].copy(), self.w_[self.alive, :].copy()