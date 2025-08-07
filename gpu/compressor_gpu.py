import numpy as np
# from scipy.linalg import null_space
import faiss
import warnings
from copy import copy
from tqdm import tqdm
import torch

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

def null_space_gpu(X, rcond=1e-12, device='cuda'):
    """
    Compute basis for null space of matrix X (torch.Tensor or np.ndarray) on GPU via SVD.
    Returns a torch.Tensor of shape (n, nullity) on the device.
    """
    # Move or cast input to torch Tensor on the target device
    if not isinstance(X, torch.Tensor):
        X = torch.from_numpy(X)
    # X = X.to(device=device, dtype=torch.float64)

    # Full SVD to get Vh with shape (n, n)
    U, S, Vh = torch.linalg.svd(X, full_matrices=True)

    # Determine tolerance for zero singular values
    tol = rcond * S.max()
    rank = int((S > tol).sum().item())

    # Nullity = number of zero singular values = n - rank
    n = Vh.shape[1]
    nullity = n - rank
    if nullity <= 0:
        return torch.empty((n, 0), device=device, dtype=X.dtype)

    # Null space basis = last `nullity` rows of Vh, transposed -> shape (n, nullity)
    Z = Vh[rank:, :].T
    return Z


class Compressor:
    """
    Moment compression with diameter-aware Carathéodory peeling.
    """
    def __init__(self, data, weights=None, tol=1e-12, random_state=0,
                 gpu_device=0, multi_gpu=False, index_type='flat'):
        self.w_ = np.asarray(data, dtype=float)
        if self.w_.ndim != 2:
            raise ValueError("`data` must be a 2D array of shape (d, m)")
        self.d, self.m = self.w_.shape

        self.tol = tol
        self.rng = np.random.default_rng(random_state)

        self.index_type = index_type

        if weights is None:
            self.c_ = np.full(self.d, 1.0)
        else:
            self.c_ = np.asarray(weights, dtype=float)
            assert len(self.c_) == self.d
        self.alive = np.nonzero(self.c_ > self.tol)[0]

        # alive bookkeeping and rebuild policy
        self.alive_mask = self.c_ > self.tol
        self.removed_since_rebuild = 0

        # GPU options
        print(f"Running on {'GPU' if _has_cuda() else 'CPU'}")
        self.gpu_device = int(gpu_device)
        self.multi_gpu = bool(multi_gpu)
        self.gpu_res = None

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
        self.removed_since_rebuild = 0

        # GPU placement (optional)
        place_on_gpu = _has_cuda()
        if place_on_gpu:
            try:
                # Ensure GPU resources are initialized
                if self.gpu_res is None:
                    try:
                        self.gpu_res = faiss.StandardGpuResources()
                    except AttributeError:
                        warnings.warn("faiss.StandardGpuResources not available; falling back to CPU index.")
                        return base
                # Move base index to GPU
                if self.multi_gpu:
                    idx = faiss.index_cpu_to_all_gpus(base)
                else:
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

    def _refine_prune(self, idxs, target_size: int) -> np.ndarray:
        """Greedy remove the point with largest average distance until the target size is reached."""
        while len(idxs) > target_size:
            Y = self.w_[idxs]
            norms = np.sum(Y * Y, axis=1, keepdims=True)
            D2 = norms + norms.T - 2.0 * (Y @ Y.T)
            np.fill_diagonal(D2, 0.0)
            avg = np.mean(np.sqrt(D2), axis=1)
            remove_pos = int(np.argmax(avg))
            del idxs[remove_pos]
        return np.array(idxs, dtype=int)

    def _choose_candidate_centers(self, candidate_fraction, max_candidates):
        """Return a random subset of alive original indices to serve as candidate centers."""
        ccount = int(min(max(1, candidate_fraction * self.alive.size), max_candidates))
        if ccount >= self.alive.size:
            return self.alive
        return self.rng.choice(self.alive, size=ccount, replace=False)

    def _find_best_subset(self, target_size: int, overquery: int, candidate_fraction, max_candidates):
        """
        Find a size-`target_size` subset with small diameter *among alive vectors only*.
        Works even with overquery=0 by adaptively increasing k until enough alive ids
        are available. Caps growth to avoid runaway work.
        """
        # Base neighbors and a sane cap to prevent runaway work
        base_neigh = target_size + overquery
        # If many IDs are dead, we'll need to look a bit wider; cap at ~16× base or +256
        max_neigh_cap = min(self.alive.size, max(base_neigh * 16, target_size + 256))

        best_subset = None
        best_diam = np.inf

        # Fix candidate centers for this attempt; only grow k
        candidates = self._choose_candidate_centers(candidate_fraction, max_candidates)
        center_vecs = np.ascontiguousarray(self.w_[candidates, :])

        n_neigh = base_neigh
        while n_neigh <= max_neigh_cap and best_subset is None:
            _, I = self.index.search(center_vecs, n_neigh)

            for subset in I:
                # Keep only valid & ALIVE ids; skip empty/too-short rows
                # row will contain -1 if not found
                subset = [x for x in subset if x >= 0 and self.alive_mask[x]]
                # Prune/refine down to exactly target_size and evaluate
                subset = self._refine_prune(subset, target_size)
                if len(subset) < target_size:
                    continue

                diam = self._diameter(subset)
                if diam < best_diam:
                    best_diam = diam
                    best_subset = subset

            if best_subset is None:
                # Not enough alive neighbors yet; escalate k (double, with a small linear bump)
                n_neigh = min(max_neigh_cap, max(n_neigh + 8, n_neigh * 2))

        if best_subset is None:
            raise RuntimeError("Failed to find a viable alive subset of the requested size among current points.")
        return best_diam, best_subset

    def _reduce1(self, subset):
        '''
        Caratheodory peeling
        Ensure len(subset) = Nmk+1
        '''
        # Fetch precomputed moment features for this subset (shape (subset_size, D))
        A = self.all_moments[subset]             # torch.Tensor on device
        # Transpose to shape (D, subset_size) for null-space

        # Compute null-space basis on GPU; Z has shape (subset_size, nullity)
        Z = null_space_gpu(A.T, rcond=self.tol, device=A.device)
        # Take the first null vector (torch.Tensor on device)
        alpha_t = Z[:, 0]                            # shape (subset_size,)

        # Move alpha to CPU as numpy for weight updates
        alpha = alpha_t.cpu().numpy()

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
            self.removed_since_rebuild += remove.size


    # ------------------------ Public API ------------------------
    def compress_weights(self,
        k: int,
        dstop=None,                 # stop when d <= dstop; None means binom(m+k, k)
        return_at=None,             # None or list
        candidate_fraction=0.1,     # fraction of alive points used as candidate centers
        max_candidates: int=5000,
        rebuild_dead_ratio=0.2,
        rebuild_dead_min: int=1000,
        overquery: int=5,                # extra neighbors to fetch beyond D+1
        progress_bar=False, 
        print_progress=True
        ) -> None | dict:
        """
        Execute the Carathéodory peeling until the active set size ≤ dstop.
        Returns
        -------
        a dictionary of {d: weights} if return_at is assigned; else None
        """
        if return_at is not None:
            outputs = dict()
            return_list = sorted(list(return_at))

        # precompute all moments
        exps = multi_exponents(self.m, k)
        Nmk = len(exps)
        device = 'cuda'
        self.all_moments = torch.from_numpy( np.stack([all_moments(w, exps) for w in self.w_], axis=0) ).to(device)

        # Determine stopping threshold
        if return_at is not None:
            dstop = return_list[0]
        if dstop is None:
            dstop = Nmk
        elif dstop < Nmk:
            warnings.warn("dstop can't be smaller than binom(m+k, k); setting dstop = binom(m+k, k)")
            dstop = Nmk

        # progress bar setup
        if progress_bar:
            pbar = tqdm(total=self.d, desc="Compressing", unit="pt")
            prev_alive = self.alive.size

        # main loop
        while self.alive.size > dstop:
            best_diam, best_subset = self._find_best_subset(Nmk+1, overquery, candidate_fraction, max_candidates)
            self._reduce1(best_subset)
            # Rebuild the physical index only when too many dead entries accumulate
            dead_fraction = self.removed_since_rebuild / max(1, self.alive.size)
            if (self.removed_since_rebuild >= rebuild_dead_min) or (dead_fraction >= rebuild_dead_ratio):
                self.index = self._build_flat_index()

            if return_at is not None:
                if self.alive.size in return_list:
                    outputs[self.alive.size] = self.c_.copy()
                    if print_progress:
                        print(f'Compress progress: {self.alive.size}/{self.d}')

            if progress_bar:
                removed = prev_alive - self.alive.size
                if removed > 0:
                    pbar.update(removed)
                    prev_alive = self.alive.size
                pbar.set_postfix(best_diam=f"{best_diam:.2e}")

        if progress_bar:
            pbar.close()

        if return_at is not None:
            return outputs


    def compress(self, k: int, **kwargs):
        self.compress_weights(k, **kwargs)
        c_ = self.c_[self.alive].copy()
        w_ = self.w_[self.alive, :].copy()
        return c_, w_