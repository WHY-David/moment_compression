import numpy as np
import torch
import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt
import math

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from compressor import Compressor

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, d_heads: int, d_in: int, d_out: int):
        super().__init__()
        assert d_model % d_heads == 0, "d_model must be divisible by d_heads"
        self.d_model = d_model
        self.d_heads = d_heads
        self.d_in = d_in
        self.d_out = d_out
        self.d_head = d_model // d_heads

        # Linear layers for Q, K, V
        self.W_q = nn.Linear(d_in, d_model, bias=False)
        self.W_k = nn.Linear(d_in, d_model, bias=False)
        self.W_v = nn.Linear(d_in, d_model, bias=False)

        # Output projection
        self.W_o = nn.Linear(d_model, d_out, bias=False)

    def _split_heads(self, x):
        # x: (B, T, d_model) -> (B, d_heads, T, d_head)
        B, T, _ = x.shape
        x = x.view(B, T, self.d_heads, self.d_head)
        return x.permute(0, 2, 1, 3)

    def _combine_heads(self, x):
        # x: (B, d_heads, T, d_head) -> (B, T, d_model)
        B, H, T, Dh = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(B, T, H * Dh)

    def forward(self, x, mask=None):
        # x: (B, T, d_in)
        Q = self._split_heads(self.W_q(x))
        K = self._split_heads(self.W_k(x))
        V = self._split_heads(self.W_v(x))
        # Q, K, V: (B, H, T, d_head)

        # Scaled dot-product attention
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_head)  # (B, H, T, T)

        if mask is not None:
            # mask: (B, 1, 1, T) or (B, 1, T, T),  True/1 = keep, False/0 = mask
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = scores.softmax(dim=-1)          # (B, H, T, T)
        context = attn @ V                     # (B, H, T, d_head)

        # Combine heads and project
        context = self._combine_heads(context) # (B, T, d_model)
        out = self.W_o(context)                # (B, T, d_model)
        return out
    

class MultiHeadAttentionW(nn.Module):
    def __init__(self, d_model: int, d_heads: int, d_in: int, d_out: int, weights=None):
        super().__init__()
        assert d_model % d_heads == 0, "d_model must be divisible by d_heads"

        self.d_model = d_model
        self.d_heads = d_heads
        self.d_in = d_in
        self.d_out = d_out
        self.d_head = d_model // d_heads

        # Linear layers for Q, K, V
        self.W_q = nn.Linear(d_in, d_model, bias=False)
        self.W_k = nn.Linear(d_in, d_model, bias=False)
        self.W_v = nn.Linear(d_in, d_model, bias=False)

        # Output projection
        self.W_o = nn.Linear(d_model, d_out, bias=False)

        # Per-head weights (non-trainable buffer)
        if weights is None:
            w = torch.ones(d_heads, dtype=torch.float32)
        else:
            w = torch.as_tensor(weights, dtype=torch.float32)
            assert w.shape == (d_heads,), "weights must have shape (d_heads,)"
        # Register as buffer so .to(device) moves it automatically
        self.register_buffer("weights", w.view(-1))

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, d_model) -> (B, d_heads, T, d_head)
        """
        B, T, _ = x.shape
        x = x.view(B, T, self.d_heads, self.d_head)
        return x.permute(0, 2, 1, 3)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, d_heads, T, d_head) -> (B, T, d_model)
        """
        B, H, T, Dh = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(B, T, H * Dh)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (B, T, d_in)
        mask: optional, shape (B, 1, 1, T) or (B, 1, T, T), with 1/True = keep, 0/False = mask
        """
        # Project to Q, K, V
        Q = self._split_heads(self.W_q(x))  # (B, H, T, d_head)
        K = self._split_heads(self.W_k(x))  # (B, H, T, d_head)
        V = self._split_heads(self.W_v(x))  # (B, H, T, d_head)

        # Scaled dot-product attention
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_head)  # (B, H, T, T)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = scores.softmax(dim=-1)           # (B, H, T, T)
        context = attn @ V                      # (B, H, T, d_head)

        # --- Per-head weighting ---
        # weights: (H,) -> (1, H, 1, 1) to broadcast over (B, H, T, d_head)
        w = self.weights.view(1, self.d_heads, 1, 1)
        context = context * w                   # (B, H, T, d_head), head h scaled by weights[h]

        # Combine heads and project
        context = self._combine_heads(context)  # (B, T, d_model)
        out = self.W_o(context)                 # (B, T, d_model)
        return out


def extract(mha: MultiHeadAttention):
    """
    Extract per-head parameters from a MultiHeadAttention as a 2D array w_.

    Each row corresponds to one head h and is

        [ vec(W_q^{(h)}), vec(W_k^{(h)}), vec(W_v^{(h)}), vec(W_o^{(h)}) ],

    where W_*^{(h)} is the (d_head, d_in) row-block of that head for Q/K/V,
    and W_o^{(h)} is the (d_out, d_head) column-block of that head.
    So w_.shape = (d_heads, 3 * d_head * d_in + d_out * d_head).
    """
    with torch.no_grad():
        d_model = mha.d_model
        d_heads = mha.d_heads
        d_in = mha.d_in
        d_out = mha.d_out
        d_head = mha.d_head

        # Each is (out_features=d_model, in_features=d_in)
        Wq = mha.W_q.weight.data.cpu().numpy()  # (d_model, d_in)
        Wk = mha.W_k.weight.data.cpu().numpy()
        Wv = mha.W_v.weight.data.cpu().numpy()
        Wo = mha.W_o.weight.data.cpu().numpy()

        assert Wq.shape == (d_model, d_in)
        assert Wk.shape == (d_model, d_in)
        assert Wv.shape == (d_model, d_in)
        assert Wo.shape == (d_out, d_model)
        assert d_model == d_heads * d_head, "d_model must be d_heads * d_head"

        num_params_qkv = d_head * d_in
        num_params_wo = d_out * d_head
        num_params_head = 3 * num_params_qkv + num_params_wo

        rows = []
        for h in range(d_heads):
            s = h * d_head
            e = (h + 1) * d_head

            # rows s:e correspond to head h
            Wq_h = Wq[s:e, :]     # (d_head, d_in)
            Wk_h = Wk[s:e, :]
            Wv_h = Wv[s:e, :]
            Wo_h = Wo[:, s:e]     # (d_out, d_head)

            row = np.concatenate(
                [
                    Wq_h.reshape(-1),
                    Wk_h.reshape(-1),
                    Wv_h.reshape(-1),
                    Wo_h.reshape(-1),
                ],
                axis=0,
            )
            assert row.shape[0] == num_params_head
            rows.append(row)

        w_ = np.stack(rows, axis=0)  # (d_heads, num_params_head)

    return w_

def compress_mha(
    mha: MultiHeadAttention,
    k: int = 1,
    dstop: int = 8,
    tol: float = 1e-12,
    print_progress: bool = True,
):
    """
    Compress a MultiHeadAttention into a MultiHeadAttentionW with dstop heads.

    Steps:
    1. Use `extract` to get a row-per-head matrix w_ of shape
           (d_heads_orig, 3 * d_head_orig * d_in + d_out * d_head_orig).
    2. Run `Compressor` on rows = heads to get:
           weights (length dstop),
           w_cp   (dstop, 3 * d_head_orig * d_in + d_out * d_head_orig).
    3. Build a new MultiHeadAttentionW with d_heads = dstop.
    4. Map each compressed row back to Q/K/V and W_o blocks for the new heads.

    Because d_head changes when we change d_heads, we adapt the row length
    by cropping or zero-padding so it matches 3 * d_head_new * d_in + d_out * d_head_new.
    """
    device = mha.W_q.weight.device

    # 1) Extract original per-head parameter rows
    w_orig = extract(mha)              # (d_heads_orig, num_params_head_orig)
    # d_heads_orig = w_orig.shape[0]
    # print(w_orig.shape)

    # 2) Compress across heads
    cp = Compressor(w_orig, tol=tol)
    weights, w_cp = cp.compress(k, dstop=dstop, print_progress=print_progress)
    # w_cp: (dstop, num_params_head_orig)

    d_in = mha.d_in
    d_out = mha.d_out
    d_head = mha.d_head
    d_heads_new = dstop
    d_model_new = dstop*d_head

    num_params_head = w_orig.shape[1]

    # 3) Build compressed weighted MHA
    mha_cp = MultiHeadAttentionW(
        d_model=d_model_new,
        d_heads=d_heads_new,
        d_in=d_in,
        weights=weights,
        d_out=d_out,
    ).to(device)

    # 4) Reconstruct W_q, W_k, W_v, W_o for the compressed heads
    Wq_new = np.zeros((d_model_new, d_in), dtype=np.float32)
    Wk_new = np.zeros_like(Wq_new)
    Wv_new = np.zeros_like(Wq_new)
    Wo_new = np.zeros((d_out, d_model_new), dtype=np.float32)

    for h_new in range(d_heads_new):
        row = w_cp[h_new]  # (num_params_head,)

        # Shape-adapt the row if d_head changed:
        assert len(row) == num_params_head

        q_flat = row[0:d_head*d_in]
        k_flat = row[d_head*d_in: 2 * d_head*d_in]
        v_flat = row[2 * d_head*d_in: 3 * d_head*d_in]
        o_flat = row[3 * d_head*d_in: ]

        Wq_h = q_flat.reshape(d_head, d_in)
        Wk_h = k_flat.reshape(d_head, d_in)
        Wv_h = v_flat.reshape(d_head, d_in)
        Wo_h = o_flat.reshape(d_out, d_head)

        s = h_new * d_head
        e = (h_new + 1) * d_head
        Wq_new[s:e, :] = Wq_h
        Wk_new[s:e, :] = Wk_h
        Wv_new[s:e, :] = Wv_h
        Wo_new[:, s:e] = Wo_h

    with torch.no_grad():
        # Copy compressed Q/K/V
        mha_cp.W_q.weight.copy_(torch.from_numpy(Wq_new).to(device))
        mha_cp.W_k.weight.copy_(torch.from_numpy(Wk_new).to(device))
        mha_cp.W_v.weight.copy_(torch.from_numpy(Wv_new).to(device))

        # Copy output projection from compressed weights
        mha_cp.W_o.weight.copy_(torch.from_numpy(Wo_new).to(device))

    weights_t = torch.as_tensor(weights, dtype=torch.float32, device=device)
    return mha_cp, weights_t


if __name__ == '__main__':
    mha = MultiHeadAttention(d_model=10000, d_heads=5000, d_in=2, d_out=2)
    mha_cp, head_weights = compress_mha(mha, k=2, dstop=203, print_progress=True)