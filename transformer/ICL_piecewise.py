#!/usr/bin/env python3
import math
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from mha import MultiHeadAttention, MultiHeadAttentionW, compress_mha  # your implementation
from common import make_canvas


# -----------------------------
#  Piecewise-linear function family
# -----------------------------

def generate_piecewise_linear_params(batch_size, K, sigma_f0, sigma_s, device):
    """
    Generate parameters for a batch of piecewise-linear functions on [0,1].

    For each episode b:
        - f_0[b] ~ N(0, sigma_f0^2)
        - slopes[b, j] ~ N(0, sigma_s^2), for j = 0,...,K-1.
    """
    f0 = sigma_f0 * torch.randn(batch_size, device=device)       # (B,)
    slopes = sigma_s * torch.randn(batch_size, K, device=device) # (B, K)
    return f0, slopes


def eval_piecewise_linear(x, f0, slopes, K):
    """
    Evaluate a batch of piecewise-linear functions at points x.

    x:      (B, N) in [0,1]
    f0:     (B,)
    slopes: (B, K)
    K:      number of segments

    Construction:
        - Segment j covers [j/K, (j+1)/K]
        - On segment j:
              f(x) = f(x_j) + s_j * (x - x_j),
          where x_j = j/K and continuity is enforced by:
              f(x_{j+1}) = f(x_j) + s_j * (1/K)
    """
    B, N = x.shape
    device = x.device

    # Segment index for each x: j = floor(K * x), clipped to [0, K-1]
    seg_idx = torch.clamp((x * K).long(), max=K - 1)  # (B, N)

    # delta_j = slope_j * (1/K)
    delta = slopes / K  # (B, K)

    # Inclusive cumsum of deltas: cumsum[:, j] = sum_{m=0}^j delta_m
    cumsum = torch.cumsum(delta, dim=1)  # (B, K)

    # Exclusive prefix sums: prefix_excl[:, j] = sum_{m=0}^{j-1} delta_m, with prefix_excl[:,0]=0
    zero_col = torch.zeros(B, 1, device=device)
    prefix_excl = torch.cat([zero_col, cumsum[:, :-1]], dim=1)  # (B, K)

    # f(x_j) = f0 + prefix_excl[:, j]
    start_vals = f0.unsqueeze(1) + prefix_excl  # (B, K)

    # Gather start_vals and slopes at the segment indices
    start_at_seg = start_vals.gather(1, seg_idx)    # (B, N)
    slope_at_seg = slopes.gather(1, seg_idx)        # (B, N)

    # Local coordinate in each segment: x - j/K
    seg_idx_float = seg_idx.to(x.dtype) / K         # (B, N)
    local_x = x - seg_idx_float                     # (B, N)

    fx = start_at_seg + slope_at_seg * local_x      # (B, N)
    return fx


def sample_episode_batch(
    batch_size,
    n_ctx,
    K,
    sigma_f0,
    sigma_s,
    noise_std,
    device,
):
    """
    Sample a batch of episodes for the in-context learning task.

    Per episode:
        - Sample random piecewise-linear f
        - Sample context points x_i, noisy y_i = f(x_i) + noise
        - Sample query x*, target y* = f(x*) (clean target)
        - Return sequence X: (B, T, d_in=1) with tokens [x1, y1, ..., xn, yn, x*],
          and targets y_query: (B,)
    """
    # Random function parameters
    f0, slopes = generate_piecewise_linear_params(batch_size, K, sigma_f0, sigma_s, device)

    # Context points
    x_ctx = torch.rand(batch_size, n_ctx, device=device)              # (B, n_ctx)
    y_ctx_clean = eval_piecewise_linear(x_ctx, f0, slopes, K)        # (B, n_ctx)
    y_ctx = y_ctx_clean + noise_std * torch.randn_like(y_ctx_clean)  # (B, n_ctx)

    # Query point
    x_q = torch.rand(batch_size, 1, device=device)                    # (B, 1)
    y_q = eval_piecewise_linear(x_q, f0, slopes, K)                   # (B, 1), clean
    y_q = y_q.squeeze(-1)                                             # (B,)

    # Tokens: [x1, y1, x2, y2, ..., xn, yn, x*]
    # Flatten context pairs along the sequence dimension
    seq_list = []
    for i in range(n_ctx):
        seq_list.append(x_ctx[:, i:i+1])  # (B, 1)
        seq_list.append(y_ctx[:, i:i+1])  # (B, 1)
    seq_list.append(x_q)                  # (B, 1) query x*

    X = torch.cat(seq_list, dim=1)        # (B, T=2*n_ctx+1)
    X = X.unsqueeze(-1)                   # (B, T, d_in=1)

    return X, y_q


# -----------------------------
#  Causal mask
# -----------------------------

def causal_mask(batch_size, T, device):
    """
    Returns a causal mask for attention: mask[b, 0, i, j] = 1 if j <= i else 0.

    Shape: (B, 1, T, T)
    """
    base = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))  # (T, T)
    mask = base.unsqueeze(0).unsqueeze(1)  # (1, 1, T, T)
    mask = mask.expand(batch_size, 1, T, T)
    return mask


# -----------------------------
#  Model wrapper
# -----------------------------

class ICLModel(nn.Module):
    def __init__(self, mha:nn.Module):
        super().__init__()
        self.mha = mha

    # def __init__(self, d_heads, d_head, d_in, d_out):
    #     super().__init__()
    #     d_model = d_heads * d_head
    #     assert d_in == 1, "This ICLModel is designed for d_in=1 tokens."
    #     self.mha = MultiHeadAttention(
    #         d_model=d_model,
    #         d_heads=d_heads,
    #         d_in=d_in,
    #         d_out=d_out,
    #     )

    def forward(self, X):
        """
        X: (B, T, d_in=1)
        Returns prediction for the last token (query): shape (B,)
        """
        B, T, _ = X.shape
        mask = causal_mask(B, T, X.device)
        out = self.mha(X, mask=mask)   # (B, T, d_out=1)
        y_hat = out[:, -1, 0]          # scalar at query token
        return y_hat


def make_naive_head_pruned_model(mha_orig: MultiHeadAttention, d_stop, device):
    """
    Construct a new MultiHeadAttention by randomly selecting d_stop heads from
    mha_orig and discarding the others.
    """
    assert isinstance(mha_orig, MultiHeadAttention)
    d_in = mha_orig.d_in
    d_out = mha_orig.d_out
    d_head = mha_orig.d_head
    H_orig = mha_orig.d_heads

    assert d_stop <= H_orig, "d_stop must be <= original number of heads"

    # Randomly choose d_stop head indices
    head_indices = torch.randperm(H_orig, device=device)[:d_stop]
    head_indices, _ = torch.sort(head_indices)
    head_indices = head_indices.tolist()

    # Map head indices to row/column indices in the flattened d_model dimension
    rows = []
    for h in head_indices:
        start = h * d_head
        end = start + d_head
        rows.extend(list(range(start, end)))
    rows = torch.tensor(rows, device=device, dtype=torch.long)

    # Create a new attention module with d_stop heads
    mha_new = MultiHeadAttention(
        d_model=d_stop * d_head,
        d_heads=d_stop,
        d_in=d_in,
        d_out=d_out,
    ).to(device)

    with torch.no_grad():
        # W_q, W_k, W_v have shape (d_model, d_in)
        mha_new.W_q.weight.copy_(mha_orig.W_q.weight[rows, :])
        mha_new.W_k.weight.copy_(mha_orig.W_k.weight[rows, :])
        mha_new.W_v.weight.copy_(mha_orig.W_v.weight[rows, :])

        # W_o has shape (d_out, d_model)
        mha_new.W_o.weight.copy_(mha_orig.W_o.weight[:, rows])

    return mha_new


# -----------------------------
#  Training / evaluation helpers
# -----------------------------

def train_one_epoch(
    model,
    optimizer,
    criterion,
    batch_size,
    n_ctx,
    K,
    sigma_f0,
    sigma_s,
    noise_std,
    device,
    steps_per_epoch,
):
    model.train()
    total_loss = 0.0

    for _ in range(steps_per_epoch):
        X, y = sample_episode_batch(
            batch_size=batch_size,
            n_ctx=n_ctx,
            K=K,
            sigma_f0=sigma_f0,
            sigma_s=sigma_s,
            noise_std=noise_std,
            device=device,
        )  # X:(B,T,1), y:(B,)

        y_hat = model(X)
        loss = criterion(y_hat, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / steps_per_epoch
    return avg_loss


def evaluate(
    model,
    criterion,
    batch_size,
    n_ctx,
    K,
    sigma_f0,
    sigma_s,
    noise_std,
    device,
    num_batches,
):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(num_batches):
            X, y = sample_episode_batch(
                batch_size=batch_size,
                n_ctx=n_ctx,
                K=K,
                sigma_f0=sigma_f0,
                sigma_s=sigma_s,
                noise_std=noise_std,
                device=device,
            )
            y_hat = model(X)
            loss = criterion(y_hat, y)
            total_loss += loss.item()

    avg_loss = total_loss / num_batches
    return avg_loss


def evaluate_three(
    model_orig,
    model_cp,
    model_naive,
    criterion,
    batch_size,
    n_ctx,
    K,
    sigma_f0,
    sigma_s,
    noise_std,
    device,
    num_batches,
):
    """
    Evaluate three models on the same sampled episodes, returning their average losses.
    """
    model_orig.eval()
    model_cp.eval()
    model_naive.eval()

    total_orig = 0.0
    total_cp = 0.0
    total_naive = 0.0

    with torch.no_grad():
        for _ in range(num_batches):
            X, y = sample_episode_batch(
                batch_size=batch_size,
                n_ctx=n_ctx,
                K=K,
                sigma_f0=sigma_f0,
                sigma_s=sigma_s,
                noise_std=noise_std,
                device=device,
            )
            y_hat_orig = model_orig(X)
            y_hat_cp = model_cp(X)
            y_hat_naive = model_naive(X)

            loss_orig = criterion(y_hat_orig, y)
            loss_cp = criterion(y_hat_cp, y)
            loss_naive = criterion(y_hat_naive, y)

            total_orig += loss_orig.item()
            total_cp += loss_cp.item()
            total_naive += loss_naive.item()

    avg_orig = total_orig / num_batches
    avg_cp = total_cp / num_batches
    avg_naive = total_naive / num_batches
    return avg_orig, avg_cp, avg_naive


def bptrain(
    device,
    K,
    sigma_f0,
    sigma_s,
    noise_std,
    n_ctx,
    d_in,
    d_out,
    d_heads,
    d_head,
    epochs,
    batch_size,
    steps_per_epoch,
    test_batches,
    lr,
    d_stop,
    k,
    tol,
):
    """
    Run the full training process for three models derived from one initialization:
        - model_orig: full MHA with d_heads heads
        - model_cp:   compressed via compress_nn
        - model_naive:randomly keep d_stop heads

    Returns:
        (train_orig, train_cp, train_naive,
         test_orig,  test_cp,  test_naive)
    where each is a list of length epochs+1 (including epoch 0).
    """
    # Base model initialization
    mha_orig = MultiHeadAttention(
        d_model=d_heads * d_head,
        d_heads=d_heads,
        d_in=d_in,
        d_out=d_out,
    )

    # Compressed model via user compression algorithm
    mha_cp, weights = compress_mha(
        mha_orig, dstop=d_stop, k=k, tol=tol
    )

    # Naive head-pruned model
    mha_naive = make_naive_head_pruned_model(
        mha_orig, d_stop=d_stop, device=device
    )

    model_orig = ICLModel(mha_orig).to(device)
    model_cp = ICLModel(mha_cp).to(device)
    model_naive = ICLModel(mha_naive).to(device)

    # Separate optimizers and shared criterion
    optimizer_orig = optim.Adam(model_orig.parameters(), lr=lr)
    optimizer_cp = optim.Adam(model_cp.parameters(), lr=lr)
    optimizer_naive = optim.Adam(model_naive.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_orig_losses = []
    train_cp_losses = []
    train_naive_losses = []

    test_orig_losses = []
    test_cp_losses = []
    test_naive_losses = []

    # ---- Evaluate at epoch 0 (before training) ----
    train_orig_0, train_cp_0, train_naive_0 = evaluate_three(
        model_orig=model_orig,
        model_cp=model_cp,
        model_naive=model_naive,
        criterion=criterion,
        batch_size=batch_size,
        n_ctx=n_ctx,
        K=K,
        sigma_f0=sigma_f0,
        sigma_s=sigma_s,
        noise_std=noise_std,
        device=device,
        num_batches=test_batches,
    )
    test_orig_0, test_cp_0, test_naive_0 = evaluate_three(
        model_orig=model_orig,
        model_cp=model_cp,
        model_naive=model_naive,
        criterion=criterion,
        batch_size=batch_size,
        n_ctx=n_ctx,
        K=K,
        sigma_f0=sigma_f0,
        sigma_s=sigma_s,
        noise_std=noise_std,
        device=device,
        num_batches=test_batches,
    )

    train_orig_losses.append(train_orig_0)
    train_cp_losses.append(train_cp_0)
    train_naive_losses.append(train_naive_0)

    test_orig_losses.append(test_orig_0)
    test_cp_losses.append(test_cp_0)
    test_naive_losses.append(test_naive_0)

    # ---- Training loop ----
    for epoch in range(1, epochs + 1):
        # training mode
        model_orig.train()
        model_cp.train()
        model_naive.train()

        total_train_orig = 0.0
        total_train_cp = 0.0
        total_train_naive = 0.0

        for _ in range(steps_per_epoch):
            X, y = sample_episode_batch(
                batch_size=batch_size,
                n_ctx=n_ctx,
                K=K,
                sigma_f0=sigma_f0,
                sigma_s=sigma_s,
                noise_std=noise_std,
                device=device,
            )  # X:(B,T,1), y:(B,)

            # model_orig update
            optimizer_orig.zero_grad()
            y_hat_orig = model_orig(X)
            loss_orig = criterion(y_hat_orig, y)
            loss_orig.backward()
            optimizer_orig.step()
            total_train_orig += loss_orig.item()

            # model_cp update
            optimizer_cp.zero_grad()
            y_hat_cp = model_cp(X)
            loss_cp = criterion(y_hat_cp, y)
            loss_cp.backward()
            optimizer_cp.step()
            total_train_cp += loss_cp.item()

            # model_naive update
            optimizer_naive.zero_grad()
            y_hat_naive = model_naive(X)
            loss_naive = criterion(y_hat_naive, y)
            loss_naive.backward()
            optimizer_naive.step()
            total_train_naive += loss_naive.item()

        # Average train losses for this epoch
        avg_train_orig = total_train_orig / steps_per_epoch
        avg_train_cp = total_train_cp / steps_per_epoch
        avg_train_naive = total_train_naive / steps_per_epoch

        train_orig_losses.append(avg_train_orig)
        train_cp_losses.append(avg_train_cp)
        train_naive_losses.append(avg_train_naive)

        # Evaluate test losses on shared episodes
        test_orig, test_cp, test_naive = evaluate_three(
            model_orig=model_orig,
            model_cp=model_cp,
            model_naive=model_naive,
            criterion=criterion,
            batch_size=batch_size,
            n_ctx=n_ctx,
            K=K,
            sigma_f0=sigma_f0,
            sigma_s=sigma_s,
            noise_std=noise_std,
            device=device,
            num_batches=test_batches,
        )

        test_orig_losses.append(test_orig)
        test_cp_losses.append(test_cp)
        test_naive_losses.append(test_naive)

        print(
            f"Epoch {epoch:3d} | "
            f"train_orig = {avg_train_orig:.4e}, "
            f"train_cp = {avg_train_cp:.4e}, "
            f"train_naive = {avg_train_naive:.4e} | "
            f"test_orig = {test_orig:.4e}, "
            f"test_cp = {test_cp:.4e}, "
            f"test_naive = {test_naive:.4e}"
        )

    return (
        train_orig_losses,
        train_cp_losses,
        train_naive_losses,
        test_orig_losses,
        test_cp_losses,
        test_naive_losses,
    )


if __name__ == "__main__":
    # ---- Constants (only here) ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    # Function prior
    K = 16             # number of linear segments
    sigma_f0 = 0.5     # std of initial value
    sigma_s = 1.0      # std of slopes
    noise_std = 0.3    # observation noise for y_i

    # ICL / data
    n_ctx = 8          # number of context points per episode
    d_in = 1
    d_out = 1

    # Model
    d_heads = 4000
    d_head = 2
    d_stop = 800
    k = 3
    tol = 1e-6

    # Training hyperparameters
    epochs = 50
    batch_size = 256
    steps_per_epoch = 5
    test_batches = 10
    lr = 1e-4
    save_csv = True
    save_pdf = True
    filename = f"icl_piecewise_losses_d{d_heads*d_head}_dstop{d_stop}_lr{lr}_ep{epochs}_spe{steps_per_epoch}"

    (
        train_orig,
        train_cp,
        train_naive,
        test_orig,
        test_cp,
        test_naive,
    ) = bptrain(
        device=device,
        K=K,
        sigma_f0=sigma_f0,
        sigma_s=sigma_s,
        noise_std=noise_std,
        n_ctx=n_ctx,
        d_in=d_in,
        d_out=d_out,
        d_heads=d_heads,
        d_head=d_head,
        epochs=epochs,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        test_batches=test_batches,
        lr=lr,
        d_stop=d_stop,
        k=k,
        tol=tol,
    )

    # ---- Save losses and plot ----
    epoch_range = list(range(0, epochs + 1))
    train_loss_orig = train_orig
    test_loss_orig = test_orig
    train_loss_cp = train_cp
    test_loss_cp = test_cp
    train_loss_naive = train_naive
    test_loss_naive = test_naive
    d = d_heads * d_head
    dstop = d_stop

    if save_csv:
        with open(filename + ".csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "epoch",
                    "train_loss_orig",
                    "test_loss_orig",
                    "train_loss_cp",
                    "test_loss_cp",
                    "train_loss_naive",
                    "test_loss_naive",
                ]
            )
            for i in range(len(epoch_range)):
                writer.writerow(
                    [
                        epoch_range[i],
                        train_loss_orig[i],
                        test_loss_orig[i],
                        train_loss_cp[i],
                        test_loss_cp[i],
                        train_loss_naive[i],
                        test_loss_naive[i],
                    ]
                )

    if save_pdf:
        fig, axs = make_canvas(rows=2, cols=1, axes_width_pt=300)

        axs[0].plot(
            epoch_range,
            train_loss_cp,
            color="tab:orange",
            marker=None,
            markersize=2,
            label=f"Compressed d'={dstop}",
        )
        axs[0].plot(
            epoch_range,
            train_loss_naive,
            color="tab:blue",
            marker=None,
            markersize=2,
            label=f"Naive d'={dstop}",
        )
        axs[0].plot(
            epoch_range,
            train_loss_orig,
            color="tab:green",
            marker=None,
            markersize=2,
            ls="--",
            label=f"Original d={d}",
        )
        axs[0].set_ylabel("Train loss")
        axs[0].set_yscale("log")
        axs[0].legend()

        axs[1].plot(
            epoch_range,
            test_loss_cp,
            color="tab:orange",
            marker=None,
            markersize=2,
            label=f"Compressed d'={dstop}",
        )
        axs[1].plot(
            epoch_range,
            test_loss_naive,
            color="tab:blue",
            marker=None,
            markersize=2,
            label=f"Naive d'={dstop}",
        )
        axs[1].plot(
            epoch_range,
            test_loss_orig,
            color="tab:green",
            marker=None,
            markersize=2,
            ls="--",
            label=f"Original d={d_heads}",
        )
        axs[1].set_ylabel("Test loss")
        axs[1].set_xlabel("Epoch")
        axs[1].set_yscale("log")

        plt.tight_layout()
        plt.savefig(filename + ".pdf", format="pdf", bbox_inches="tight", pad_inches=0)
        plt.show()
