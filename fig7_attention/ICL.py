#!/usr/bin/env python3
import math
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from common import fix_random_seed, compress_nn, make_canvas
from mha import MultiHeadAttention, MultiHeadAttentionW, compress_mha  # your implementation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)


# ---- Data: random smooth 1D functions f(x) via mixture of Gaussians ----

def sample_random_functions(batch_size, K=4):
    """
    Returns parameters for a batch of random functions f(x):
        f(x) = sum_k a_k * exp(-(x - mu_k)^2 / (2 sigma_k^2)) + b0 + b1*x
    """
    # Shape: (B, K)
    a = torch.randn(batch_size, K, device=device) * 0.8
    mu = torch.rand(batch_size, K, device=device)  # in [0,1]
    sigma = torch.rand(batch_size, K, device=device) * 0.25 + 0.05  # avoid tiny sigmas

    # Linear trend
    b0 = torch.randn(batch_size, 1, device=device) * 0.5
    b1 = torch.randn(batch_size, 1, device=device) * 0.5

    return a, mu, sigma, b0, b1


def eval_functions(x, a, mu, sigma, b0, b1):
    """
    x: (B, N)
    a, mu, sigma: (B, K)
    b0, b1: (B, 1)
    returns f(x) with shape (B, N)
    """
    B, N = x.shape
    K = a.shape[1]

    # Expand for broadcasting
    x_exp = x.unsqueeze(-1)            # (B, N, 1)
    a_exp = a.unsqueeze(1)            # (B, 1, K)
    mu_exp = mu.unsqueeze(1)          # (B, 1, K)
    sigma_exp = sigma.unsqueeze(1)    # (B, 1, K)

    gauss = torch.exp(- (x_exp - mu_exp) ** 2 / (2.0 * sigma_exp ** 2))  # (B, N, K)
    mix = (a_exp * gauss).sum(dim=-1)                                     # (B, N)

    fx = mix + b0 + b1 * x                                                # (B, N)
    return fx


def sample_episode_batch(batch_size, n_ctx, noise_std=0.05):
    """
    Sample a batch of episodes:

    - Sample random functions f
    - Sample context points x_i, noisy y_i = f(x_i) + noise
    - Sample query x*, target y* = f(x*) (no noise)
    - Return tokens X: (B, T=n_ctx+1, d_in=2) with tokens (x_i, y_i) and (x*, 0),
      and targets y_query: (B,)
    """
    # Random function params
    a, mu, sigma, b0, b1 = sample_random_functions(batch_size)

    # Context points
    x_ctx = torch.rand(batch_size, n_ctx, device=device)
    y_ctx_clean = eval_functions(x_ctx, a, mu, sigma, b0, b1)
    y_ctx = y_ctx_clean + noise_std * torch.randn_like(y_ctx_clean)

    # Query point
    x_q = torch.rand(batch_size, 1, device=device)
    y_q = eval_functions(x_q, a, mu, sigma, b0, b1)  # clean (no noise)
    y_q = y_q.squeeze(-1)  # (B,)

    # Tokens: (x_i, y_i) for context; (x_q, 0) for query
    z_ctx = torch.stack([x_ctx, y_ctx], dim=-1)  # (B, n_ctx, 2)
    z_q = torch.stack(
        [x_q.squeeze(-1), torch.zeros_like(x_q).squeeze(-1)],
        dim=-1,
    )  # (B, 2)
    z_q = z_q.unsqueeze(1)  # (B, 1, 2)

    X = torch.cat([z_ctx, z_q], dim=1)  # (B, T=n_ctx+1, 2)
    return X, y_q


# ---- Causal mask ----

def causal_mask(batch_size, T):
    """
    Returns causal mask of shape (B, 1, T, T)
    mask[b,0,i,j] = 1 if j <= i, else 0
    """
    mask = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))
    mask = mask.unsqueeze(0).unsqueeze(1)  # (1, 1, T, T)
    mask = mask.expand(batch_size, -1, -1, -1)  # (B, 1, T, T)
    return mask


# ---- Model ----

class ICLModel(nn.Module):
    def __init__(self, d_heads=512, d_head=2, d_in=2, d_out=1):
        super().__init__()
        d_model = d_heads * d_head
        self.mha = MultiHeadAttention(
            d_model=d_model,
            d_heads=d_heads,
            d_in=d_in,
            d_out=d_out,
        )

    def forward(self, X):
        """
        X: (B, T, d_in=2)
        Returns prediction for the last token (query): shape (B,)
        """
        B, T, _ = X.shape
        mask = causal_mask(B, T)
        out = self.mha(X, mask=mask)   # (B, T, d_out=1)
        y_hat = out[:, -1, 0]          # scalar at last token
        return y_hat


# ---- Training loop ----

def train_icl(
    epochs=50,
    batch_size=64,
    n_ctx=8,
    d_heads=512,   
    d_head=2,
    lr=1e-3,
    train_steps_per_epoch=200,
    test_batches=100,
):
    model = ICLModel(d_heads=d_heads, d_head=d_head, d_in=2, d_out=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_losses = []
    test_losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_train_loss = 0.0

        for _ in range(train_steps_per_epoch):
            X, y = sample_episode_batch(batch_size, n_ctx)  # X:(B,T,2), y:(B,)
            y_hat = model(X)
            loss = criterion(y_hat, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        epoch_train_loss /= train_steps_per_epoch
        train_losses.append(epoch_train_loss)

        # Evaluation
        model.eval()
        with torch.no_grad():
            epoch_test_loss = 0.0
            for _ in range(test_batches):
                X, y = sample_episode_batch(batch_size, n_ctx)
                y_hat = model(X)
                loss = criterion(y_hat, y)
                epoch_test_loss += loss.item()
            epoch_test_loss /= test_batches
            test_losses.append(epoch_test_loss)

        print(
            f"Epoch {epoch:3d} | "
            f"train MSE = {epoch_train_loss:.4e} | "
            f"test MSE = {epoch_test_loss:.4e}"
        )

    return model, train_losses, test_losses


if __name__ == "__main__":
    save_csv = True
    save_pdf = True
    seed = 42
    fix_random_seed(seed)

    epochs = 50
    epochs = epochs
    batch_size = 256 
    n_ctx = 16
    d_heads = 64 
    d_head = 2

    train_steps_per_epoch = 100
    test_batches = 100

    algo_name = 'Adam'
    lr = 1e-3
    algo = torch.optim.Adam


    # construct the models
    model_orig = MultiHeadAttention(
        d_model=d_heads * d_head,
        d_heads=d_heads,
        d_in=2,
        d_out=1,
    ).to(device)

    
    model, train_losses, test_losses = train_icl(
        epochs=epochs,
        batch_size=batch_size,
        n_ctx=n_ctx,
        d_heads=d_heads, 
        d_head=d_head,
        lr=lr,
        train_steps_per_epoch=train_steps_per_epoch,
        test_batches=test_batches,
    )

    # Plot train/test loss vs epoch
    xs = list(range(1, epochs + 1))
    plt.figure()
    plt.plot(xs, train_losses, label="Train MSE")
    plt.plot(xs, test_losses, label="Test MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.show()