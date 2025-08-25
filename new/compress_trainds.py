import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
plt.rc('font', family='Helvetica', size=8)

from common import TwoLayerNet, fix_random_seed, make_canvas
from data_gen import generate_train_data

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from compressor import Compressor

device = torch.device("cpu")

def make_loader(data, batch_size=None):
    X = torch.from_numpy(data[:, :2]).float().to(device)      # shape (N, 2) -> (x, y)
    y = torch.from_numpy(data[:, [2]]).float().to(device)     # shape (N, 1) -> f(x, y)
    ds = TensorDataset(X, y)
    if batch_size is None:
        batch_size = len(ds)  # full-batch
    # Deterministic order for all cases; no sampler and no shuffling.
    return DataLoader(ds, batch_size=batch_size, shuffle=False)

def make_test_loader(data, batch_size=None):
    X = torch.from_numpy(data[:, :2]).float().to(device)      # shape (d, 2) -> (x, y)
    y = torch.from_numpy(data[:, [2]]).float().to(device)     # shape (d, 1) -> f(x, y)
    ds = TensorDataset(X, y)
    if batch_size is None:
        batch_size = len(ds)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)

def compute_loss(net, loader, weights=None):
    """Manual MSE. If weights is None: (1/N) * sum_i ||f(x_i)-y_i||^2.
    If weights given (length N): (1/sum w_i) * sum_i w_i * ||f(x_i)-y_i||^2.
    Assumes full-batch DataLoader (one batch) and deterministic ordering.
    """
    net.eval()
    with torch.no_grad():
        for inputs, labels in loader:  # single full batch
            outputs = net(inputs)
            sq_err = (outputs - labels).pow(2).squeeze(-1)  # shape (N,)
            if weights is None:
                loss = sq_err.mean()
            else:
                w = torch.as_tensor(weights, dtype=torch.float, device=inputs.device).view(-1)
                loss = (w * sq_err).sum() / w.sum()
            return loss.item()

def bptrain(train_loader, test_loader, hidden_dim, epochs, lr, train_weights=None):
    net = TwoLayerNet(2, hidden_dim).to(device)
    opt = torch.optim.SGD(net.parameters(), lr=lr)

    # initial losses
    train_losses = [compute_loss(net, train_loader, weights=train_weights)]
    test_losses = [compute_loss(net, test_loader, weights=None)]

    for epoch in range(1, epochs + 1):
        net.train()
        for inputs, labels in train_loader:  # single full batch
            opt.zero_grad()
            outputs = net(inputs)
            sq_err = (outputs - labels).pow(2).squeeze(-1)
            if train_weights is None:
                loss = sq_err.mean()
            else:
                w = torch.as_tensor(train_weights, dtype=torch.float, device=inputs.device).view(-1)
                loss = (w * sq_err).sum() / w.sum()
            loss.backward()
            opt.step()

        # record losses after the update
        train_loss = compute_loss(net, train_loader, weights=train_weights)
        test_loss = compute_loss(net, test_loader, weights=None)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        print(f"Epoch {epoch}/{epochs}. Train loss: {train_loss:.3e}, test loss: {test_loss:.3e}")

    return train_losses, test_losses

if __name__ == "__main__":
    d = 50_000
    dstop = 10_000
    k = 3
    test_size = 10_000
    hidden_dim = 200
    lr = 2e-3
    epochs = 100
    batch_size = None
    seed = 42

    train_data = generate_train_data(d, noise=0.0, seed=seed**2, return_tensor=False, device=device)
    cp = Compressor(train_data)
    c_, train_cp = cp.compress(3, dstop=dstop, print_progress=True)
    print(f"Compression completed. d={d} -> d'={dstop}")
    train_naive = train_data[:dstop, :]
    test_data = generate_train_data(d, noise=0.0, seed=seed*10, return_tensor=False, device=device)

    train_loader = make_loader(train_data, batch_size=batch_size)
    train_loader_cp = make_loader(train_cp, batch_size=batch_size)
    train_loader_naive = make_loader(train_naive, batch_size=batch_size)
    test_loader = make_test_loader(test_data, batch_size=batch_size)

    # plots
    epochs_range = list(range(epochs+1))
    fig, axs = plt.subplots(2, 1, figsize=(6, 4), sharex=True)

    # train on the full data
    fix_random_seed(seed)
    train_losses, test_losses = bptrain(train_loader, test_loader, hidden_dim, epochs, lr, train_weights=None)
    axs[0].plot(epochs_range, train_losses, marker='o', markersize=3, label=f"d={d}")
    axs[1].plot(epochs_range, test_losses, marker='o', markersize=3, label=f"d={d}")

    # train on compressed data (weighted loss)
    fix_random_seed(seed)
    train_losses, test_losses = bptrain(train_loader_cp, test_loader, hidden_dim, epochs, lr, train_weights=c_)
    axs[0].plot(epochs_range, train_losses, marker='o', markersize=3, label=f"cp d={dstop}")
    axs[1].plot(epochs_range, test_losses, marker='o', markersize=3, label=f"cp d={dstop}")

    # train on naive subset (unweighted loss)
    fix_random_seed(seed)
    train_losses, test_losses = bptrain(train_loader_naive, test_loader, hidden_dim, epochs, lr, train_weights=None)
    axs[0].plot(epochs_range, train_losses, marker='o', markersize=3, label=f"d={dstop}")
    axs[1].plot(epochs_range, test_losses, marker='o', markersize=3, label=f"d={dstop}")

    # final adjustments to the plot
    axs[0].set_yscale('log')
    axs[0].set_ylabel('Train MSE')
    # axs[0].grid(True)
    axs[0].legend()
    axs[1].set_yscale('log')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Test MSE')
    # axs[1].grid(True)
    # axs[1].legend()
    plt.tight_layout()
    plt.savefig(f"compress_trainds.pdf", format='pdf', bbox_inches='tight', pad_inches=0)
    # plt.show()
