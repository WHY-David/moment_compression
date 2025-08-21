import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler, RandomSampler

from common import TwoLayerNet, fix_random_seed, make_canvas
from data_gen import generate_train_data

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from compressor import Compressor

device = torch.device("cpu")

def make_loader(data, num_samples:int, weights=None, batch_size=64):
    X = torch.from_numpy(data[:, :2]).float().to(device)      # shape (N, 2) -> (x, y)
    y = torch.from_numpy(data[:, [2]]).float().to(device)     # shape (N, 1) -> f(x, y)
    ds = TensorDataset(X, y)
    if weights is None:
        sampler = RandomSampler(
            data_source=ds,
            replacement=True,
            num_samples=num_samples
        )
    else:
        weights_t = torch.tensor(weights, dtype=torch.float, device=device)
        sampler = WeightedRandomSampler(
            weights=weights_t,
            num_samples=num_samples,
            replacement=True
        )
    return DataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler
    )

def make_test_loader(data, batch_size=64):
    X = torch.from_numpy(data[:, :2]).float().to(device)      # shape (N, 2) -> (x, y)
    y = torch.from_numpy(data[:, [2]]).float().to(device)     # shape (N, 1) -> f(x, y)
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=batch_size)

def compute_loss(net, loader, loss_fn):
    net.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
    return total_loss / len(loader.sampler)

def bptrain(train_loader, test_loader, hidden_dim, epochs, lr):
    net = TwoLayerNet(2,hidden_dim).to(device)
    loss_fn = nn.MSELoss()
    opt = torch.optim.SGD(net.parameters(), lr=lr)

    train_losses = [compute_loss(net, train_loader, loss_fn)]
    test_losses = [compute_loss(net, test_loader, loss_fn)]

    # Training loop
    for epoch in range(1, epochs+1):
        net.train()
        # total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            opt.zero_grad()
            outputs = net(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            opt.step()
            # total_loss += loss.item() * inputs.size(0)
        # train_loss = total_loss / len(train_loader.sampler)
        train_loss = compute_loss(net, train_loader, loss_fn)
        train_losses.append(train_loss)
        test_loss = compute_loss(net, test_loader, loss_fn)
        test_losses.append(test_loss)
        print(f"Epoch {epoch}/{epochs}. Train loss: {train_loss:.3e}, test loss: {test_loss:.3e}")

    return train_losses, test_losses

if __name__ == "__main__":
    d = 100_000
    dstop = 500
    k = 3
    test_size = 10_000
    hidden_dim = 200
    lr = 1e-3
    epochs = 100
    batch_size = d
    seed = 42

    train_data = generate_train_data(d, noise=0.0, seed=seed, return_tensor=False, device=device)
    cp = Compressor(train_data)
    c_, train_cp = cp.compress(3, dstop=dstop, print_progress=True)
    print(f"Compression completed. d={d} -> d'={dstop}")
    train_naive = train_data[:dstop, :]
    test_data = generate_train_data(d, noise=0.0, seed=seed*10, return_tensor=False, device=device)

    train_loader = make_loader(train_data, d, batch_size=batch_size)
    train_loader_cp = make_loader(train_cp, d, weights=c_, batch_size=batch_size)
    train_loader_naive = make_loader(train_naive, d, batch_size=batch_size)
    test_loader = make_test_loader(test_data, batch_size=batch_size)

    # plots
    epochs_range = list(range(epochs+1))
    fig, axs = plt.subplots(2, 1, figsize=(6, 4), sharex=True)

    # train on the full data
    fix_random_seed(seed)
    train_losses, test_losses = bptrain(train_loader, test_loader, hidden_dim, epochs, lr)
    axs[0].plot(epochs_range, train_losses, marker='o', label=f"d={d}")
    axs[1].plot(epochs_range, test_losses, marker='o', label=f"d={d}")

    # train on compressed data
    fix_random_seed(seed)
    train_losses, test_losses = bptrain(train_loader_cp, test_loader, hidden_dim, epochs, lr)
    axs[0].plot(epochs_range, train_losses, marker='o', label=f"cp d={dstop}")
    axs[1].plot(epochs_range, test_losses, marker='o', label=f"cp d={dstop}")

    # train on naive subset
    fix_random_seed(seed)
    train_losses, test_losses = bptrain(train_loader_naive, test_loader, hidden_dim, epochs, lr)
    axs[0].plot(epochs_range, train_losses, marker='o', label=f"d={dstop}")
    axs[1].plot(epochs_range, test_losses, marker='o', label=f"d={dstop}")

    # final adjustments to the plot
    axs[0].set_yscale('log')
    axs[0].set_ylabel('Train MSE')
    axs[0].grid(True)
    axs[0].legend()
    axs[1].set_yscale('log')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Test MSE')
    # axs[1].grid(True)
    # axs[1].legend()
    plt.tight_layout()
    plt.savefig(f"compress_trainds.pdf", format='pdf', bbox_inches='tight', pad_inches=0)
    # plt.show()
