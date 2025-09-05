import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
plt.rc('font', family='Helvetica', size=8)
import csv

from common import TwoLayerNet, fix_random_seed, make_canvas, cyl_harmonic
from data_gen import generate_train_data

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from compressor import Compressor

# Device configuration
# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
device = torch.device("cpu")


def make_loader(data, batch_size=None):
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    X = data[:, :2].float().to(device)
    y = data[:, [2]].float().to(device)

    ds = TensorDataset(X, y)
    if batch_size is None:
        batch_size = len(ds)  # full-batch
    # Deterministic order for all cases; no sampler and no shuffling.
    return DataLoader(ds, batch_size=batch_size, shuffle=False)

# def make_test_loader(data, batch_size=None):
#     if isinstance(data, np.ndarray):
#         data = torch.from_numpy(data)
#     X = data[:, :2].float().to(device)
#     y = data[:, [2]].float().to(device)
#     ds = TensorDataset(X, y)
#     if batch_size is None:
#         batch_size = len(ds)
#     return DataLoader(ds, batch_size=batch_size, shuffle=False)

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

def bptrain(train_loader, test_loader, hidden_dim, epochs=5, train_weights=None, seed=0, algo=torch.optim.SGD, **opt_params):
    fix_random_seed(seed)
    net = TwoLayerNet(2, hidden_dim).to(device)
    opt = algo(net.parameters(), **opt_params)

    # initial losses
    train_losses = [compute_loss(net, train_loader, weights=train_weights)]
    test_losses = [compute_loss(net, test_loader, weights=None)]

    if train_weights is not None:
        w = torch.as_tensor(train_weights, dtype=torch.float, device=device).view(-1)

    for epoch in range(1, epochs + 1):
        net.train()
        for inputs, labels in train_loader:  # single full batch
            opt.zero_grad()
            outputs = net(inputs)
            sq_err = (outputs - labels).pow(2).squeeze(-1)
            if train_weights is None:
                loss = sq_err.mean()
            else:
                loss = (w * sq_err).sum() / w.sum()
            loss.backward()
            opt.step()

        # record losses after the update
        train_loss = compute_loss(net, train_loader, weights=train_weights)
        test_loss = compute_loss(net, test_loader, weights=None)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        if epoch % 10 ==0:
            print(f"Epoch {epoch}/{epochs}. Train loss: {train_loss:.3e}, test loss: {test_loss:.3e}")

    return train_losses, test_losses

if __name__ == "__main__":
    seed = 42
    fix_random_seed(seed)

    d = 50_000  # train size
    dstop = 100 # compressed training dataset size
    k = 5
    train_noise = 0.
    test_size = 10_000
    hidden_dim = 200
    epochs = 100
    batch_size = None

    algo_name = 'Adam'
    lr = 5e-3
    algo = torch.optim.Adam

    train_data = generate_train_data(d, f=cyl_harmonic, noise=train_noise, seed=seed**2, return_tensor=True, device=device)
    cp = Compressor(train_data, random_state=seed)
    c_, train_cp = cp.compress(k, dstop=dstop, print_progress=True)
    print(f"Compression completed. d={d} -> d'={dstop}")
    train_naive = train_data[:dstop, :]
    test_data = generate_train_data(d, f=cyl_harmonic, noise=0, seed=seed**3, return_tensor=True, device=device)

    train_loader = make_loader(train_data, batch_size=batch_size)
    train_loader_cp = make_loader(train_cp, batch_size=batch_size)
    train_loader_naive = make_loader(train_naive, batch_size=batch_size)
    test_loader = make_loader(test_data, batch_size=batch_size)

    # train three cases with identical hyperparameters
    train_loss_orig, test_loss_orig = bptrain(train_loader, test_loader, hidden_dim, epochs=epochs, train_weights=None, seed=seed, algo=algo, lr=lr)
    train_loss_cp, test_loss_cp = bptrain(train_loader_cp, test_loader, hidden_dim, epochs=epochs, train_weights=c_, seed=seed, algo=algo, lr=lr)
    train_loss_naive, test_loss_naive = bptrain(train_loader_naive, test_loader, hidden_dim, epochs=epochs, train_weights=None, seed=seed, algo=algo, lr=lr)

    # plots
    epoch_range = list(range(epochs+1))
    fig, axs = make_canvas(rows=2, cols=1, axes_width_pt=300)

    # Plot Train Loss vs. epoch
    axs[0].plot(epoch_range, train_loss_orig,  color='tab:green', marker=None, markersize=2, ls='--', label=f'Original d={d}')
    axs[0].plot(epoch_range, train_loss_cp, color='tab:orange',marker=None, markersize=2, label=f'Compressed d\'={dstop}')
    axs[0].plot(epoch_range, train_loss_naive, color='tab:blue',  marker=None, markersize=2, label=f'Naive d\'={dstop}')
    axs[0].set_ylabel('Train loss')
    axs[0].set_yscale('log')
    # axs[0].grid(True, linewidth=0.25)
    axs[0].legend()

    # Plot Test Loss vs. epoch
    axs[1].plot(epoch_range, test_loss_orig,  color='tab:green', marker=None, markersize=2, ls='--', label=f'Original d={d}')
    axs[1].plot(epoch_range, test_loss_cp, color='tab:orange', marker=None, markersize=2, label=f'Compressed d\'={dstop}')
    axs[1].plot(epoch_range, test_loss_naive, color='tab:blue', marker=None, markersize=2, label=f'Naive d\'={dstop}')
    axs[1].set_ylabel('Test loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_yscale('log')
    # axs[1].grid(True, linewidth=0.25)

    plt.tight_layout()

    filename = f'CPTDS/harm_{algo_name}_d{d}_dstop{dstop}_k{k}_noise{train_noise}_hidden{hidden_dim}_bs{batch_size}_lr{lr}'
    with open(filename + '.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow([
            'epoch',
            'train_loss_orig',
            'test_loss_orig',
            'train_loss_cp',
            'test_loss_cp',
            'train_loss_naive',
            'test_loss_naive'
        ])
        # Write data rows
        for i in range(len(epoch_range)):
            writer.writerow([
                epoch_range[i],
                train_loss_orig[i],
                test_loss_orig[i],
                train_loss_cp[i],
                test_loss_cp[i],
                train_loss_naive[i],
                test_loss_naive[i]
            ])
    plt.savefig(filename+'.pdf', format='pdf', bbox_inches='tight', pad_inches=0)
    plt.show()
