import numpy as np

import random
import matplotlib.pyplot as plt
import torch
from torch.nn import MSELoss

from data_gen import generate_train_data
from common import TwoLayerNet, fix_random_seed

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

def f(x,y):
    return (x-y)%1.


def compute_loss(net, loader, loss_fn):
    net.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
    return total_loss / len(loader.dataset)

def bptrain(train_loader, test_loader, epochs, lr):
    net = TwoLayerNet(2,256).to(device)
    loss_fn = MSELoss()
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    train_losses = [compute_loss(net, train_loader, loss_fn)]
    test_losses = [compute_loss(net, test_loader, loss_fn)]

    # Training loop
    for epoch in range(1, epochs+1):
        net.train()
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            opt.zero_grad()
            outputs = net(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            opt.step()
            total_loss += loss.item() * inputs.size(0)
        train_loss = total_loss / len(train_ds)
        train_losses.append(train_loss)
        test_loss = compute_loss(net, test_loader, loss_fn)
        test_losses.append(test_loss)
        print(f"Epoch {epoch}/{epochs}. Train loss: {train_loss:.3f}, test loss: {test_loss:.3f}")

    return train_losses, test_losses



if __name__ == '__main__':
    d = 100
    epochs = 200
    lr = 1e-3
    seed = 0
    batch_size = min(200, d//2)

    train_data = generate_train_data(d, func=f, noise=0, seed=0)
    train_inputs = torch.from_numpy(train_data[:, :2]).float().to(device)
    train_labels = torch.from_numpy(train_data[:, 2:]).float().to(device)
    train_ds = torch.utils.data.TensorDataset(train_inputs, train_labels)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=False)

    test_data = generate_train_data(10000, func=f, noise=0, seed=seed+42)
    test_inputs = torch.from_numpy(test_data[:, :2]).float().to(device)
    test_labels = torch.from_numpy(test_data[:, 2:]).float().to(device)
    test_ds = torch.utils.data.TensorDataset(test_inputs, test_labels)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # plots
    epochs_range = list(range(0, epochs+1))
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # train on the original data
    fix_random_seed(seed)
    train_losses, test_losses = bptrain(train_loader, test_loader, epochs, lr)
    axs[0].plot(epochs_range, train_losses, marker='o', label=f"d={len(train_ds)}")
    axs[1].plot(epochs_range, test_losses, marker='o', label=f"d={len(train_ds)}")

    # final adjustments to the plot
    # axs[0].set_yscale('log')
    axs[0].set_ylabel('Train MSE')
    axs[0].grid(True)
    axs[0].legend()
    # axs[1].set_yscale('log')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Test MSE')
    axs[1].grid(True)
    # axs[1].legend()
    plt.tight_layout()
    plt.savefig(f"train_on_{d}.pdf", format='pdf', bbox_inches='tight', pad_inches=0)
    plt.show()
