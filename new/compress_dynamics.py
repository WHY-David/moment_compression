import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from matplotlib import pyplot as plt
plt.rc('font', family='Helvetica', size=8)

from scipy.special import jv

from common import TwoLayerNet, WeightedTwoLayerNet, fix_random_seed, compress_nn, make_canvas, Sin
from data_gen import generate_train_data

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from compressor import Compressor

# Device configuration
# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
device = torch.device('cpu')


def cyl_harmonic(x, y):
    # Convert torch tensors to numpy arrays if needed
    if torch.is_tensor(x):
        x_np = x.cpu().numpy()
    else:
        x_np = np.array(x)
    if torch.is_tensor(y):
        y_np = y.cpu().numpy()
    else:
        y_np = np.array(y)

    r = np.sqrt(x_np**2 + y_np**2)
    theta = np.arctan2(y_np, x_np)
    result = jv(6, 20 * r) * np.sin(6 * theta)

    # Convert result back to torch tensor, preserving dtype and device if possible
    if torch.is_tensor(x):
        result = torch.from_numpy(result).to(x.device).type_as(x)
    else:
        result = torch.from_numpy(result)
    return result

def make_loader(dataset, batch_size=64, seed=0):
    """Deterministic DataLoader using a fixed shuffled index order."""
    n = len(dataset)
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    sampler = torch.utils.data.SubsetRandomSampler(indices.tolist())
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=False)
    return loader

def compute_loss(net, loader, loss_fn=nn.MSELoss()):
    """
    Evaluate average loss over `loader` without altering the caller's train/eval mode.
    """
    was_training = net.training  # remember current mode
    net.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = net(inputs)
            loss = loss_fn(outputs, labels)
            bsz = inputs.size(0)
            total_loss += loss.item() * bsz
            total_count += bsz
    # restore original mode
    if was_training:
        net.train()
    else:
        net.eval()
    return total_loss / total_count

def train_orig(net: TwoLayerNet,
               train_ds: TensorDataset,
               test_ds: TensorDataset,
               epochs=5,
               batch_size=64,
               seed=0,
               algo=torch.optim.SGD, 
               **opt_params):
    """Train the baseline network. Returns (train_losses, test_losses) with snapshots per epoch (incl. epoch 0)."""
    fix_random_seed(seed)
    train_loader = make_loader(train_ds, batch_size=batch_size, seed=seed)
    test_loader = make_loader(test_ds, batch_size=batch_size, seed=seed)
    loss_fn = nn.MSELoss()
    opt = algo(net.parameters(), **opt_params)

    # Initial evaluation (does not change mode due to compute_loss restoration)
    train_losses = [compute_loss(net, train_loader, loss_fn)]
    test_losses = [compute_loss(net, test_loader, loss_fn)]

    for epoch in range(epochs):
        # ensure we're in training mode (compute_loss may have put us in eval temporarily)
        net.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(net(x), y)
            loss.backward()
            opt.step()

        # End-of-epoch evaluation (compute_loss preserves/restores mode)
        train_losses.append(compute_loss(net, train_loader, loss_fn))
        test_losses.append(compute_loss(net, test_loader, loss_fn))
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} completed")
    return train_losses, test_losses


def train_weighted(net: WeightedTwoLayerNet,
               train_ds: TensorDataset,
               test_ds: TensorDataset,
               epochs=5,
               batch_size=64,
               seed=0,
               algo=torch.optim.SGD, **opt_params):
    fix_random_seed(seed)
    train_loader = make_loader(train_ds, batch_size=batch_size, seed=seed)
    test_loader = make_loader(test_ds, batch_size=batch_size, seed=seed)
    loss_fn = nn.MSELoss()
    opt = algo(net.parameters(), **opt_params)

    weights_t = net.weights
    inv_c = torch.where(weights_t > tol,
                        1.0 / weights_t,
                        torch.zeros_like(weights_t))

    # Scale per-element gradients via hooks
    net.fc1.weight.register_hook(lambda grad: grad * inv_c.view(-1, 1))
    net.fc1.bias.register_hook(lambda grad: grad * inv_c)
    net.fc2.weight.register_hook(lambda grad: grad * inv_c.view(1, -1))

    # Initial evaluation (does not change mode due to compute_loss restoration)
    train_losses = [compute_loss(net, train_loader, loss_fn)]
    test_losses = [compute_loss(net, test_loader, loss_fn)]

    for epoch in range(epochs):
        # ensure we're in training mode (compute_loss may have put us in eval temporarily)
        net.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(net(x), y)
            loss.backward()
            opt.step()

        # End-of-epoch evaluation (compute_loss preserves/restores mode)
        train_losses.append(compute_loss(net, train_loader, loss_fn))
        test_losses.append(compute_loss(net, test_loader, loss_fn))
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} completed")
    return train_losses, test_losses



def naive_prune(net_orig:TwoLayerNet, hidden_dim:int):
    input_dim = net_orig.fc1.weight.shape[1]

    # Temporarily disable random init for Linear
    orig_reset = nn.Linear.reset_parameters
    nn.Linear.reset_parameters = lambda self, *args, **kwargs: None
    # Instantiate on same device
    net = TwoLayerNet(input_dim, hidden_dim).to(net_orig.fc1.weight.device)
    # Restore init method
    nn.Linear.reset_parameters = orig_reset

    with torch.no_grad():
        W1 = net_orig.fc1.weight[:hidden_dim, :].clone()
        b1 = net_orig.fc1.bias[:hidden_dim].clone()
        W2 = net_orig.fc2.weight[:, :hidden_dim].clone()
        b2 = net_orig.fc2.bias.clone()

        net.fc1.weight.copy_(W1)
        net.fc1.bias.copy_(b1)
        net.fc2.weight.copy_(W2)
        net.fc2.bias.copy_(b2)
    return net
    


if __name__ == '__main__':
    # Determinism
    seed = 42
    fix_random_seed(seed)

    # Hyperparameters
    d = 10000
    dstop = 1000
    k = 5
    train_size = 100_000
    test_size = train_size
    train_noise = 0.2
    tol = 1e-12
    epochs = 30
    batch_size = 1024

    # # Adam
    # algo_name = 'Adam'
    # lr = 1e-4
    # algo = torch.optim.Adam

    # SGD
    algo_name = 'SGD'
    lr = 1e-4
    algo = torch.optim.SGD

    # TensorDataset
    # net_truth = TwoLayerNet(input_dim=2, hidden_dim=1000, init_uniform=None, activation=nn.ReLU).to(device)
    train_data = generate_train_data(train_size, f=cyl_harmonic, noise=train_noise, seed=seed**2, return_tensor=True, device=device)
    train_ds = TensorDataset(train_data[:, :2], train_data[:, 2:])
    test_data = generate_train_data(test_size, f=cyl_harmonic, noise=0., seed=seed**3, return_tensor=True, device=device)
    test_ds = TensorDataset(test_data[:, :2], test_data[:, 2:])

    # 1) Original network
    net_orig = TwoLayerNet(input_dim=2, hidden_dim=d, init_uniform=None, activation=nn.ReLU).to(device)
    net_cp, weights_t = compress_nn(net_orig, dstop=dstop, k=k, tol=tol)
    print(f'Compression completed. d={d} -> dstop={dstop}')
    net_naive = naive_prune(net_orig, dstop)

    # 3) Train all cases with identical minibatches/order
    train_loss_orig, test_loss_orig = train_orig(net_orig, train_ds, test_ds, epochs=epochs, batch_size=batch_size, seed=seed, algo=algo, lr=lr)
    train_loss_cp, test_loss_cp = train_weighted(net_cp, train_ds, test_ds, epochs=epochs, batch_size=batch_size, lr=lr, seed=seed, algo=algo)
    train_loss_naive, test_loss_naive = train_orig(net_naive, train_ds, test_ds, epochs=epochs, batch_size=batch_size, seed=seed, algo=algo, lr=lr)

    # 5) Plot: three subplots
    fig, axs = make_canvas(rows=2, cols=1, axes_width_pt=300)
    epoch_range = list(range(epochs+1))

    # Plot Train Loss vs. epoch
    axs[0].plot(epoch_range, train_loss_naive, color='tab:blue',  marker=None, markersize=2, label=f'Naive d\'={dstop}')
    axs[0].plot(epoch_range, train_loss_orig,  color='tab:green', marker=None, markersize=2, label=f'Original d={d}')
    axs[0].plot(epoch_range, train_loss_cp, color='tab:orange',marker=None, markersize=2, label=f'Compressed d\'={dstop}')
    axs[0].set_ylabel('Train loss')
    axs[0].set_yscale('log')
    # axs[0].grid(True, linewidth=0.25)
    axs[0].legend()

    # Plot Test Loss vs. epoch
    axs[1].plot(epoch_range, test_loss_naive, color='tab:blue', marker=None, markersize=2, label=f'Naive d\'={dstop}')
    axs[1].plot(epoch_range, test_loss_orig,  color='tab:green', marker=None, markersize=2, label=f'Original d={d}')
    axs[1].plot(epoch_range, test_loss_cp, color='tab:orange', marker=None, markersize=2, label=f'Compressed d\'={dstop}')
    axs[1].set_ylabel('Test loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_yscale('log')
    # axs[1].grid(True, linewidth=0.25)

    plt.tight_layout()

    filename = f'LTH_harm_{algo_name}_d{d}_dstop{dstop}_k{k}_noise{train_noise}_bs{batch_size}_lr{lr}.pdf'
    plt.savefig(filename, format='pdf', bbox_inches='tight', pad_inches=0)
    plt.show()
