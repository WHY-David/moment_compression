import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from matplotlib import pyplot as plt
plt.rc('font', family='Helvetica', size=8)
import csv

from common import TwoLayerNet, WeightedTwoLayerNet, fix_random_seed, compress_nn, make_canvas, cyl_harmonic
from data_gen import generate_data

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from compressor import Compressor

# Device configuration
# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

def bptrain(net,
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
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=0.)

    if isinstance(net, WeightedTwoLayerNet):
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

    for epoch in range(1, epochs+1):
        # ensure we're in training mode (compute_loss may have put us in eval temporarily)
        net.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(net(x), y)
            loss.backward()
            opt.step()
        sched.step()

        # End-of-epoch evaluation (compute_loss preserves/restores mode)
        if epoch % 10 == 0:
            trl = compute_loss(net, train_loader, loss_fn)
            tel = compute_loss(net, test_loader, loss_fn)
            train_losses.append(trl)
            test_losses.append(tel)
            print(f"Epoch {epoch}/{epochs}. Train loss: {trl:.3e}, test loss: {tel:.3e}")
    return train_losses, test_losses
    


if __name__ == '__main__':
    save_csv = True
    save_pdf = True

    # Determinism
    seed = 42
    fix_random_seed(seed)

    # Hyperparameters
    dlist = [2**n for n in range(8,18)]
    dstop = lambda d: int(16*np.sqrt(d))
    k = 6
    train_size = 1_000_000
    test_size = 100_000
    train_noise = 0.2
    tol = 1e-12
    epochs = 30
    batch_size = 512

    # # Adam
    # algo_name = 'Adam'
    # lr = 5e-5
    # algo = torch.optim.Adam

    # # SGD
    # algo_name = 'SGD'
    # lr = 5e-4
    # algo = torch.optim.SGD

    # AdamW
    algo_name = 'AdamW'
    lr = 1e-3
    algo = torch.optim.AdamW

    task_name = "harm"
    # net_truth = TwoLayerNet(input_dim=2, hidden_dim=1000, init_uniform=None, activation=nn.ReLU).to(device)
    f = lambda x, y: cyl_harmonic(x, y, n=6, k=20)
    # f = lambda x, y: x*y/(x**2+y**2)

    train_data = generate_data(train_size, f=f, noise=train_noise, seed=seed**2, return_tensor=True, device=device)
    train_ds = TensorDataset(train_data[:, :2], train_data[:, 2:])
    test_data = generate_data(test_size, f=f, noise=0., seed=seed**3, return_tensor=True, device=device)
    test_ds = TensorDataset(test_data[:, :2], test_data[:, 2:])

    train_losses = []
    test_losses = []
    train_losses_cp = []
    test_losses_cp = []

    for d in dlist:
        net_orig = TwoLayerNet(input_dim=2, hidden_dim=d, init_uniform=None, activation=nn.ReLU).to(device)
        net_cp, weights_t = compress_nn(net_orig, dstop=dstop(d), k=k, tol=tol)
        print(f'Compression completed. d={d} -> dstop={dstop(d)}')

        # Train all cases with identical minibatches/order — sequential execution
        train_loss, test_loss = bptrain(net_orig, train_ds, test_ds, epochs=epochs, batch_size=batch_size, seed=seed, algo=algo, lr=lr)
        train_loss_cp, test_loss_cp = bptrain(net_cp, train_ds, test_ds, epochs=epochs, batch_size=batch_size, seed=seed, algo=algo, lr=lr)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_losses_cp.append(train_loss_cp)
        test_losses_cp.append(test_loss_cp)

    epoch_range = np.arange(0, epochs+1, 10)

    os.makedirs('LTH', exist_ok=True)
    filename = f'LTH/{task_name}_{algo_name}_k{k}_noise{train_noise}_bs{batch_size}_lr{lr}'
    # if save_csv:
    #     with open(filename + '.csv', 'w', newline='') as csvfile:
    #         writer = csv.writer(csvfile)
    #         # Write header
    #         writer.writerow([
    #             'epoch',
    #             'train_loss_orig',
    #             'test_loss_orig',
    #             'train_loss_cp',
    #             'test_loss_cp',
    #         ])
    #         # Write data rows
    #         for i in range(len(epoch_range)):
    #             writer.writerow([
    #                 epoch_range[i],
    #                 train_loss_orig[i],
    #                 test_loss_orig[i],
    #                 train_loss_cp[i],
    #                 test_loss_cp[i],
    #                 train_loss_naive[i],
    #                 test_loss_naive[i]
    #             ])
    if save_pdf:
        fig, axs = make_canvas(rows=2, cols=1, axes_width_pt=300)

        for n, d in enumerate(dlist):
            axs[0].plot(epoch_range, test_losses[n], marker=None, ls='-', label=f"d={d}")
            axs[1].plot(epoch_range, test_losses_cp[n], marker=None, ls='-', label=f"d'={dstop(d)}")

        axs[0].legend()
        axs[0].set_ylabel('Test loss - orig')
        axs[0].set_yscale('log')
        axs[1].set_ylabel('Test loss - cp')
        axs[1].set_xlabel('Epoch')
        axs[1].set_yscale('log')

        plt.tight_layout()
        plt.savefig(filename+'.pdf', format='pdf', bbox_inches='tight', pad_inches=0)
        plt.show()
