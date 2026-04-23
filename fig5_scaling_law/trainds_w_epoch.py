import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
plt.rc('font', family='Helvetica', size=8)
import csv

from common import TwoLayerNet, fix_random_seed, make_canvas, cyl_harmonic
from data_gen import generate_data

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from compressor import Compressor

# Device configuration
# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
device = torch.device("cpu")


def make_loader(data, num_samples=None, batch_size=None, weights=None):
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    X = data[:, :2].float().to(device)
    y = data[:, [2]].float().to(device)

    ds = TensorDataset(X, y)
    if batch_size is None:
        batch_size = len(ds)  # full-batch
    if num_samples is None:
        num_samples = batch_size

    if weights is None:
        sampler = torch.utils.data.RandomSampler(ds, replacement=True, num_samples=num_samples)
    else:
        weights = torch.as_tensor(weights)
        sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)
    return DataLoader(ds, batch_size=batch_size, sampler=sampler)

def make_test_loader(data, batch_size=4096):
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    X = data[:, :2].float().to(device)
    y = data[:, [2]].float().to(device)

    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=batch_size)

def compute_loss(net, loader):
    loss_fn = nn.MSELoss(reduction="sum")
    total_loss = 0.
    total_samples = 0

    net.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = net(inputs)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            total_samples += labels.numel()            
    return total_loss/total_samples

def bptrain(train_loader, test_loader, hidden_dim:int, epochs=5, eval_every=16, seed=0, algo=torch.optim.SGD, **opt_params):
    fix_random_seed(seed)
    net = TwoLayerNet(2, hidden_dim).to(device)
    opt = algo(net.parameters(), **opt_params)
    loss_fn = nn.MSELoss()
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=0.)

    # initial losses
    test_losses = [compute_loss(net, test_loader)]

    print_fraction = 0.

    for epoch in range(1, epochs + 1):
        net.train()
        for inputs, labels in train_loader:
            opt.zero_grad()
            outputs = net(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            opt.step()
        sched.step()

        if (epoch % eval_every == 0) or (epoch==epochs):
            test_loss = compute_loss(net, test_loader)
            test_losses.append(test_loss)

        if epoch / epochs >= print_fraction:
            print(f"Epoch {epoch}/{epochs}")
            print_fraction += 1/10

    return test_losses

if __name__ == "__main__":
    seed = 42
    # num_seeds = 10

    dlist = [2**n for n in range(8, 20)]
    dstop = lambda d: int(16*np.sqrt(d))
    k = 6

    train_noise = 3.0
    test_size = 100_000
    hidden_dim = 50
    compute_budget = 2**20
    batch_size = 512
    epochs = compute_budget // batch_size
    eval_every = 16

    algo_name = 'AdamW'
    lr = 1e-3
    algo = torch.optim.AdamW

    task_name = 'teacher'

    for d in dlist:
        test_losses = []
        test_losses_cp = []
        
        fix_random_seed(seed*10)
        # f = lambda x, y: cyl_harmonic(x, y, n=6, k=20)
        truth_net = TwoLayerNet(2, hidden_dim, init_uniform=1.).to(device)
        test_data = generate_data(test_size, net=truth_net, noise=0, seed=seed**3, return_tensor=True, device=device)
        test_loader = make_test_loader(test_data)

        # Generate training data with per-run seed
        train_data = generate_data(
            d,
            net=truth_net,
            noise=train_noise,
            seed=seed**2 + d,
            return_tensor=True,
            device=device,
        )

        # Compress with per-run random state
        cp = Compressor(train_data.to("cpu").numpy(), random_state=seed)
        c_, train_cp = cp.compress(k, dstop=dstop(d), print_progress=False)

        train_loader = make_loader(train_data, num_samples=d, batch_size=min(batch_size, d//2))
        train_loader_cp = make_loader(train_cp, num_samples=d, batch_size=min(batch_size, d//2), weights=c_)

        test_loss = bptrain(
            train_loader, test_loader, hidden_dim, epochs=epochs, seed=seed, eval_every=16, algo=algo, lr=lr
        )
        test_loss_cp = bptrain(
            train_loader_cp, test_loader, hidden_dim, epochs=epochs, seed=seed, eval_every=16, algo=algo, lr=lr
        )

        test_losses.append(test_loss)
        test_losses_cp.append(test_loss_cp)


    print("original test losses:\n")
    for loss in test_losses:
        print(f"{loss[-1]:.4f}", end=", ")
    print("\ncompressed test losses:\n")
    for loss in test_losses_cp:
        print(f"{loss[-1]:.4f}", end=", ")

    # plots
    epoch_range = list(range(0, epochs+1, eval_every))
    if epoch_range[-1] != epochs:
        epoch_range.append(epochs)
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

    # fig, ax = make_canvas(axes_width_pt=300)
    # ax.plot(dlist, test_losses, marker='o', ls='-', label=f"Original")
    # ax.plot([dstop(d) for d in dlist], test_losses_cp, marker=None, ls='^', label=f"Compressed")

    # ax.legend()
    # ax.set_xscale("log")
    # ax.set_yscale("log")
    # ax.set_ylabel("Test loss")
    # ax.set_xlabel("dataset size")
    # ax.set_title(r"$d\to 8\sqrt{d}$")
    # plt.tight_layout()

    # os.makedirs('CPTDS', exist_ok=True)
    # filename = f'CPTDS/{task_name}_{algo_name}_k{k}_noise{train_noise}_hidden{hidden_dim}_bs{batch_size}_lr{lr}'
    # with open(filename + '.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     # Write header
    #     writer.writerow([
    #         'd',
    #         'dstop',
    #         'train_loss_orig',
    #         'test_loss_orig',
    #         'train_loss_cp',
    #         'test_loss_cp',
    #     ])
    #     # Write data rows
    #     for n in range(len(dlist)):
    #         writer.writerow([
    #             dlist[n],
    #             dstop(dlist[n]),
    #             train_losses[n],
    #             test_losses[n],
    #             train_losses_cp[n],
    #             test_losses_cp[n]
    #         ])
    # plt.savefig(filename+'.pdf', format='pdf', bbox_inches='tight', pad_inches=0)
    plt.show()
