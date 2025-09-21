import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
# from torch.amp import autocast, GradScaler
from matplotlib import pyplot as plt
plt.rc('font', family='Helvetica', size=8)
import csv

from common import TwoLayerNet, WeightedTwoLayerNet, compress_nn, make_canvas, cyl_harmonic
from data_gen import generate_data

import sys
import os

# Device configuration
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device('cuda')
else:
    device = torch.device("cpu")


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

def train_pair(net_orig: nn.Module,
               net_cp: nn.Module,
               train_ds: TensorDataset,
               test_ds: TensorDataset,
               epochs: int = 5,
               batch_size: int = 64,
               seed: int = 0,
               algo=torch.optim.SGD,
               **opt_params):
    """Train original and compressed networks side-by-side using shared minibatches."""
    # set_training_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=len(test_ds))
    loss_fn = nn.MSELoss()

    opt_orig = algo(net_orig.parameters(), **opt_params)
    opt_cp = algo(net_cp.parameters(), **opt_params)
    sched_orig = torch.optim.lr_scheduler.CosineAnnealingLR(opt_orig, T_max=epochs, eta_min=0.)
    sched_cp = torch.optim.lr_scheduler.CosineAnnealingLR(opt_cp, T_max=epochs, eta_min=0.)

    use_amp = device.type == 'cuda'
    # scaler = GradScaler(device.type, enabled=use_amp)

    if isinstance(net_cp, WeightedTwoLayerNet):
        weights_t = net_cp.weights
        inv_c = torch.where(weights_t > tol,
                            1.0 / weights_t,
                            torch.zeros_like(weights_t))

        # Scale per-element gradients via hooks
        net_cp.fc1.weight.register_hook(lambda grad: grad * inv_c.view(-1, 1))
        net_cp.fc1.bias.register_hook(lambda grad: grad * inv_c)
        net_cp.fc2.weight.register_hook(lambda grad: grad * inv_c.view(1, -1))

    test_losses_orig = [compute_loss(net_orig, test_loader, loss_fn)]
    test_losses_cp = [compute_loss(net_cp, test_loader, loss_fn)]

    for epoch in range(1, epochs + 1):
        net_orig.train()
        net_cp.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            opt_orig.zero_grad(set_to_none=True)
            opt_cp.zero_grad(set_to_none=True)

            # with autocast(device.type, enabled=use_amp):
            outputs_orig = net_orig(inputs)
            outputs_cp = net_cp(inputs)
            loss_orig = loss_fn(outputs_orig, labels)
            loss_cp = loss_fn(outputs_cp, labels)
            loss_total = loss_orig + loss_cp

            if use_amp:
                scaler.scale(loss_total).backward()
                scaler.step(opt_orig)
                scaler.step(opt_cp)
                scaler.update()
            else:
                loss_total.backward()
                opt_orig.step()
                opt_cp.step()

        sched_orig.step()
        sched_cp.step()

        tel_orig = compute_loss(net_orig, test_loader, loss_fn)
        tel_cp = compute_loss(net_cp, test_loader, loss_fn)
        test_losses_orig.append(tel_orig)
        test_losses_cp.append(tel_cp)
        print(f"Epoch {epoch}/{epochs}. Test loss: orig={tel_orig:.3e}, cp={tel_cp:.3e}")

    return test_losses_orig, test_losses_cp
    


if __name__ == '__main__':
    save_csv = True
    save_pdf = True

    # seed = 42
    # dlist = [2**n for n in range(8,10)]
    sid = int(os.environ["SLURM_ARRAY_TASK_ID"])
    seed = sid % 20
    log2d = sid // 20
    d = 2**log2d
    print(f'SLURM_ARRAY_TASK_ID={sid} -> seed={seed}, d={d}')

    # Hyperparameters
    dstop = lambda d: int(16*np.sqrt(d))
    k = 6
    train_size = 10**7
    test_size = 10**5
    train_noise = 0.2
    tol = 1e-12
    epochs = 200
    batch_size = 512

    # AdamW
    algo_name = 'AdamW'
    lr = 1e-3
    algo = torch.optim.AdamW

    task_name = "harm"
    # net_truth = TwoLayerNet(input_dim=2, hidden_dim=1000, init_uniform=None, activation=nn.ReLU).to(device)
    f = lambda x, y: cyl_harmonic(x, y, n=6, k=20)

    train_data = generate_data(train_size, f=f, noise=train_noise, seed=seed**2, return_tensor=True, device=device)
    train_ds = TensorDataset(train_data[:, :2], train_data[:, 2:])

    test_data = generate_data(test_size, f=f, noise=0., seed=seed**3, return_tensor=True, device=device)
    test_ds = TensorDataset(test_data[:, :2], test_data[:, 2:])

    net_orig = TwoLayerNet(input_dim=2, hidden_dim=d, init_uniform=None, activation=nn.ReLU).to(device)
    net_cp, _ = compress_nn(net_orig, dstop=dstop(d), k=k, tol=tol)
    print(f'Compression completed. d={d} -> dstop={dstop(d)}')

    # Train all cases with identical minibatches/order — sequential execution
    test_loss, test_loss_cp = train_pair(
        net_orig,
        net_cp,
        train_ds,
        test_ds,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        algo=algo,
        lr=lr,
    )

    epoch_range = list(range(0, epochs+1))

    os.makedirs(f'LTH_{task_name}_{algo_name}_k{k}_noise{train_noise}_bs{batch_size}_lr{lr}', exist_ok=True)
    filename = f'LTH_{task_name}_{algo_name}_k{k}_noise{train_noise}_bs{batch_size}_lr{lr}/d{d}_dstop{dstop(d)}_seed{seed}'
    if save_csv:
        with open(filename + '.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(['d', 'test_loss_orig', 'test_loss_cp'])
            # Write one row per width with final-epoch test losses
            for n in epoch_range:
                writer.writerow([n, test_loss[n], test_loss_cp[n]])
    if save_pdf and seed == 0:
        fig, axs = make_canvas(rows=1, cols=1, axes_width_pt=200)

        axs.plot(epoch_range, test_loss, marker=None, ls='-', label=f"d={d}")
        axs.plot(epoch_range, test_loss_cp, marker=None, ls='--', label=f"d'={dstop(d)}")

        axs.legend()
        axs.set_ylabel('Test loss')
        axs.set_yscale('log')
        axs.set_xlabel('Epoch')

        plt.tight_layout()
        plt.savefig(filename+'.pdf', format='pdf', bbox_inches='tight', pad_inches=0)
        # plt.show()
