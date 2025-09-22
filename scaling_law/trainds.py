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

def make_test_loader(data, batch_size=2048):
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

def bptrain(train_loader, test_loader, hidden_dim:int, eval_every=16, epochs=5, seed=0, algo=torch.optim.SGD, **opt_params):
    fix_random_seed(seed)
    net = TwoLayerNet(2, hidden_dim).to(device)
    opt = algo(net.parameters(), **opt_params)
    loss_fn = nn.MSELoss()
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=0.)

    # # initial losses
    # train_losses = [compute_loss(net, train_loader, weights=None)]
    # test_losses = [compute_loss(net, test_loader, weights=None)]

    # if train_weights is not None:
    #     w = torch.as_tensor(train_weights, dtype=torch.float, device=device).view(-1)

    # print_fraction = 0

    for epoch in range(1, epochs + 1):
        net.train()
        for inputs, labels in train_loader:
            opt.zero_grad()
            outputs = net(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            opt.step()
        sched.step()

        # # record losses after the update
        # train_loss = compute_loss(net, train_loader, weights=None)
        # test_loss = compute_loss(net, test_loader, weights=None)
        # train_losses.append(train_loss)
        # test_losses.append(test_loss)

        # if epoch / epochs >= print_fraction:
        #     # print(f"Epoch {epoch}/{epochs}. Train loss: {train_loss:.3e}, test loss: {test_loss:.3e}")
        #     print(f"Epoch {epoch}/{epochs}")
        #     print_fraction += 1/10

    return compute_loss(net, test_loader)

def main(d, seed):
    dstop = lambda d: int(16*np.sqrt(d))
    k = 6

    train_noise = 3.0
    test_size = 100_000
    hidden_dim = 50
    compute_budget = 2**20
    batch_size = 512
    epochs = compute_budget // batch_size

    algo_name = 'AdamW'
    lr = 1e-3
    algo = torch.optim.AdamW
    task_name = 'teacher'

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
        train_loader, test_loader, hidden_dim, epochs=epochs, seed=seed, algo=algo, lr=lr
    )
    test_loss_cp = bptrain(
        train_loader_cp, test_loader, hidden_dim, epochs=epochs, seed=seed, algo=algo, lr=lr
    )

    return test_loss, test_loss_cp


if __name__ == "__main__":
    # seedlist = list(range(20))
    # dlist = [2**n for n in range(17, 20)]
    sid = int(os.environ["SLURM_ARRAY_TASK_ID"])
    seed = sid % 20
    log2d = sid // 20
    d = 2**log2d
    print(f'SLURM_ARRAY_TASK_ID={sid} -> seed={seed}, d={d}')

    csv_path = 'trainds_scaling_temp.csv'
    file_exists = os.path.exists(csv_path)
    write_header = (not file_exists) or os.path.getsize(csv_path) == 0

    mode = 'a' if file_exists else 'w'
    with open(csv_path, mode, newline='') as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow([
                'seed',
                'd',
                'dstop',
                'test_loss_orig',
                'test_loss_cp',
            ])
        
        # evaluate and write data rows
        # for d in dlist:
        #     for seed in seedlist:
        test_loss, test_loss_cp = main(d, seed)
        writer.writerow([
            seed,
            d,
            int(16*np.sqrt(d)),
            test_loss,
            test_loss_cp
        ])
        print(f"seed={seed}, d={d} done.")
