import numpy as np
import os
import random
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
import torch.nn as nn
from torch.utils.data import DataLoader

from common import TwoLayerNet, WeightedTwoLayerNet

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# thread tuning (good default on Apple Silicon)
p = (os.cpu_count() or 8) // 2 or 1
print(f"p = {p}")
os.environ.setdefault("OMP_NUM_THREADS", str(p))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(p))
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(p))
os.environ.setdefault("NUMEXPR_MAX_THREADS", str(p))
torch.set_num_threads(max(1, p))
torch.set_num_interop_threads(2)  # usually enough

# PyTorch on Apple/MPS (if you use it)
if torch.backends.mps.is_available():
    torch.set_float32_matmul_precision("high")  # PyTorch ≥2.0



def fix_random_seed(seed=0):
    # Set a fixed random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)

    # Enforce deterministic algorithms where available
    torch.use_deterministic_algorithms(True)

    # Backend-specific determinism settings
    # CUDA (if available): disable benchmark and ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # MPS (for macOS Metal): disable benchmark and enforce determinism
    torch.backends.mps.deterministic = True
    torch.backends.mps.benchmark = False


def compute_loss(net, loader, loss_fn):
    net.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)

def weighted_loader(dataset, weight=None, batch_size=64):
    if weight is None:
        sampler = torch.utils.data.RandomSampler(
            data_source=dataset,
            replacement=True,
            num_samples=len(dataset)
        )
    else:
        # Keep weights on CPU to avoid MPS float64 conversion issues
        weights_tensor = torch.tensor(weight, dtype=torch.float, device='cpu')
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights_tensor,
            num_samples=len(dataset),
            replacement=True
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler
    )

def bptrain(loader, epochs, lr):
    net = CNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    train_losses = []
    test_losses = []

    # Training loop
    for epoch in range(1, epochs+1):
        net.train()
        total_loss = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
        train_loss = total_loss / len(train_ds)
        train_losses.append(train_loss)
        test_loss = compute_loss(net, test_loader, loss_fn)
        test_losses.append(test_loss)
        print(f"Epoch {epoch}/{epochs}. Train loss: {train_loss:.3f}, test loss: {test_loss:.3f}")

    return train_losses, test_losses



if __name__ == '__main__':
    # hyperparams
    epochs = 20
    batch_size = 64
    lr = 1e-4
    seed = 0

    # data loaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_ds = datasets.MNIST(root='data', train=True, download=False, transform=transform)
    test_ds  = datasets.MNIST(root='data', train=False, download=False, transform=transform)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    archive = np.load("weights_k1.npz")
    weight_dict = {int(name.split("_",1)[1]): archive[name] for name in archive}


    # plots
    epochs_range = list(range(1, epochs+1))
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    loader = weighted_loader(train_ds, batch_size=batch_size)
    fix_random_seed(seed)
    train_losses, test_losses = bptrain(loader, epochs, lr)
    axs[0].plot(epochs_range, train_losses, marker='o', label=f"d={len(train_ds)}")
    axs[1].plot(epochs_range, test_losses, marker='o', label=f"d={len(train_ds)}")

    # for d in [200, 2000, 20000]:
    #     weight = weight_dict[d]
    #     loader = weighted_loader(train_ds, weight, batch_size=batch_size)
    #     fix_random_seed(seed)
    #     train_losses, test_losses = bptrain(loader, epochs, lr)
    #     axs[0].plot(epochs_range, train_losses, marker='o', label=f"d={d}")
    #     axs[1].plot(epochs_range, test_losses, marker='o', label=f"d={d}")

    d = 20000

    weight = weight_dict[d]
    loader = weighted_loader(train_ds, weight, batch_size=batch_size)
    fix_random_seed(seed)
    train_losses, test_losses = bptrain(loader, epochs, lr)
    axs[0].plot(epochs_range, train_losses, marker='o', label=f"k=1 compress d={d}")
    axs[1].plot(epochs_range, test_losses, marker='o', label=f"k=1 compress d={d}")

    # the naive pruning
    weight = np.zeros(60000, dtype=int)
    weight[:d] = 1
    np.random.shuffle(weight)
    loader = weighted_loader(train_ds, weight, batch_size=batch_size)
    fix_random_seed(seed)
    train_losses, test_losses = bptrain(loader, epochs, lr)
    axs[0].plot(epochs_range, train_losses, marker='o', label=f"Naive prune d={d}")
    axs[1].plot(epochs_range, test_losses, marker='o', label=f"Naive prune d={d}")

    
    # final adjustments to the plot
    axs[0].set_ylabel('Train CE')
    axs[0].grid(True)
    axs[0].legend()
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Test CE')
    axs[1].grid(True)
    axs[1].legend()
    plt.tight_layout()
    plt.savefig("train_on_20000_1.pdf", format='pdf', bbox_inches='tight', pad_inches=0)
    plt.show()
