
import numpy as np
import random
import math
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# Device: prefer Apple Silicon MPS when available
device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))

# ----------------------------
# Utilities
# ----------------------------

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_mod_add_dataset(P: int = 97, train_frac: float = 0.5, seed: int = 0):
    """Build the full modular-addition table and split into train/test without replacement.
    Inputs are two reals in [-1, 1]; target is an integer class 0..P-1.
    """
    xs, ys = np.meshgrid(np.arange(P, dtype=np.int64), np.arange(P, dtype=np.int64), indexing='ij')
    xs = xs.reshape(-1)
    ys = ys.reshape(-1)

    # labels: (x + y) mod P
    labels = (xs + ys) % P

    # encode inputs to [-1, 1]
    X = np.stack([2.0 * (xs / (P - 1.0)) - 1.0,
                  2.0 * (ys / (P - 1.0)) - 1.0], axis=1).astype(np.float32)
    y = labels.astype(np.int64)

    # split
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    n_train = int(train_frac * len(X))
    idx_tr = perm[:n_train]
    idx_te = perm[n_train:]

    Xtr = torch.from_numpy(X[idx_tr])
    ytr = torch.from_numpy(y[idx_tr])
    Xte = torch.from_numpy(X[idx_te])
    yte = torch.from_numpy(y[idx_te])

    return Xtr, ytr, Xte, yte


class TwoLayerNet(nn.Module):
    def __init__(self, width: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, num_classes),  # logits for classes 0..P-1
        )

    def forward(self, x):
        return self.net(x)


def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            total_loss += loss.item() * xb.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += xb.size(0)
    return total_loss / total, correct / total


def train_grokking(
    P: int = 97,
    train_frac: float = 0.5,
    width: int = 256,
    lr: float = 5e-4,
    weight_decay: float = 5e-4,
    steps: int = 30000,
    batch_size: int = 4096,
    eval_every: int = 100,
    seed: int = 0,
    logy: bool = True,
):
    set_seed(seed)

    # Data
    Xtr, ytr, Xte, yte = build_mod_add_dataset(P=P, train_frac=train_frac, seed=seed)
    train_ds = TensorDataset(Xtr, ytr)
    test_ds = TensorDataset(Xte, yte)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=8192, shuffle=False)

    # Model & optimizer
    model = TwoLayerNet(width=width, num_classes=P).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    # Initial evaluation
    tr_loss, tr_acc = evaluate(model, train_loader, loss_fn)
    te_loss, te_acc = evaluate(model, test_loader, loss_fn)

    steps_axis = [0]
    train_losses = [tr_loss]
    test_losses = [te_loss]
    train_accs = [tr_acc]
    test_accs = [te_acc]

    # Training loop
    it = 0
    while it < steps:
        for xb, yb in train_loader:
            it += 1
            model.train()
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

            if it % eval_every == 0 or it == steps:
                tr_loss, tr_acc = evaluate(model, train_loader, loss_fn)
                te_loss, te_acc = evaluate(model, test_loader, loss_fn)
                steps_axis.append(it)
                train_losses.append(tr_loss)
                test_losses.append(te_loss)
                train_accs.append(tr_acc)
                test_accs.append(te_acc)
                print(f"step {it:6d}/{steps}: train loss {tr_loss:.4f} acc {tr_acc*100:.1f}% | test loss {te_loss:.4f} acc {te_acc*100:.1f}%")

            if it >= steps:
                break

    # Plot losses (log-scale is helpful to see the sharp drop)
    plt.figure(figsize=(7, 5))
    plt.plot(steps_axis, train_losses, label='Train loss')
    plt.plot(steps_axis, test_losses, label='Test loss')
    if logy:
        plt.yscale('log')
    plt.xlabel('Steps')
    plt.ylabel('Cross-Entropy Loss')
    plt.title(f'Grokking on modular addition: P={P}, train_frac={train_frac}, width={width}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"grokking_P{P}_frac{int(train_frac*100)}_w{width}.pdf", bbox_inches='tight', pad_inches=0)
    plt.show()

    return model, (steps_axis, train_losses, test_losses, train_accs, test_accs)


if __name__ == '__main__':
    # Defaults chosen to reliably show grokking on a laptop CPU/GPU/MPS. If you don't see
    # a clear generalization jump, increase `steps` to 50_000 or set `train_frac` in [0.4, 0.6].
    train_grokking(
        P=97,
        train_frac=0.5,      # try 0.4–0.6 for sharpest grokking
        width=256,           # 128–512 work; 256 is a common sweet spot
        lr=1e-4,
        weight_decay=1e-3,   # key to induce the memorization→rule transition
        steps=30_000,        # raise to 50_000 if needed
        batch_size=2048,
        eval_every=100,
        seed=0,
        logy=True,
    )
