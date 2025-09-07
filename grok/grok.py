import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from compressor import Compressor

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


def make_dataset(pairs, P: int, device: torch.device):
    """Create TensorDataset where inputs=(a,b) and labels=(a+b)%P on given device."""
    ab = torch.tensor(pairs, dtype=torch.long).to(device)
    y = (ab[:, 0] + ab[:, 1]) % int(P)
    return TensorDataset(ab, y)

# ----- Model -----
class TinyAddTransformer(nn.Module):
    def __init__(self, P, d_model=128, n_heads=4, n_layers=1):
        super().__init__()
        self.emb = nn.Embedding(P, d_model)
        self.pos = nn.Parameter(torch.zeros(2, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4*d_model,
            batch_first=True, activation="gelu"
        )
        self.tr = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, P)

    def forward(self, x2):
        h = self.emb(x2) + self.pos.unsqueeze(0)
        h = self.tr(h)
        h = self.norm(h.mean(dim=1))
        return self.head(h)

def accuracy(logits, y):
    return (logits.argmax(dim=-1) == y).float().mean().item()

@torch.no_grad()
def eval_model(model, loader, loss_fn=None):
    """
    Evaluate accuracy (and optional loss) over a single DataLoader.

    - loss_fn: callable(logits, y) or None. If provided, computes average loss.
    Returns: (accuracy, avg_loss or None)
    """
    model.eval()
    device = next(model.parameters()).device

    total_samples = 0
    total_correct = 0
    total_loss = 0.0
    use_loss = loss_fn is not None

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits = model(xb)                # single forward
        preds = logits.argmax(dim=-1)
        batch_size = yb.size(0)

        total_correct += int((preds == yb).sum().item())
        total_samples += batch_size

        if use_loss:
            loss = loss_fn(logits, yb)
            # If loss_fn returns a scalar mean per-batch, multiply by batch size.
            # If it already returns a sum (reduction='sum'), this is correct too.
            total_loss += float(loss) * batch_size if loss.dim() == 0 else float(loss.sum())

    acc = total_correct / total_samples
    avg_loss = (total_loss / total_samples) if use_loss else None
    return acc, avg_loss


# ----- Training & Plotting -----
def bptrain(
    train_loader,
    test_loader,
    *,
    P: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    lr: float,
    wd: float,
    max_epochs: int,
    device: torch.device,
):
    model = TinyAddTransformer(P, d_model, n_heads, n_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    # sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS, eta_min=LR*0.1)

    # Initial metrics at epoch 0
    tr_acc, tr_loss = eval_model(model, train_loader, loss_fn=cross_entropy)
    te_acc, te_loss = eval_model(model, test_loader, loss_fn=cross_entropy)
    train_accs, test_accs = [tr_acc], [te_acc]
    train_losses, test_losses = [tr_loss], [te_loss]
    epochs = list(range(1, max_epochs + 1))

    for epoch in epochs:
        model.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            loss = cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        # sched.step()

        if epoch % 10 == 0:
            tr_acc, tr_loss = eval_model(model, train_loader, loss_fn=cross_entropy)
            te_acc, te_loss = eval_model(model, test_loader, loss_fn=cross_entropy)
            train_accs.append(tr_acc)
            test_accs.append(te_acc)
            train_losses.append(tr_loss)
            test_losses.append(te_loss)
            print(f"Epoch {epoch:5d} — Train: {tr_acc*100:5.2f}%, Test: {te_acc*100:5.2f}% | Loss tr/te: {tr_loss:.4f}/{te_loss:.4f}")

    return train_accs, test_accs, train_losses, test_losses



def compress_loader(ds: TensorDataset, dstop=2000, k=2, batch_size=512, seed=0):
    """Compress the given dataset using the Compressor and return a DataLoader."""
    d = len(ds)
    ab, y = ds.tensors  # unpack tensors
    device = ab.device
    # Move to CPU and numpy
    ab_np = ab.cpu().numpy()
    y_np  = y.cpu().numpy().reshape(-1, 1)
    w = np.hstack([ab_np, y_np])
    assert w.shape == (d, 3), f"Unexpected shape: {w.shape}"

    cp = Compressor(w, random_state=seed)
    c_, w_cp = cp.compress(k, dstop=dstop, print_progress=True)
    print(f"Compression completed. d={d} -> d'={dstop}")

    # form a new TensorDataset from w_cp
    ds_cp = TensorDataset(torch.tensor(w_cp[:, :2], dtype=torch.long, device=device), torch.tensor(w_cp[:, 2], dtype=torch.long, device=device))
    c_t = torch.tensor(c_, dtype=torch.float)
    sampler = torch.utils.data.WeightedRandomSampler(weights=c_t, num_samples=d, replacement=True)
    return DataLoader(ds_cp, batch_size=batch_size, sampler=sampler)


if __name__ == "__main__":
    # ----- Config -----
    P = 97
    N_TOTAL = P * P
    N_BIG = 4000
    N_SMALL = 2000
    D_MODEL = 128
    N_HEADS = 4
    N_LAYERS = 1
    LR = 1e-5
    WD = 1.0
    BATCH_SIZE = min(512, N_SMALL // 2)
    # BATCH_SIZE = 512
    MAX_EPOCHS = 1000
    # DEVICE = torch.device(
    #     'cuda' if torch.cuda.is_available() else (
    #         'mps' if torch.backends.mps.is_available() else 'cpu'
    #     )
    # )
    DEVICE = torch.device('cpu')
    SEED = 123

    fix_random_seed(SEED)

    # ----- Dataset -----
    pairs = [(a, b) for a in range(P) for b in range(P)]
    test_ds = make_dataset(pairs, P=P, device=DEVICE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # various train loaders
    random.shuffle(pairs)
    train_ds_big = make_dataset(pairs[:N_BIG], P=P, device=DEVICE)
    train_loader_big = DataLoader(
        train_ds_big,
        batch_size=BATCH_SIZE,
        sampler=torch.utils.data.RandomSampler(train_ds_big, replacement=True, num_samples=N_BIG),
    )

    train_ds_small = make_dataset(pairs[:N_SMALL], P=P, device=DEVICE)
    train_loader_small = DataLoader(
        train_ds_small,
        batch_size=BATCH_SIZE,
        sampler=torch.utils.data.RandomSampler(train_ds_small, replacement=True, num_samples=N_BIG),
    )

    train_loader_cp = compress_loader(train_ds_big, dstop=N_SMALL, k=4, batch_size=BATCH_SIZE, seed=SEED)


    fix_random_seed(SEED)
    train_accs_big, test_accs_big, train_losses_big, test_losses_big = bptrain(train_loader_big, test_loader, P=P, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, lr=LR, wd=WD, max_epochs=MAX_EPOCHS, device=DEVICE)
    fix_random_seed(SEED)
    train_accs_small, test_accs_small, train_losses_small, test_losses_small = bptrain(train_loader_small, test_loader, P=P, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, lr=LR, wd=WD, max_epochs=MAX_EPOCHS, device=DEVICE)
    fix_random_seed(SEED)
    train_accs_cp, test_accs_cp, train_losses_cp, test_losses_cp = bptrain(train_loader_cp, test_loader, P=P, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, lr=LR, wd=WD, max_epochs=MAX_EPOCHS, device=DEVICE)


    epoch_range = np.arange(0, MAX_EPOCHS+1, 10)
    color_big = 'tab:green'
    color_small = 'tab:orange'
    color_cp = 'tab:blue'

    fig, axs = plt.subplots(4, 1, figsize=(6, 6), sharex=True)
    # axs[0]: train accuracy vs epoch_range
    axs[0].plot(epoch_range, train_accs_big, label='Big', color=color_big, marker=None)
    axs[0].plot(epoch_range, train_accs_small, label='Small', color=color_small, marker=None)
    axs[0].plot(epoch_range, train_accs_cp, label='Compressed', color=color_cp, marker=None)
    axs[0].legend()
    axs[0].set_ylabel('Train Acc')
    axs[0].grid(True)

    # axs[1]: test accuracy
    axs[1].plot(epoch_range, test_accs_big, label='Big', color=color_big, marker=None)
    axs[1].plot(epoch_range, test_accs_small, label='Small', color=color_small, marker=None)
    axs[1].plot(epoch_range, test_accs_cp, label='Compressed', color=color_cp, marker=None)
    axs[1].set_ylabel('Test Acc')
    axs[1].grid(True)

    # axs[2]: train loss
    axs[2].plot(epoch_range, train_losses_big, label='Big', color=color_big, marker=None)
    axs[2].plot(epoch_range, train_losses_small, label='Small', color=color_small, marker=None)
    axs[2].plot(epoch_range, train_losses_cp, label='Compressed', color=color_cp, marker=None)
    axs[2].set_ylabel('Train Loss')
    axs[2].grid(True)

    # axs[3]: test loss
    axs[3].plot(epoch_range, test_losses_big, label='Big', color=color_big, marker=None)
    axs[3].plot(epoch_range, test_losses_small, label='Small', color=color_small, marker=None)
    axs[3].plot(epoch_range, test_losses_cp, label='Compressed', color=color_cp, marker=None)
    axs[3].set_ylabel('Test Loss')
    axs[3].set_xlabel('Epoch')
    axs[3].grid(True)

    plt.tight_layout()
    plt.show()
