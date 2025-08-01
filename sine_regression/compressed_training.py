import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from matplotlib import pyplot as plt
from copy import deepcopy


# from train import TwoLayerNet
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from compressor import Compressor

# Device configuration
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# model: input → hidden Tanh → output
class TwoLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        return self.fc2(self.tanh(self.fc1(x)))
    
class WeightedTwoLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, weights=None):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim, 1)
        # weights is a non-trainable multiplicative mask/scaling per hidden unit
        if weights is None:
            w = torch.ones(hidden_dim, dtype=torch.float32)
        else:
            w = torch.as_tensor(weights, dtype=torch.float32)
            if w.numel() != hidden_dim:
                raise ValueError(f"weights length {w.numel()} != hidden_dim {hidden_dim}")
        self.register_buffer('weights', w.view(-1))

    def forward(self, x):
        h = self.tanh(self.fc1(x))                 # (batch, d)
        h = h * self.weights.view(1, -1)           # elementwise scale by c_j
        return self.fc2(h)


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


# def load_model(model_path='sine_model.pth', hidden_dim=1000):
#     """
#     Loads the trained TwoLayerNet for sin(2πx) regression.
    
#     Args:
#         model_path (str): Path to the .pth file.
#         hidden_dim (int): Width of the hidden layer (should match training).
#     Returns:
#         model (nn.Module): Loaded model in eval mode.
#     """
#     model = TwoLayerNet(input_dim=1, hidden_dim=hidden_dim).to(device)
#     state = torch.load(model_path, map_location=device)
#     model.load_state_dict(state)
#     model.eval()
#     return model


def loaddata():
    # data
    data = np.loadtxt('noisy_sin_10000.csv', delimiter=',')
    x = torch.from_numpy(data[:, [0]]).float().to(device)
    y = torch.from_numpy(data[:, [1]]).float().to(device)
    return TensorDataset(x, y)


# --- Extraction and compression utilities ---
def extract(net: TwoLayerNet):
    """
    Extracts parameters from a trained TwoLayerNet as arrays:
    w_ with shape (d, 3) where d is hidden_dim, and scalar bias c.
    Columns of w_: [a_i, w_i, b_i].
    """
    # fc1: weight shape (d,1), bias shape (d,)
    # fc2: weight shape (1,d), bias scalar
    with torch.no_grad():
        W2 = net.fc2.weight.data.cpu().numpy().reshape(-1)  # shape (d,)
        W1 = net.fc1.weight.data.cpu().numpy().reshape(-1)  # shape (d,)
        b1 = net.fc1.bias.data.cpu().numpy()               # shape (d,)
        b2 = net.fc2.bias.data.item()
    # Stack into (d, 3)
    w_ = np.stack([W2, W1, b1], axis=1)
    return w_, b2




def compress_nn(net, k=1, dstop=100, tol=1e-12):
    """
    Use the Compressor to compute per-unit weights c_j, then build a
    WeightedTwoLayerNet that keeps the original width but applies the
    multiplicative weights. Also return the alive indices (c_j > tol).
    """
    # Extract [a_j, w_j, b_j] from net (shape (d,3))
    w_orig, b2 = extract(net)
    cp = Compressor(w_orig, tol=tol)
    weights = cp.compress_weights(k, dstop=dstop, return_at=[dstop])[dstop]

    d = w_orig.shape[0]
    net_cp = WeightedTwoLayerNet(input_dim=1, hidden_dim=d, weights=weights).to(device)

    # Copy parameters from the original net so both start identically
    with torch.no_grad():
        net_cp.fc1.weight.copy_(net.fc1.weight)
        net_cp.fc1.bias.copy_(net.fc1.bias)
        net_cp.fc2.weight.copy_(net.fc2.weight)
        net_cp.fc2.bias.copy_(net.fc2.bias)

    weights_t = torch.as_tensor(weights, dtype=torch.float32, device=device)
    return net_cp, weights_t


def make_loader(dataset, batch_size=64, seed=0):
    """Deterministic DataLoader using a fixed shuffled index order."""
    n = len(dataset)
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    sampler = torch.utils.data.SubsetRandomSampler(indices.tolist())
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=False)
    return loader


def clone_net(net):
    """Return a parameter-identical clone of `net` on `device`.
    Supports TwoLayerNet and WeightedTwoLayerNet.
    """
    if isinstance(net, TwoLayerNet):
        c = TwoLayerNet(input_dim=net.fc1.weight.shape[1],
                        hidden_dim=net.fc1.weight.shape[0]).to(device)
        with torch.no_grad():
            c.fc1.weight.copy_(net.fc1.weight)
            c.fc1.bias.copy_(net.fc1.bias)
            c.fc2.weight.copy_(net.fc2.weight)
            c.fc2.bias.copy_(net.fc2.bias)
        return c

    if isinstance(net, WeightedTwoLayerNet):
        input_dim = net.fc1.weight.shape[1]
        hidden_dim = net.fc1.weight.shape[0]
        # Preserve the exact per-unit weights (buffer)
        weights_buf = net.weights.detach().clone()
        c = WeightedTwoLayerNet(input_dim=input_dim,
                                hidden_dim=hidden_dim,
                                weights=weights_buf).to(device)
        with torch.no_grad():
            c.fc1.weight.copy_(net.fc1.weight)
            c.fc1.bias.copy_(net.fc1.bias)
            c.fc2.weight.copy_(net.fc2.weight)
            c.fc2.bias.copy_(net.fc2.bias)
            # Ensure buffer matches exactly
            c.weights.copy_(net.weights)
        return c

    raise TypeError(f"clone_net: unsupported network type {type(net)}")


def train_orig(net, dataset, epochs=5, batch_size=64, lr=0.01, seed=0):
    """SGD training of the baseline network. Returns snapshots per epoch (including epoch 0)."""
    fix_random_seed(seed)
    loader = make_loader(dataset, batch_size=batch_size, seed=seed)
    criterion = nn.MSELoss()
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    snapshots = [clone_net(net)]  # epoch 0 (untrained)
    for epoch in range(epochs):
        for x, y in loader:
            opt.zero_grad()
            loss = criterion(net(x), y)
            loss.backward()
            opt.step()
        snapshots.append(clone_net(net))
        if (epoch+1)%100 == 0:
            print(f"Epoch {epoch+1}/{epochs} completed")
    return snapshots


def train_weighted(net_w: WeightedTwoLayerNet, weights_t: torch.Tensor,
                   dataset, epochs=5, batch_size=64, lr=0.01, seed=0):
    """
    Train with the custom rule:
      - b2 uses standard SGD step
      - For j in alive: W1_j, W2_j, b1_j use w <- w - (lr / c_j) * grad
      - For j not in alive: no update
    Returns snapshots per epoch (including epoch 0).
    """
    fix_random_seed(seed)
    loader = make_loader(dataset, batch_size=batch_size, seed=seed)
    criterion = nn.MSELoss()

    # lr_scale = 1/c_j for alive, 0 for dead
    inv_c = torch.where(weights_t > tol,
                        1.0 / weights_t,
                        torch.zeros_like(weights_t))

    # Scale per-element gradients via hooks; keeps using stock SGD
    net_w.fc1.weight.register_hook(lambda grad: grad * inv_c.view(-1, 1))
    net_w.fc1.bias.register_hook(lambda grad: grad * inv_c)
    net_w.fc2.weight.register_hook(lambda grad: grad * inv_c.view(1, -1))

    opt = torch.optim.Adam(net_w.parameters(), lr=lr)

    snapshots = [clone_net(net_w)]
    for epoch in range(epochs):
        for x, y in loader:
            opt.zero_grad()
            loss = criterion(net_w(x), y)
            loss.backward()
            opt.step()
        snapshots.append(clone_net(net_w))
        if (epoch+1)%100 == 0:
            print(f"Epoch {epoch+1}/{epochs} completed")
    return snapshots


def pred_dif(net1, net2, n=10):
    """max |net1(x) - net2(x)| over n fixed points in (0,1)."""
    # Generate exactly n evenly spaced points in [0,1]
    xs = torch.linspace(0.0, 1.0, n+1, device=device).unsqueeze(1)
    with torch.no_grad():
        diffs = torch.abs(net1(xs) - net2(xs))
        return diffs.max().item()

def pred(net, x=0.618):
    xt = torch.tensor([[x]], device=device)
    with torch.no_grad():
        # return torch.abs(net1(xt)-net2(xt)).item()
        return net(xt).item()


if __name__ == '__main__':
    # Determinism
    seed = 0
    fix_random_seed(seed)

    # Hyperparameters
    d = 2000
    dstop = 1000
    k = 3
    tol = 1e-12
    lr = 0.01
    epochs = 10
    batch_size = 64

    # Data
    dataset = loaddata()

    # 1) Original network
    net_orig = TwoLayerNet(input_dim=1, hidden_dim=d).to(device)

    # 2) Build weighted (compressed) network using Compressor-derived weights
    net_cp, weights_t = compress_nn(net_orig, dstop=dstop, k=k, tol=tol)
    print(f'Compression completed. d={d}->dstop={dstop}')

    # 3) Train both with identical minibatches/order
    snaps_orig = train_orig(net_orig, dataset, epochs=epochs, batch_size=batch_size, lr=lr, seed=seed)
    snaps_cp   = train_weighted(net_cp, weights_t, dataset, epochs=epochs, batch_size=batch_size, lr=lr, seed=seed)

    # 4) Compare predictions per epoch
    diffs = [pred_dif(s_o, s_c, n=20) for s_o, s_c in zip(snaps_orig, snaps_cp)]
    pred1 = [pred(net) for net in snaps_orig]
    pred2 = [pred(net) for net in snaps_cp]

    # 5) Plot: two subplots
    fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    # Upper: predictions at x=0.618
    axs[0].plot(range(0, epochs + 1), pred1, marker='o', label=f'd={d} original dynamics')
    axs[0].plot(range(0, epochs + 1), pred2, marker='^', label=f'dstop={dstop} compressed dynamics')
    axs[0].set_ylabel('prediction at x=0.618')
    axs[0].set_title('Prediction vs. epoch')
    axs[0].grid(True, linewidth=0.25)
    axs[0].legend()

    # Lower: pred_dif vs epoch
    axs[1].plot(range(0, epochs + 1), diffs, marker='s', color='tab:red')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('max |net_orig(x) - net_cp(x)|')
    # axs[1].set_title('Prediction difference vs. epoch')
    axs[1].grid(True, linewidth=0.25)

    plt.tight_layout()
    plt.show()

'''
- Initialize a TwoLayerNet net_orig, hidden layer dim = d
- w_orig, b2 = extract(net_orig)
- cp = Compressor(w_orig, tol=tol)
- weights = cp.compress_weights(k, dstop=dstop)
- alive = np.nonzero(weights > tol)[0]
- Complete the code of a new class called WeightedTwoLayerNet. It has the same structure as TwoLayerNet, but each model output is calculated as self.fc2(weights * self.tanh(self.fc1(x))), where weights * tanh is element-wise product and the result is a vector of dimension d
- Instantiate this as net_cp

- Write two functions for the following two training. 
- Train net_orig as follows:
    lr = 0.5, epochs = 0...5, batch_size = 64, loss function = MSE(net(x), y), optimizer=SGD. return a copy of net after each epoch (when epoch=1 return the untrained net)
- Train net_cp as follows. Make sure that the SGD chooses the exact same minibatches as the previous case by resetting random seeds!!!
    same hyperparams. Also, return a copy of net after each epoch. But change the SGD update rule. Say the usual SGD rule is w_j <- w_j - \eta \partial(loss)/\partial{w_j}. I want the update rule for b2 to be still this. But for w=W1, W2, b1, use the update rule w_j <- w_j - \eta/c_j \partial(loss)/\partial{w_j} for j in alive; If j is not in alive, don't update. Here, j=range(d) and note that W1, W2, b1 each is a d dimensional vector. Try to this update rule compact to best exert the power of pytorch

- Finally, compare results. Define a function pred_dif(net1, net2) = max|net1(x) - net2(x)|  over 10 randomly chosen x's in the range 0~1
- Plot pred_dif vs epoch for epoch in 0~5
'''