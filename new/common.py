import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from matplotlib import pyplot as plt

import sys, os
from compressor import Compressor

class TwoLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

    def clone(self):
        """Return a deep copy of this TwoLayerNet without consuming RNG."""
        # Dimensions
        input_dim = self.fc1.weight.shape[1]
        hidden_dim = self.fc1.weight.shape[0]
        # Temporarily disable random init for Linear
        orig_reset = nn.Linear.reset_parameters
        nn.Linear.reset_parameters = lambda self, *args, **kwargs: None
        # Instantiate on same device
        new_net = TwoLayerNet(input_dim, hidden_dim).to(self.fc1.weight.device)
        # Restore init method
        nn.Linear.reset_parameters = orig_reset
        # Copy parameters exactly
        with torch.no_grad():
            new_net.fc1.weight.copy_(self.fc1.weight)
            new_net.fc1.bias.copy_(self.fc1.bias)
            new_net.fc2.weight.copy_(self.fc2.weight)
            new_net.fc2.bias.copy_(self.fc2.bias)
        return new_net
    
class WeightedTwoLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, weights=None):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
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
        h = self.relu(self.fc1(x))                 # (batch, d)
        h = h * self.weights.view(1, -1)           # elementwise scale by c_j
        return self.fc2(h)

    def clone(self):
        """Return a deep copy of this WeightedTwoLayerNet without consuming RNG."""
        # Dimensions and buffer
        input_dim = self.fc1.weight.shape[1]
        hidden_dim = self.fc1.weight.shape[0]
        weights_buf = self.weights.detach().clone()
        # Disable random init for Linear
        orig_reset = nn.Linear.reset_parameters
        nn.Linear.reset_parameters = lambda self, *args, **kwargs: None
        new_net = WeightedTwoLayerNet(input_dim, hidden_dim, weights=weights_buf).to(self.fc1.weight.device)
        nn.Linear.reset_parameters = orig_reset
        # Copy parameters and buffer
        with torch.no_grad():
            new_net.fc1.weight.copy_(self.fc1.weight)
            new_net.fc1.bias.copy_(self.fc1.bias)
            new_net.fc2.weight.copy_(self.fc2.weight)
            new_net.fc2.bias.copy_(self.fc2.bias)
            new_net.weights.copy_(self.weights)
        return new_net

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
        W1 = net.fc1.weight.data.cpu().numpy().reshape(-1)  # shape (d,)
        b1 = net.fc1.bias.data.cpu().numpy()               # shape (d,)
        W2 = net.fc2.weight.data.cpu().numpy().reshape(-1)  # shape (d,)
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
    device = net.fc1.weight.device
    # Extract [W2_j, W1_j, b1_j] from net (shape (d,3))
    w_orig, b2 = extract(net)
    cp = Compressor(w_orig, tol=tol)
    weights, w_cp = cp.compress(k, dstop=dstop)

    net_cp = WeightedTwoLayerNet(input_dim=1, hidden_dim=dstop, weights=weights).to(device)

    with torch.no_grad():
        net_cp.fc1.weight.copy_(torch.from_numpy(w_cp[:, 1].astype(np.float32).reshape(-1, 1)).to(device))  # W1
        net_cp.fc1.bias.copy_(torch.from_numpy(w_cp[:, 2].astype(np.float32)).to(device))                  # b1
        net_cp.fc2.weight.copy_(torch.from_numpy(w_cp[:, 0].astype(np.float32).reshape(1, -1)).to(device)) # W2
        net_cp.fc2.bias.copy_(torch.tensor([b2], dtype=torch.float32, device=device))                      # b2

    weights_t = torch.as_tensor(weights, dtype=torch.float32, device=device)
    return net_cp, weights_t


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


def load_data(filename):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    data = np.loadtxt(filename, delimiter=',')
    x = torch.from_numpy(data[:, [0]]).float().to(device)
    y = torch.from_numpy(data[:, [1]]).float().to(device)
    return TensorDataset(x, y)

def make_canvas(
    axes_width_pt: float = 300.0,
    axes_aspect: float = 2/3,
    left_pt: float = 40.0,
    right_pt: float = 20.0,
    bottom_pt: float = 35.0,
    top_pt: float = 20.0,
    fontsize: float = 8.0,
):
    _PT_PER_IN = 72.0
    # Use PDF “base 14” fonts (Helvetica) — no TTF embedding, no fontTools warnings
    plt.rcParams.update({
        "pdf.use14corefonts": True,   # key line
        "ps.useafm": True,            # for .ps if you ever use it
        # Do NOT set pdf.fonttype/ps.fonttype when using core fonts
        "text.usetex": False,         # set True only if you want LaTeX (see Option C)
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"],
        "font.size": fontsize,
        "axes.titlesize": fontsize,
        "axes.labelsize": fontsize,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
        "legend.fontsize": fontsize,
    })
    # Make math text look sans-ish to match Helvetica
    plt.rcParams.update({
        "mathtext.fontset": "stixsans",
    })

    axes_h_pt = axes_width_pt * float(axes_aspect)
    fig_w_pt = left_pt + axes_width_pt + right_pt
    fig_h_pt = bottom_pt + axes_h_pt + top_pt

    fig = plt.figure(figsize=(fig_w_pt/_PT_PER_IN, fig_h_pt/_PT_PER_IN))
    ax = fig.add_axes([
        left_pt/fig_w_pt,
        bottom_pt/fig_h_pt,
        axes_width_pt/fig_w_pt,
        axes_h_pt/fig_h_pt,
    ])
    # ax.grid(True, which="both", linestyle=":", linewidth=0.5)
    return fig, ax