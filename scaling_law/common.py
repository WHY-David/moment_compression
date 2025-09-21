import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from matplotlib import pyplot as plt
from scipy.special import jv

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from compressor import Compressor

class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(2*torch.pi*20*x)

class TwoLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, init_uniform=None, bias_uniform=None, activation=nn.ReLU):
        """
        init_uniform: None (use PyTorch defaults) or
                      float r  -> initialize weights/biases ~ U(-r, r)
                      (a, b)   -> initialize weights/biases ~ U(a, b)
        bias_uniform: None -> use `init_uniform` for biases; otherwise same semantics as above.
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = activation()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # Optional custom initialization
        if (init_uniform is not None) or (bias_uniform is not None):
            self._custom_init(init_uniform, bias_uniform)

    def _custom_init(self, w_range=None, b_range=None):
        def _parse(r):
            if r is None:
                return None
            if isinstance(r, (tuple, list)) and len(r) == 2:
                a, b = float(r[0]), float(r[1])
            else:
                a, b = -float(r), float(r)
            return a, b

        w_ab = _parse(w_range)
        b_ab = _parse(b_range) if b_range is not None else w_ab
        with torch.no_grad():
            if w_ab is not None:
                nn.init.uniform_(self.fc1.weight, *w_ab)
                nn.init.uniform_(self.fc2.weight, *w_ab)
            if b_ab is not None:
                nn.init.uniform_(self.fc1.bias, *b_ab)
                nn.init.uniform_(self.fc2.bias, *b_ab)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class WeightedTwoLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, weights=None, init_uniform=None, bias_uniform=None, activation=nn.ReLU):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = activation()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # weights is a non-trainable multiplicative mask/scaling per hidden unit
        if weights is None:
            w = torch.ones(hidden_dim, dtype=torch.float32)
        else:
            w = torch.as_tensor(weights, dtype=torch.float32)
            if w.numel() != hidden_dim:
                raise ValueError(f"weights length {w.numel()} != hidden_dim {hidden_dim}")
        self.register_buffer('weights', w.view(-1))

        # Optional custom initialization
        if (init_uniform is not None) or (bias_uniform is not None):
            self._custom_init(init_uniform, bias_uniform)

    def _custom_init(self, w_range=None, b_range=None):
        def _parse(r):
            if r is None:
                return None
            if isinstance(r, (tuple, list)) and len(r) == 2:
                a, b = float(r[0]), float(r[1])
            else:
                a, b = -float(r), float(r)
            return a, b

        w_ab = _parse(w_range)
        b_ab = _parse(b_range) if b_range is not None else w_ab
        with torch.no_grad():
            if w_ab is not None:
                nn.init.uniform_(self.fc1.weight, *w_ab)
                nn.init.uniform_(self.fc2.weight, *w_ab)
            if b_ab is not None:
                nn.init.uniform_(self.fc1.bias, *b_ab)
                nn.init.uniform_(self.fc2.bias, *b_ab)

    def forward(self, x):
        h = self.act(self.fc1(x))                 # (batch, d)
        h = h * self.weights.view(1, -1)          # elementwise scale by c_j
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
    Extracts parameters from a TwoLayerNet as arrays:
    w_ with shape (hidden, input+output+1) where d is hidden_dim, and scalar bias c.
    """
    with torch.no_grad():
        W1 = net.fc1.weight.data.cpu().numpy()              # shape (hidden, input)
        b1 = net.fc1.bias.data.cpu().numpy().reshape(-1,1)  # shape (hidden, 1)
        W2 = net.fc2.weight.data.cpu().numpy().transpose()  # shape (hidden, output)
        b2 = net.fc2.bias.data.cpu().numpy()
        # Assert dimensions agree
        assert W1.shape[0] == W2.shape[0] == b1.shape[0], "Dimension mismatch in W1, W2, b1"
        # Stack into big matrix: columns [W2, W1, b1]
        w_ = np.concatenate([W2, W1, b1], axis=1)
    return w_, b2


def compress_nn(net: TwoLayerNet, k=1, dstop=100, tol=1e-12, print_progress=True):
    """
    Use the Compressor to compute per-unit weights c_j, then build a
    WeightedTwoLayerNet that keeps the original width but applies the
    multiplicative weights. Also return the alive indices (c_j > tol).
    """
    device = net.fc1.weight.device
    w_orig, b2 = extract(net)
    cp = Compressor(w_orig, tol=tol)
    weights, w_cp = cp.compress(k, dstop=dstop, print_progress=print_progress)

    # Infer original dims
    input_dim = net.fc1.in_features
    output_dim = net.fc2.out_features

    # Build compressed network with correct input dim
    net_cp = WeightedTwoLayerNet(input_dim=input_dim, hidden_dim=dstop, weights=weights).to(device)

    # w_cp columns are [W2 (dstop, output_dim), W1 (dstop, input_dim), b1 (dstop, 1)]
    W2_cp = w_cp[:, :output_dim]                            # (dstop, out)
    W1_cp = w_cp[:, output_dim:output_dim + input_dim]      # (dstop, in)
    b1_cp = w_cp[:, output_dim + input_dim]                 # (dstop,)

    with torch.no_grad():
        # fc1: (hidden, in)
        net_cp.fc1.weight.copy_(torch.from_numpy(W1_cp.astype(np.float32)).to(device))
        net_cp.fc1.bias.copy_(torch.from_numpy(b1_cp.astype(np.float32)).to(device))
        # fc2: weight (out, hidden) is the transpose of stored (hidden, out)
        net_cp.fc2.weight.copy_(torch.from_numpy(W2_cp.astype(np.float32).T).to(device))
        net_cp.fc2.bias.copy_(torch.from_numpy(b2.astype(np.float32)).to(device))

    weights_t = torch.as_tensor(weights, dtype=torch.float32, device=device)
    return net_cp, weights_t


def fix_random_seed(seed=0):
    # Set a fixed random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # CUDA requires a workspace setting for deterministic cuBLAS kernels
    if torch.cuda.is_available():
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

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

def cyl_harmonic(x, y, n:int=6, k:int=20):
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
    result = jv(n, k * r) * np.cos(n * theta)

    # Convert result back to torch tensor, preserving dtype and device if possible
    if torch.is_tensor(x):
        result = torch.from_numpy(result).to(x.device).type_as(x)
    else:
        result = torch.from_numpy(result)
    return result

def make_canvas(
    axes_width_pt: float = 300.0,
    axes_aspect: float = 2/3,
    left_pt: float = 40.0,
    right_pt: float = 20.0,
    bottom_pt: float = 35.0,
    top_pt: float = 20.0,
    fontsize: float = 8.0,
    rows: int = 1,
    cols: int = 1,
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

    # ax.grid(True, which="both", linestyle=":", linewidth=0.5)
    return plt.subplots(rows, cols, figsize=(fig_w_pt/_PT_PER_IN, fig_h_pt/_PT_PER_IN), sharex=True)
