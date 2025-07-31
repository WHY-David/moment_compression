import numpy as np
import torch
import torch.nn as nn

from train import TwoLayerNet
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from compressor import Compressor

# Device configuration
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')



def load_model(model_path='sine_model.pth', hidden_dim=1000):
    """
    Loads the trained TwoLayerNet for sin(2πx) regression.
    
    Args:
        model_path (str): Path to the .pth file.
        hidden_dim (int): Width of the hidden layer (should match training).
    Returns:
        model (nn.Module): Loaded model in eval mode.
    """
    model = TwoLayerNet(input_dim=1, hidden_dim=hidden_dim).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


# --- Extraction and compression utilities ---
def extract(net):
    """
    Extracts parameters from a trained TwoLayerNet as arrays:
    w_ with shape (d, 3) where d is hidden_dim, and scalar bias c.
    Columns of w_: [a_i, w_i, b_i].
    """
    # fc1: weight shape (d,1), bias shape (d,)
    # fc2: weight shape (1,d), bias scalar
    with torch.no_grad():
        a = net.fc2.weight.data.cpu().numpy().reshape(-1)  # shape (d,)
        w1 = net.fc1.weight.data.cpu().numpy().reshape(-1)  # shape (d,)
        b1 = net.fc1.bias.data.cpu().numpy()               # shape (d,)
        c = net.fc2.bias.data.item()
    # Stack into (d, 3)
    w_ = np.stack([a, w1, b1], axis=1)
    return w_, c


def compress_nn(net, dstop=100, k=1):
    """
    Compresses the two-layer net to width dstop
    Returns a new TwoLayerNet with hidden_dim=dstop.
    """
    # Extract original weights
    w_orig, c = extract(net)
    cp = Compressor(w_orig)
    c_pruned, w_pruned = cp.compress(k, dstop=dstop)
    # Build new network
    new_net = TwoLayerNet(input_dim=1, hidden_dim=dstop).to(device)
    new_net.eval()
    # Assign parameters
    # fc1 weights and bias
    w_pruned = w_pruned.T
    w1_new = torch.from_numpy(w_pruned[1]).float().view(dstop, 1).to(device)
    b1_new = torch.from_numpy(w_pruned[2]).float().to(device)
    new_net.fc1.weight.data.copy_(w1_new)
    new_net.fc1.bias.data.copy_(b1_new)
    # fc2 weights and bias: a_j = c_pruned[j] * w_pruned[0,j]
    a_new = torch.from_numpy(c_pruned * w_pruned[0]).float().view(1, dstop).to(device)
    new_net.fc2.weight.data.copy_(a_new)
    new_net.fc2.bias.data.fill_(c)
    return new_net

if __name__ == '__main__':
    dstop = 20
    k = 2

    net = load_model(hidden_dim=5000)
    compressed_net = compress_nn(net, dstop=dstop, k=k)

    with torch.no_grad():
        # grid of inputs
        x_vals = torch.linspace(0, 1, 1000, device=device).unsqueeze(1)
        y_orig = net(x_vals)
        y_comp = compressed_net(x_vals)
        max_diff = (y_orig - y_comp).abs().max().item()
        print(max_diff)

        import matplotlib.pyplot as plt

        x_np = x_vals.cpu().numpy().flatten()
        y_orig_np = y_orig.cpu().numpy().flatten()
        y_comp_np = y_comp.cpu().numpy().flatten()

        plt.figure(figsize=(6, 4))
        plt.plot(x_np, y_orig_np, label='Original')
        plt.plot(x_np, y_comp_np, label='Compressed', linestyle='--')
        plt.title(f'Max abs difference: {max_diff:.2e}, dstop={dstop}, k={k}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.tight_layout()
        output_path = 'comparison.png'
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
        plt.close()
