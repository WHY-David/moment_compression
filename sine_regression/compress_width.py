import numpy as np
import torch
import torch.nn as nn

from common import TwoLayerNet, WeightedTwoLayerNet, compress_nn
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


if __name__ == '__main__':
    dstop = 30
    k = 2

    net = load_model(hidden_dim=5000)
    net_cp, weights = compress_nn(net, dstop=dstop, k=k)

    with torch.no_grad():
        # grid of inputs
        x_vals = torch.linspace(0, 1, 1000, device=device).unsqueeze(1)
        y_orig = net(x_vals)
        y_comp = net_cp(x_vals)
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
        # output_path = 'compress_width.png'
        # plt.savefig(output_path)
        # print(f"Plot saved to {output_path}")
        # plt.close()
        plt.show()
