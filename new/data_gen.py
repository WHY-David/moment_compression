import numpy as np
import torch
from torch import nn
from typing import Optional, Union
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection side-effects)

from common import fix_random_seed

ArrayLike = Union[np.ndarray, torch.Tensor, float, int]
Tensor = torch.Tensor

class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(2*torch.pi*20*x)

class TwoLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

def _infer_device(x: ArrayLike, y: ArrayLike) -> torch.device:
    """Pick a sensible device given possible torch inputs; default to CPU."""
    if isinstance(x, torch.Tensor):
        return x.device
    if isinstance(y, torch.Tensor):
        return y.device
    return torch.device("cpu")


def _to_tensor(x: ArrayLike, device: Optional[torch.device] = None, dtype=torch.float32) -> Tensor:
    """Convert numpy/float/int/torch to a torch.Tensor on `device` without copying if possible."""
    if isinstance(x, torch.Tensor):
        t = x.to(dtype)
        if device is not None:
            t = t.to(device)
        return t
    return torch.as_tensor(x, dtype=dtype, device=device)


def f(
    x: ArrayLike,
    y: ArrayLike,
    seed: int = 0,
    net: Optional[torch.nn.Module] = None,
    device: Optional[torch.device] = None,
    return_numpy: bool = False,
):
    """
    Vectorized model evaluation compatible with numpy or torch inputs.

    Inputs `x` and `y` can be scalars, numpy arrays, or torch.Tensors of the same shape.
    Returns a torch.Tensor by default, or a numpy array if `return_numpy=True`.
    """
    fix_random_seed(seed)

    if net is None:
        net = TwoLayerNet(2, 200)

    if device is None:
        device = _infer_device(x, y)

    net = net.to(device).eval()

    with torch.no_grad():
        xt = _to_tensor(x, device=device).reshape(-1)
        yt = _to_tensor(y, device=device).reshape(-1)
        if xt.shape != yt.shape:
            raise ValueError(f"x and y must have the same shape; got {xt.shape} vs {yt.shape}")
        inp = torch.stack((xt, yt), dim=1)  # (N, 2)
        out = net(inp).squeeze(-1)  # (N,)

    if return_numpy:
        return out.detach().cpu().numpy()
    return out


def generate_train_data(
    size: int,
    func=f,
    noise: float = 0.0,
    seed: int = 0,
    return_tensor: bool = False,
    device: Optional[torch.device] = None,
):
    """
    Generate `size` samples (x, y, f(x, y)+noise) in [0,1)×[0,1).

    - Fully torch-compatible pipeline (sampling, evaluation, noise) to avoid numpy↔torch casts.
    - Return numpy array shape (size, 3) by default, or a torch tensor if `return_tensor=True`.
    """
    fix_random_seed(seed)

    if device is None:
        device = torch.device("cpu")

    # sample x, y ~ U[0,1)
    x = torch.rand(size, device=device, dtype=torch.float32)
    y = torch.rand(size, device=device, dtype=torch.float32)

    # model output
    fv = func(x, y, seed=seed, device=device, return_numpy=False)  # (size,)

    # add Gaussian noise
    if noise and noise > 0:
        noise_vals = torch.randn(size, device=device, dtype=fv.dtype) * noise
    else:
        noise_vals = torch.zeros(size, device=device, dtype=fv.dtype)

    fv_noisy = fv + noise_vals

    data_t = torch.stack((x, y, fv_noisy), dim=1)  # (size, 3)
    if return_tensor:
        return data_t
    return data_t.detach().cpu().numpy()


if __name__ == "__main__":
    # Example usage
    size = 1_000_000
    data = generate_train_data(size, func=f, noise=0.0, seed=42, return_tensor=False)

    # 3D scatter plot of (x, y, f(x,y))
    xs = data[:, 0]
    ys = data[:, 1]
    zs = data[:, 2]

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # ax.scatter(xs, ys, zs, s=2)
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("f(x, y)")
    # ax.set_title("Generated data: 3D scatter")
    # plt.show()

    # Uncomment to save to CSV
    output_path = 'data.csv'
    np.savetxt(output_path, data, delimiter=',', header='x,y,f', comments='')
    print(f"Data saved to {output_path} (shape: {data.shape})")