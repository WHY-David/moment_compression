import numpy as np
import torch
from torch import nn
from typing import Optional, Union
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection side-effects)

from common import fix_random_seed, TwoLayerNet

ArrayLike = Union[np.ndarray, torch.Tensor, float, int]
Tensor = torch.Tensor


def _infer_device(x: ArrayLike, y: ArrayLike) -> torch.device:
    """Pick a sensible device given possible torch inputs; default to CPU."""
    if isinstance(x, torch.Tensor):
        return x.device
    if isinstance(y, torch.Tensor):
        return y.device
    return torch.device("cpu")

def fnet(
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
    if net is None:
        fix_random_seed(seed)
        net = TwoLayerNet(2, 200)

    if device is None:
        device = _infer_device(x, y)

    net = net.to(device).eval()

    with torch.no_grad():
        if x.shape != y.shape:
            raise ValueError(f"x and y must have the same shape; got {x.shape} vs {y.shape}")
        inp = torch.stack((x, y), dim=1)  # (N, 2)
        out = net(inp).squeeze(-1)  # (N,)

    if return_numpy:
        return out.detach().cpu().numpy()
    return out


def generate_train_data(
    size: int,
    net=None,
    f=None,
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

    # sample x, y ~ U[-1,1)
    x = torch.rand(size, device=device, dtype=torch.float32)*2-1.
    y = torch.rand(size, device=device, dtype=torch.float32)*2-1.

    # model output
    if net is not None:
        fv = fnet(x, y, net=net, device=device, return_numpy=False)  # (size,)
    elif f is not None:
        fv = f(x,y)
    else:
        raise ValueError("Either net or f must be provided")
        

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