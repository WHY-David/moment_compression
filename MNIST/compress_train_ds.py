import numpy as np
import torch
import torch.nn as nn

from autoencoder import Autoencoder

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from compressor import Compressor
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
import os


def MNIST_train_loader(batch_size=256):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    train_ds = datasets.MNIST(root='data', train=True, download=False, transform=transform)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    return loader


def compress_train_ds(return_list, tol=1e-12, latent_dim=16, batch_size=256):
    train_loader = MNIST_train_loader(batch_size=256)

    # Load pretrained autoencoder
    ae = Autoencoder(latent_dim=latent_dim).to(device)
    state_path = os.path.join(os.path.dirname(__file__), 'autoencoder.pth')
    state_dict = torch.load(state_path, map_location=device)
    ae.load_state_dict(state_dict)
    ae.eval()

    # Generate latent representations
    latents = []
    with torch.no_grad():
        for xb, _ in train_loader:
            xb = xb.to(device)
            _, z = ae(xb)
            latents.append(z.cpu().numpy())
    data_np = np.concatenate(latents, axis=0)
    assert data_np.shape == (60000, latent_dim)
    print("Autoencode complete")

    # Compress with Compressor
    cp = Compressor(data_np, tol=tol)
    weight_dict = cp.compress_weights(2, return_at=return_list)

    # # Extract alive images from original dataset
    # all_images = train_ds.data.numpy().reshape(len(train_ds), -1)
    # alive_images = all_images[alive]

    # return weights[alive], alive_images
    return weight_dict


# If run as script, run compression and print result
if __name__ == '__main__':
    # best strategy: use k=2 for d>10000; else k=1
    return_list = [200, 400, 600, 800, 1000, 1500, 2000, 4000, 6000, 8000, 10000, 20000, 30000, 40000, 50000]
    weight_dict = compress_train_ds(return_list)

    # If your keys are floats, convert them to strings (valid Python identifiers):
    to_save = {f"d_{d}": c_ for d, c_ in weight_dict.items()}

    # Or compressed
    np.savez_compressed("weights_k2.npz", **to_save)

    # weights, images = compress_train_ds(return_list, tol=1e-12, latent_dim=16, batch_size=256)
    # print(f"Compressed to {len(weights)} images with nonzero weights.")