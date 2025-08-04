import numpy as np
import torch
from torchvision import datasets, transforms
import torch.nn as nn

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

archive = np.load("weights.npz")
weights_dict = {float(name.split("_",1)[1]): archive[name] for name in archive}

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Define various MNIST loaders: plain loader, weighted loader, uniform loader among only a part of 60000 images


import matplotlib.pyplot as plt

