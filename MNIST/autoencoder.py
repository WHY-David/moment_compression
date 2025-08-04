import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os



# # Import compression
# from moment_matching import 
# import sys, os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

class Autoencoder(nn.Module):
    def __init__(self, input_dim=28*28, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )
    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec, z

def MNIST_autoencoder(latent_dim = 16, batch_size = 256):
    # Learning rate schedule: list of (lr, num_epochs)
    schedule = [
        (0.01, 20),
        (0.002, 50),
        (0.001, 50),
        (0.0002, 100),
        (0.0001, 200)
    ]
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    train_ds = datasets.MNIST(root='data', train=True, download=False, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    net = Autoencoder(latent_dim=latent_dim).to(device)
    criterion = nn.MSELoss()
    # initialize optimizer with first schedule learning rate
    optimizer = optim.Adam(net.parameters(), lr=schedule[0][0])
    total_epochs = sum(ep for _, ep in schedule)
    losses = []
    epoch_counter = 0

    # 4. Train Autoencoder with LR schedule
    net.train()
    for lr_val, num_epochs in schedule:
        for g in optimizer.param_groups:
            g['lr'] = lr_val
        for _ in range(num_epochs):
            epoch_counter += 1
            epoch_loss = 0.0
            for xb, _ in train_loader:
                xb = xb.to(device)
                optimizer.zero_grad()
                x_rec, _ = net(xb)
                loss = criterion(x_rec, xb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            epoch_loss /= len(train_ds)
            print(f"Epoch {epoch_counter}/{total_epochs}, LR: {lr_val}, Loss: {epoch_loss:.6f}")
            losses.append(epoch_loss)

    net.eval()

    return net, train_loader, losses

if __name__ == '__main__':
    # Run training and capture losses
    net, train_loader, losses = MNIST_autoencoder(latent_dim=16, batch_size=256)
    # Plot training loss vs. epoch
    plt.figure()
    plt.plot(range(1, len(losses)+1), losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss vs. Epoch')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # Save encoder & decoder state
    save_path = os.path.join(os.path.dirname(__file__), 'autoencoder.pth')
    torch.save(net.state_dict(), save_path)
    print(f"Saved autoencoder state_dict to {save_path}")
