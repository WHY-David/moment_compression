import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

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

def MNIST_autoencoder(latent_dim = 16, batch_size = 256, epochs = 30, lr = 1e-3):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    train_ds = datasets.MNIST(root='data', train=True, download=False, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    net = Autoencoder(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # 4. Train Autoencoder
    net.train()
    for epoch in range(1, epochs+1):
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
        print(f"Epoch {epoch}/{epochs}, Loss: {epoch_loss:.6f}")

    net.eval()
    return net, train_loader




# draw examples and compare original vs reconstructed
import matplotlib.pyplot as plt

net, train_loader = MNIST_autoencoder(latent_dim = 32, batch_size = 256, epochs = 30, lr = 1e-2)
# Switch to eval mode
net.eval()
examples = []
with torch.no_grad():
    for xb, _ in train_loader:
        xb = xb.to(device)
        x_rec, _ = net(xb)
        examples = [(xb[i].cpu().numpy(), x_rec[i].cpu().numpy()) for i in range(5)]
        break  # Only need first batch

fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, (orig, recon) in enumerate(examples):
    axes[0, i].imshow(orig.reshape(28, 28), cmap='gray')
    axes[0, i].set_title('Original')
    axes[0, i].axis('off')
    axes[1, i].imshow(recon.reshape(28, 28), cmap='gray')
    axes[1, i].set_title('Reconstructed')
    axes[1, i].axis('off')
plt.tight_layout()
plt.show()

# # 5. Extract latent features for entire dataset
# dev_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
# latents = []
# net.eval()
# with torch.no_grad():
#     for xb, _ in dev_loader:
#         xb = xb.to(device)
#         _, z = net(xb)
#         latents.append(z.cpu().numpy())
# latents = np.vstack(latents)  # shape (60000, dim_latent)

# # 6. Compress latent representations to weighted atoms
# # e.g., match first two moments: k=2
# c, W = compress(latents, k=2, index_type='ivf', nprobe=16)
# print("Compressed to", W.shape[0], "atoms in", W.shape[1], "D")

# # c: weights, W: latent atoms
# # To reconstruct image prototypes, map W through decoder:
# W_tensor = torch.from_numpy(W).float().to(device)
# with torch.no_grad():
#     recs = net.decoder(W_tensor).cpu().numpy()  # shape (dstop, 784)
# # recs are prototype images

# # 7. Save or visualize prototypes
# import matplotlib.pyplot as plt
# fig, axes = plt.subplots(4, 8, figsize=(8,4))
# for ax, img in zip(axes.flatten(), recs[:32]):
#     ax.imshow(img.reshape(28,28), cmap='gray')
#     ax.axis('off')
# plt.suptitle('Compressed Prototypes via Autoencoder + Moment Matching')
# plt.tight_layout()
# plt.show()