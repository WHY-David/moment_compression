import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

def compute_loss(net, loader, loss_fn):
    net.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)

def compute_accuracy(net, loader):
    net.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

def show_examples():
    # 6. Show some example predictions
    n_examples = 10
    examples = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, preds = torch.max(outputs, 1)
            for i in range(n_examples):
                examples.append((images[i].cpu().squeeze().numpy(), preds[i].cpu().item(), labels[i].cpu().item()))
            break

    plt.figure(figsize=(15, 6))
    for idx, (img, pred, true) in enumerate(examples):
        plt.subplot(2, 5, idx+1)
        plt.imshow(img, cmap='gray')
        plt.title(f"Prediction: {pred}\nTruth: {true}")
        plt.axis('off')
    plt.suptitle('Sample MNIST Predictions')
    plt.tight_layout()
    plt.show()

# 1. Hyperparameters
epochs = 10
batch_size = 64
learning_rate = 1e-4
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# 2. Data loaders
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_ds = datasets.MNIST(root='data', train=True, download=False, transform=transform)
test_ds  = datasets.MNIST(root='data', train=False, download=False, transform=transform)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

# 3. Define a simple CNN
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

net = CNN().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

train_losses = []
test_accuracies = []

# Training loop
for epoch in range(1, epochs+1):
    net.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    avg_loss = total_loss / len(train_ds)
    print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")
    train_losses.append(avg_loss)
    test_acc = compute_accuracy(net, test_loader)
    test_accuracies.append(test_acc)
    print(f"  Test Accuracy: {test_acc:.3f}")

# Plot training loss and test accuracy vs. epoch
epochs_range = list(range(1, epochs+1))
fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
axs[0].plot(epochs_range, train_losses, marker='o', label='Training Loss')
axs[0].set_ylabel('Loss')
axs[0].grid(True)
axs[0].legend()
axs[1].plot(epochs_range, test_accuracies, marker='o', label='Test Accuracy')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Accuracy')
axs[1].grid(True)
axs[1].legend()
plt.tight_layout()
plt.show()
