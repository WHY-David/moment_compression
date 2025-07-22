import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 1. Data: download & wrap in DataLoader
transform = transforms.Compose([
    transforms.ToTensor(),                       # converts to [0,1] tensor
    transforms.Normalize((0.1307,), (0.3081,)),  # mean & std of MNIST
])
train_ds = datasets.MNIST(root='./data',
                         train=True,
                         download=True,
                         transform=transform)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

# Prepare test dataset
test_ds = datasets.MNIST(root='./data',
                         train=False,
                         download=True,
                         transform=transform)
test_loader = DataLoader(test_ds, batch_size=1000, shuffle=False)

# 2. Model: simple CNN
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*7*7, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)

# 3. Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 4. Multi-epoch training and visualization
num_epochs = 5
train_losses = []

for epoch in range(1, num_epochs+1):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader, start=1):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    print(f'Epoch {epoch}/{num_epochs}  Loss: {epoch_loss:.4f}')

# Plot training loss vs epoch
plt.figure()
plt.plot(range(1, num_epochs+1), train_losses, marker='o')
plt.title('Training Loss vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('train_loss.png')
print('Saved plot train_loss.png')

# 5. Evaluate and visualize test examples
model.eval()
with torch.no_grad():
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, preds = torch.max(outputs, 1)

correct = preds == labels
correct_idx = torch.nonzero(correct).flatten()[:6]
incorrect_idx = torch.nonzero(~correct).flatten()[:6]

plt.figure(figsize=(12, 4))
# Plot correct predictions
for i, idx in enumerate(correct_idx):
    plt.subplot(2, 6, i+1)
    plt.imshow(images[idx].cpu().squeeze(), cmap='gray')
    plt.title(f'pred:{preds[idx].item()} true:{labels[idx].item()}')
    plt.axis('off')
# Plot incorrect predictions
for i, idx in enumerate(incorrect_idx):
    plt.subplot(2, 6, 6 + i + 1)
    plt.imshow(images[idx].cpu().squeeze(), cmap='gray')
    plt.title(f'pred:{preds[idx].item()} true:{labels[idx].item()}')
    plt.axis('off')
plt.tight_layout()
plt.savefig('test_examples.png')
print('Saved plot test_examples.png')