import torch
import torch.nn as nn
import torch.optim as optim

# 1. Define a tiny MLP with one hidden layer
class Net(nn.Module):
    def __init__(self, in_dim=10, hidden_dim=5, out_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# 2. Instantiate
net = Net()
base_lr = 0.1
optimizer = optim.SGD(net.parameters(), lr=base_lr)

# 3. Create per-neuron learning-rate scales for layer1 (size = hidden_dim)
#    For example, neuron i gets lr_i = base_lr * scales[i]
scales = torch.tensor([0.5, 1.0, 2.0, 0.1, 1.5])  # shape (hidden_dim,)

# 4. Register gradient hooks to scale gradients for fc1 weights and biases
#    fc1.weight.grad has shape (hidden_dim, in_dim)
net.fc1.weight.register_hook(lambda grad: grad * scales.unsqueeze(1))
#    fc1.bias.grad has shape (hidden_dim,)
net.fc1.bias  .register_hook(lambda grad: grad * scales)

# 5. Dummy training loop
for epoch in range(5):
    # fake data: batch_size=16, in_dim=10
    x = torch.randn(16, 10)
    y = torch.randn(16, 1)

    optimizer.zero_grad()
    pred = net(x)
    loss = nn.MSELoss()(pred, y)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch} loss = {loss.item():.4f}")