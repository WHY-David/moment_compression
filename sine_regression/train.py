import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import os

# # Device configuration
# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# model: input → hidden Tanh → output
class TwoLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        return self.fc2(self.tanh(self.fc1(x)))
    


if __name__ == '__main__':
    # device (uses MPS on M1/M2/M4 Macs if available)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # hyperparams
    hidden_dim = 5000
    lr = 5e-3
    batch_size = 32
    epochs = 1000

    # data
    N = 2000
    x = torch.rand(N, 1, device=device)
    y = torch.sin(2 * torch.pi * x)
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    # instantiate
    net = TwoLayerNet(1, hidden_dim).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # training loop
    for epoch in range(1, epochs+1):
        for xb, yb in loader:
            pred = net(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d}  Loss: {loss.item():.4f}")

    # Save trained model
    model_path = 'sine_model.pth'
    torch.save(net.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # evaluation
    with torch.no_grad():
        x_test = torch.linspace(0,1,1000, device=device).unsqueeze(1)
        y_test = torch.sin(2 * torch.pi * x_test)
        y_pred = net(x_test)
        test_loss = loss_fn(y_pred, y_test)
        print(f"\nTest MSE: {test_loss.item():.4f}")

        # Plot true function and NN predictions
        x_plot = x_test.cpu().numpy().squeeze()
        y_true = y_test.cpu().numpy().squeeze()
        y_pred_np = y_pred.cpu().numpy().squeeze()

        plt.figure()
        plt.plot(x_plot, y_true, label='sin(2πx)')
        plt.plot(x_plot, y_pred_np, label='NN prediction')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.title('True vs NN Predictions')
        plt.show()

