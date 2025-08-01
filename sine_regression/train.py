import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import os
import numpy as np

from common import TwoLayerNet, load_data


if __name__ == '__main__':
    # device (uses MPS on M1/M2/M4 Macs if available)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # hyperparams
    hidden_dim = 5000
    lr = 1e-2
    batch_size = 64
    epochs = 1000

    # data
    dataset = load_data()
    # uniform weights
    weights = torch.ones(len(dataset), device='cpu')
    # draw len(weights) indices *with replacement* according to weights
    sampler = torch.utils.data.WeightedRandomSampler(
        weights,          # uniform weights
        num_samples=len(dataset),
        replacement=True  # True allows repeats within a batch
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,  # sampler OR shuffle, not both
        num_workers=0,    # optional
    )

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

        # Plot true function, NN predictions, and training data
        x_plot = x_test.cpu().numpy().squeeze()
        y_true = y_test.cpu().numpy().squeeze()
        y_pred_np = y_pred.cpu().numpy().squeeze()

        plt.figure()
        plt.plot(x_plot, y_true, 'g--', label='sin(2πx)')
        plt.plot(x_plot, y_pred_np, color='orange', label='NN prediction')
        x, y = dataset.tensors
        plt.scatter(x.cpu().numpy().squeeze(), y.cpu().numpy().squeeze(), s=10, alpha=0.3, label='Original training data')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.title('True vs NN Predictions')
        plt.show()

