# Give a huge data set. compress to weighted atoms
# effective m=1 for y=sin(x)? A crossover to m=2 if add noise?

import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
from moment_matching import compress, compress_naive, multi_exponents, all_moments

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

def generate_data(d, seed=0, noise=0.):
    np.random.seed(seed)
    data_x = np.random.rand(d,1)
    data = np.hstack([data_x, np.sin(2*np.pi*data_x)+noise*np.random.randn(d,1)])  # shape (d, 2)
    return data






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
    hidden_dim = 100
    lr = 1e-2
    epochs = 1000

    # training mode: full-batch (False) or weighted SGD (True)
    use_sgd = True
    batch_size = 64

    # generate weighted training set via moment-matching compression
    data_full = np.loadtxt('noisy_sin_10000.csv', delimiter=',')
    c_, w_ = compress(data_full, k=3, dstop=50)
    print("Compression finished. ")
    x = torch.from_numpy(w_[:, [0]]).float().to(device)
    y = torch.from_numpy(w_[:, [1]]).float().to(device)
    weights = torch.from_numpy(c_).float()  # positive weights

    # instantiate
    net = TwoLayerNet(1, hidden_dim).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)    

    if use_sgd:
        sampler = torch.utils.data.WeightedRandomSampler(
            weights,          # uniform weights
            num_samples=len(data_full),
            replacement=True  # True allows repeats within a batch
        )

        loader = DataLoader(
            TensorDataset(x, y),
            batch_size=batch_size,
            sampler=sampler,  # sampler OR shuffle, not both
            num_workers=0,    # optional
        )

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
    else:
        for epoch in range(1, epochs+1):
            # # full-batch training
            # pred = net(x.to(device))
            # # compute weighted MSE over the full set
            # per_sample_mse = ((pred - y.to(device)).view(-1, pred.size(-1))**2).mean(dim=1)
            # loss = (w.to(device) * per_sample_mse).sum()/w.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if epoch % 100 == 0:
                print(f"Epoch {epoch:4d}  Loss: {loss.item():.4f}")



    # evaluation and visualization
    with torch.no_grad():
        # prepare test grid
        x_test = torch.linspace(0, 1, 1000, device=device).unsqueeze(1)
        y_test = torch.sin(2 * torch.pi * x_test)
        y_pred = net(x_test)
        test_loss = ((y_pred - y_test) ** 2).mean()
        print(f"\nTest MSE: {test_loss.item():.4f}")

        # convert to numpy for plotting
        x_plot = x_test.cpu().numpy().squeeze()
        y_true = y_test.cpu().numpy().squeeze()
        y_pred_np = y_pred.cpu().numpy().squeeze()
        w_np = w_
        c_np = c_

        # plot ground truth, weighted data scatter, and NN output
        plt.figure()
        # ground truth sine curve
        plt.plot(x_plot, y_true, 'g--', label='sin(2πx)')
        plt.plot(x_plot, y_pred_np, color='orange', label='NN prediction')
        plt.scatter(w_np[:, 0], w_np[:, 1], s=c_np, alpha=0.6, label='compressed data')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.title('Ground Truth, Compressed Data, and NN Predictions')
        plt.show()

