import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from matplotlib import pyplot as plt
plt.rc('font', family='Helvetica', size=8)

from common import TwoLayerNet, WeightedTwoLayerNet, compress_nn, fix_random_seed, load_data

# Device configuration
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


def make_loader(dataset, batch_size=64, seed=0):
    """Deterministic DataLoader using a fixed shuffled index order."""
    n = len(dataset)
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    sampler = torch.utils.data.SubsetRandomSampler(indices.tolist())
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=False)
    return loader



def train_orig(net, dataset, epochs=5, batch_size=64, lr=0.01, seed=0, algo=torch.optim.SGD):
    """training of the baseline network. Returns snapshots per epoch (including epoch 0)."""
    fix_random_seed(seed)
    loader = make_loader(dataset, batch_size=batch_size, seed=seed)
    criterion = nn.MSELoss()
    opt = algo(net.parameters(), lr=lr)

    snapshots = [net.clone()]  # epoch 0 (untrained)
    for epoch in range(epochs):
        for x, y in loader:
            opt.zero_grad()
            loss = criterion(net(x), y)
            loss.backward()
            opt.step()
        snapshots.append(net.clone())
        if (epoch+1)%20 == 0:
            print(f"Epoch {epoch+1}/{epochs} completed")
    return snapshots


def train_weighted(net_w: WeightedTwoLayerNet, weights_t: torch.Tensor,
                   dataset, epochs=5, batch_size=64, lr=0.01, seed=0, algo=torch.optim.SGD):
    """
    Train with the custom rule:
      - b2 uses standard SGD/Adam step
      - For j in alive: W1_j, W2_j, b1_j use w <- w - (lr / c_j) * grad
      - For j not in alive: no update
    Returns snapshots per epoch (including epoch 0).
    """
    fix_random_seed(seed)
    loader = make_loader(dataset, batch_size=batch_size, seed=seed)
    criterion = nn.MSELoss()

    # lr_scale = 1/c_j for alive, 0 for dead
    inv_c = torch.where(weights_t > tol,
                        1.0 / weights_t,
                        torch.zeros_like(weights_t))

    # Scale per-element gradients via hooks; keeps using stock SGD
    net_w.fc1.weight.register_hook(lambda grad: grad * inv_c.view(-1, 1))
    net_w.fc1.bias.register_hook(lambda grad: grad * inv_c)
    net_w.fc2.weight.register_hook(lambda grad: grad * inv_c.view(1, -1))

    opt = algo(net_w.parameters(), lr=lr)

    snapshots = [net_w.clone()]
    for epoch in range(epochs):
        for x, y in loader:
            opt.zero_grad()
            loss = criterion(net_w(x), y)
            loss.backward()
            opt.step()
        snapshots.append(net_w.clone())
        if (epoch+1)%20 == 0:
            print(f"Epoch {epoch+1}/{epochs} completed")
    return snapshots


def naive_prune(net_orig:TwoLayerNet, hidden_dim:int):
    input_dim = net_orig.fc1.weight.shape[1]

    # Temporarily disable random init for Linear
    orig_reset = nn.Linear.reset_parameters
    nn.Linear.reset_parameters = lambda self, *args, **kwargs: None
    # Instantiate on same device
    net = TwoLayerNet(input_dim, hidden_dim).to(net_orig.fc1.weight.device)
    # Restore init method
    nn.Linear.reset_parameters = orig_reset

    with torch.no_grad():
        W1 = net_orig.fc1.weight[:hidden_dim, :].clone()
        b1 = net_orig.fc1.bias[:hidden_dim].clone()
        W2 = net_orig.fc2.weight[:, :hidden_dim].clone()
        b2 = net_orig.fc2.bias.clone()

        net.fc1.weight.copy_(W1)
        net.fc1.bias.copy_(b1)
        net.fc2.weight.copy_(W2)
        net.fc2.bias.copy_(b2)
    return net


def pred_dif(net1, net2, n=10):
    """max |net1(x) - net2(x)| over n fixed points in (0,1)."""
    # Generate exactly n evenly spaced points in [0,1]
    xs = torch.linspace(0.0, 1.0, n+1, device=device).unsqueeze(1)
    with torch.no_grad():
        diffs = torch.abs(net1(xs) - net2(xs))
        return diffs.max().item()

def pred(net, x=0.618):
    xt = torch.tensor([[x]], device=device)
    with torch.no_grad():
        # return torch.abs(net1(xt)-net2(xt)).item()
        return net(xt).item()
    
def calculate_MSE(net):
    criterion = nn.MSELoss()
    loader = make_loader(dataset, batch_size=batch_size, seed=seed)
    total_loss, cnt = 0.0, 0
    with torch.no_grad():
        for x_e, y_e in loader:
            preds = net(x_e.to(device))
            total_loss += criterion(preds, y_e.to(device)).item()
            cnt += 1
    return total_loss / cnt


if __name__ == '__main__':
    # Determinism
    seed = 42
    fix_random_seed(seed)

    # Hyperparameters
    d = 2000
    dstop = 200
    k = 3
    tol = 1e-12
    lr = 0.0001
    epochs = 50
    batch_size = 64
    algo = torch.optim.Adam

    # TensorDataset
    dataset = load_data()

    # 1) Original network
    net_orig = TwoLayerNet(input_dim=1, hidden_dim=d).to(device)

    # 2) Build weighted (compressed) network using Compressor-derived weights
    net_cp, weights_t = compress_nn(net_orig, dstop=dstop, k=k, tol=tol)
    # print(weights_t.min().item())
    print(f'Compression completed. d={d} -> dstop={dstop}')
    net_naive = naive_prune(net_orig, dstop)

    # 3) Train both with identical minibatches/order
    snaps_naive = train_orig(net_naive, dataset, epochs=epochs, batch_size=batch_size, lr=lr, seed=seed, algo=algo)
    snaps_orig = train_orig(net_orig, dataset, epochs=epochs, batch_size=batch_size, lr=lr, seed=seed, algo=algo)
    snaps_cp   = train_weighted(net_cp, weights_t, dataset, epochs=epochs, batch_size=batch_size, lr=lr, seed=seed, algo=algo)

    losses_naive = [calculate_MSE(net) for net in snaps_naive]
    losses_orig  = [calculate_MSE(net) for net in snaps_orig]
    losses_cp    = [calculate_MSE(net) for net in snaps_cp]

    # 4) Compare predictions per epoch
    diffs = [pred_dif(s_o, s_c, n=20) for s_o, s_c in zip(snaps_orig, snaps_cp)]
    pred1 = [pred(net) for net in snaps_orig]
    pred2 = [pred(net) for net in snaps_cp]
    pred_naive = [pred(net) for net in snaps_naive]

    # 5) Plot: three subplots
    fig, axs = plt.subplots(3, 1, figsize=(6, 6), sharex=True)
    # ensure Helvetica font at size 8 for all ticks
    for ax in axs:
        ax.tick_params(axis='both', which='major', labelsize=8)

    # Plot MSE Loss vs. epoch (log scale)
    axs[0].plot(range(0, epochs + 1), losses_naive, color='tab:blue',  marker='d', markersize=6, label=f'Naive d\'={dstop}')
    axs[0].plot(range(0, epochs + 1), losses_orig,  color='tab:green', marker='o', markersize=6, label=f'Original d={d}')
    axs[0].plot(range(0, epochs + 1), losses_cp,    color='tab:orange',marker='^', markersize=6, label=f'Compressed d\'={dstop}')
    axs[0].set_ylabel('MSE Loss')
    axs[0].set_yscale('log')
    # axs[0].grid(True, linewidth=0.25)
    axs[0].legend()

    # Predictions at x=0.618
    axs[1].plot(range(0, epochs + 1), pred_naive, color='tab:blue',  marker='d', markersize=6, label=f'Naive d\'={dstop}')
    axs[1].plot(range(0, epochs + 1), pred1,      color='tab:green', marker='o', markersize=6, label=f'Original d={d}')
    axs[1].plot(range(0, epochs + 1), pred2,      color='tab:orange',marker='^', markersize=6, label=f'Compressed d\'={dstop}')
    axs[1].set_ylabel('prediction at x=0.618')
    # axs[1].grid(True, linewidth=0.25)
    axs[1].legend()

    # Lower: pred_dif vs epoch
    axs[2].plot(range(0, epochs + 1), diffs, color='tab:red', marker='s', markersize=6)
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel(r'$\max_x |NN_{orig}(x) - NN_{comp}(x)|$')
    # axs[2].grid(True, linewidth=0.25)

    plt.tight_layout()
    plt.savefig('training_relu_adam.pdf', format='pdf', bbox_inches='tight', pad_inches=0)
    plt.show()

'''
- Initialize a TwoLayerNet net_orig, hidden layer dim = d
- w_orig, b2 = extract(net_orig)
- cp = Compressor(w_orig, tol=tol)
- weights = cp.compress_weights(k, dstop=dstop)
- alive = np.nonzero(weights > tol)[0]
- Complete the code of a new class called WeightedTwoLayerNet. It has the same structure as TwoLayerNet, but each model output is calculated as self.fc2(weights * self.tanh(self.fc1(x))), where weights * tanh is element-wise product and the result is a vector of dimension d
- Instantiate this as net_cp

- Write two functions for the following two training. 
- Train net_orig as follows:
    lr = 0.5, epochs = 0...5, batch_size = 64, loss function = MSE(net(x), y), optimizer=SGD. return a copy of net after each epoch (when epoch=1 return the untrained net)
- Train net_cp as follows. Make sure that the SGD chooses the exact same minibatches as the previous case by resetting random seeds!!!
    same hyperparams. Also, return a copy of net after each epoch. But change the SGD update rule. Say the usual SGD rule is w_j <- w_j - eta partial(loss)/partial{w_j}. I want the update rule for b2 to be still this. But for w=W1, W2, b1, use the update rule w_j <- w_j - eta/c_j partial(loss)/partial{w_j} for j in alive; If j is not in alive, don't update. Here, j=range(d) and note that W1, W2, b1 each is a d dimensional vector. Try to this update rule compact to best exert the power of pytorch

- Finally, compare results. Define a function pred_dif(net1, net2) = max|net1(x) - net2(x)|  over 10 randomly chosen x's in the range 0~1
- Plot pred_dif vs epoch for epoch in 0~5
'''