import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Synthetic linear ICL dataset
p = 5
K = 5
T = K + 1
Ntrain = 2000
Ntest = 500
epochs = 500

def sample_episode(batch):
    x = torch.randn(batch, T, p)
    w = torch.randn(batch, p)
    y = (x * w.unsqueeze(1)).sum(-1)
    noise = 0.1 * torch.randn(batch, T)
    y = y + noise
    return x, y

# Encode tokens
D = 32
Px = nn.Linear(p, D)
Py = nn.Linear(1, D)

def encode(x, y):
    B = x.size(0)
    Z = Px(x)
    y_full = y.unsqueeze(-1)
    Zy = Py(y_full)
    Zy[:, -1, :] = 0.0
    return Z + Zy

# Minimal multihead attention model
class AttnICL(nn.Module):
    def __init__(self, D, H):
        super().__init__()
        self.mha = nn.MultiheadAttention(D, H, batch_first=True)
        self.readout = nn.Linear(D, 1)
    def forward(self, Z):
        out,_= self.mha(Z,Z,Z)
        q = out[:,-1,:]
        return self.readout(q).squeeze(-1)

H=8
model = AttnICL(D,H)
opt = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

train_losses=[]
test_losses=[]

for epoch in range(epochs):
    x,y = sample_episode(Ntrain)
    Z = encode(x,y)
    pred = model(Z)
    target = y[:,-1]
    loss = loss_fn(pred, target)
    opt.zero_grad()
    loss.backward()
    opt.step()
    train_losses.append(loss.item())

    with torch.no_grad():
        x_t,y_t = sample_episode(Ntest)
        Zt = encode(x_t,y_t)
        predt = model(Zt)
        lt = loss_fn(predt, y_t[:,-1])
        test_losses.append(lt.item())

# Plot
plt.plot(train_losses, label="train")
plt.plot(test_losses, label="test")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.show()