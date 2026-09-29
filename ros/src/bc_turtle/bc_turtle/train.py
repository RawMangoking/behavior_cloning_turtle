import numpy as np
import torch
import torch.nn as nn

DATA = '/root/ros2_ws/data/demos.csv'
OUT = '/root/ros2_ws/data/policy.pt'

data = np.loadtxt(DATA, delimiter=',', skiprows=1).astype(np.float32)
X = data[:, :2]   # inputs:  dist, heading_error
Y = data[:, 2:]   # labels:  linear, angular

# normalize so inputs and outputs have a similar scale
x_mean, x_std = X.mean(0), X.std(0) + 1e-6
y_mean, y_std = Y.mean(0), Y.std(0) + 1e-6
Xn = (X - x_mean) / x_std
Yn = (Y - y_mean) / y_std

# shuffle, then hold out 10% to check the model on data it never trained on
rng = np.random.default_rng(0)
idx = rng.permutation(len(Xn))
split = int(0.9 * len(Xn))
train_idx, val_idx = idx[:split], idx[split:]
Xt, Yt = torch.tensor(Xn[train_idx]), torch.tensor(Yn[train_idx])
Xv, Yv = torch.tensor(Xn[val_idx]), torch.tensor(Yn[val_idx])

model = nn.Sequential(
    nn.Linear(2, 64), nn.ReLU(),
    nn.Linear(64, 64), nn.ReLU(),
    nn.Linear(64, 2),
)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(1, 101):
    perm = torch.randperm(len(Xt))
    for i in range(0, len(Xt), 256):
        b = perm[i:i + 256]
        opt.zero_grad()
        loss = loss_fn(model(Xt[b]), Yt[b])
        loss.backward()
        opt.step()
    if epoch % 10 == 0:
        with torch.no_grad():
            val = loss_fn(model(Xv), Yv).item()
        print(f'epoch {epoch:3d}  train {loss.item():.4f}  val {val:.4f}')

torch.save({
    'model': model.state_dict(),
    'x_mean': torch.tensor(x_mean), 'x_std': torch.tensor(x_std),
    'y_mean': torch.tensor(y_mean), 'y_std': torch.tensor(y_std),
}, OUT)
print(f'saved {OUT} trained on {len(train_idx)} samples')
