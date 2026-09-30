"""Behavior cloning: MLP, MSE loss, normalized inputs and outputs.

Validation split is by EPISODE, not by row. Rows from the same episode are
nearly identical, so a random row split would leak and look better than it is.

    ros2 run bc_tb3 train
    ros2 run bc_tb3 train --epochs 200
"""
import argparse

import numpy as np
import torch
import torch.nn as nn

from bc_tb3.model import make_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/root/ros2_ws/data/tb3_demos.csv')
    ap.add_argument('--out', default='/root/ros2_ws/data/tb3_policy.pt')
    ap.add_argument('--epochs', type=int, default=100)
    a = ap.parse_args()

    data = np.loadtxt(a.data, delimiter=',', skiprows=1).astype(np.float32)
    episodes = data[:, 0]
    X = data[:, 1:-2]   # inputs:  24 rays, dist, heading_error
    Y = data[:, -2:]    # labels:  linear, angular

    # hold out 10% of the EPISODES for validation
    rng = np.random.default_rng(0)
    ids = rng.permutation(np.unique(episodes))
    val_ids = ids[:max(1, len(ids) // 10)]
    val_mask = np.isin(episodes, val_ids)

    # normalize using training data only
    x_mean, x_std = X[~val_mask].mean(0), X[~val_mask].std(0) + 1e-6
    y_mean, y_std = Y[~val_mask].mean(0), Y[~val_mask].std(0) + 1e-6
    Xn, Yn = (X - x_mean) / x_std, (Y - y_mean) / y_std
    Xt, Yt = torch.tensor(Xn[~val_mask]), torch.tensor(Yn[~val_mask])
    Xv, Yv = torch.tensor(Xn[val_mask]), torch.tensor(Yn[val_mask])
    print(f'{len(ids)} episodes: {len(Xt)} train rows, {len(Xv)} val rows')

    model = make_model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    best_val, best_state = float('inf'), None
    for epoch in range(1, a.epochs + 1):
        model.train()
        perm = torch.randperm(len(Xt))
        total = 0.0
        for i in range(0, len(Xt), 256):
            b = perm[i:i + 256]
            opt.zero_grad()
            loss = loss_fn(model(Xt[b]), Yt[b])
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        train_loss = total / len(Xt)   # average over the whole epoch
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(Xv), Yv).item()
        if val < best_val:   # keep the best epoch, not just the last
            best_val = val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0:
            print(f'epoch {epoch:3d}  train {train_loss:.4f}  val {val:.4f}')

    torch.save({
        'model': best_state,
        'x_mean': torch.tensor(x_mean), 'x_std': torch.tensor(x_std),
        'y_mean': torch.tensor(y_mean), 'y_std': torch.tensor(y_std),
    }, a.out)
    print(f'saved {a.out} (best val {best_val:.4f})')


if __name__ == '__main__':
    main()
