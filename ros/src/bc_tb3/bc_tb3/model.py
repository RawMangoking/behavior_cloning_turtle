"""Network used by BOTH train.py and policy.py, so the architecture can't drift apart."""
import torch.nn as nn

from bc_tb3.common import N_FEATURES


def make_model():
    return nn.Sequential(
        nn.Linear(N_FEATURES, 128), nn.ReLU(),
        nn.Linear(128, 128), nn.ReLU(),
        nn.Linear(128, 2),          # linear, angular
    )
