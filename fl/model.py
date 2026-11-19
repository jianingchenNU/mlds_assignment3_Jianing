import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 62)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def init_model(seed):
    # fork_rng so the init depends only on `seed`, not on earlier global draws:
    # every compared run starts from identical weights.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return MLP()
