"""
    An Encoder based on Dense layers.
"""
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.dense_1 = nn.Linear(in_features=input_dim, out_features=24)
        self.dense_2 = nn.Linear(in_features=24, out_features=12)

    def forward(self, x):
        net = F.relu(self.dense_1(x))
        net = F.relu(self.dense_2(net))
        return net
