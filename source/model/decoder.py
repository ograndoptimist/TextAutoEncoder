"""
    A Decoder based on Dense units.
"""
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self,
                 encoder_output,
                 data_dim):
        super().__init__()
        self.dense_1 = nn.Linear(in_features=encoder_output, out_features=1)
        self.dense_1 = nn.Linear(in_features=1, out_features=12)
        self.output = nn.Linear(in_features=12, out_features=data_dim)

    def forward(self, x):
        net = F.relu(self.dense_1(x))
        net = F.relu(self.dense_2(net))
        net = F.relu(self.output(net))
        return net
