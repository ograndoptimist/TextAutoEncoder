"""
    A Decoder based on Dense units.
"""
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self,
                 decoder_input,
                 decoder_output):
        super().__init__()
        self.dense_1 = nn.Linear(in_features=decoder_input, out_features=1)
        self.dense_2 = nn.Linear(in_features=1, out_features=12)
        self.dense_3 = nn.Linear(in_features=12, out_features=24)
        self.output = nn.Linear(in_features=24, out_features=decoder_output)

    def forward(self, x):
        net = F.relu(self.dense_1(x))
        net = F.relu(self.dense_2(net))
        net = F.relu(self.dense_3(net))
        net = F.relu(self.output(net))
        return net
