"""
    Connects Embedding, Encoder and Decoder models.
"""
import torch.nn as nn

from source.model.embedding import Embedding
from source.model.encoder import Encoder
from source.model.decoder import Decoder


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = Embedding()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        net = self.embedding(x)
        net = self.encoder(net)
        net = self.decoder(net)
        return net
