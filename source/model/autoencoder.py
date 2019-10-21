"""
    Connects Embedding, Encoder and Decoder models.
"""
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self,
                 embedding,
                 encoder,
                 decoder):
        super().__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        net = self.embedding(x)
        net = self.encoder(net)
        net = self.decoder(net)
        return net
