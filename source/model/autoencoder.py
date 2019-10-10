"""
    Connects Embedding, Encoder and Decoder models.
"""
import torch.nn as nn

from source.model.embedding import Embedding
from source.model.encoder import Encoder
from source.model.decoder import Decoder


class AutoEncoder(nn.Module):
    def __init__(self,
                 input_dim,
                 embedding_dim,
                 decoder_input):
        super().__init__()
        self.embedding = Embedding(input_dim=input_dim,
                                   embedding_dim=embedding_dim)
        self.encoder = Encoder(input_dim=input_dim)
        self.decoder = Decoder(encoder_output=decoder_input,
                               data_dim=input_dim)

    def forward(self, x):
        net = self.embedding(x)
        net = self.encoder(net)
        net = self.decoder(net)
        return net
