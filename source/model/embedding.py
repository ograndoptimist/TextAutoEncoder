"""
    Embedding model.
"""
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self,
                 input_dim,
                 embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=input_dim,
                                      embedding_dim=embedding_dim)

    def forward(self, x):
        net = self.embedding(x)
        return net
