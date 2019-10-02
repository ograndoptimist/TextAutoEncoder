"""
    Embedding model.
"""
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self,
                 num_embeddings,
                 embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embeddings,
                                      embedding_dim=embedding_dim)

        self.init_weights()

    def init_weights(self):
        pass

    def forward(self, x):
        net = self.embedding(x)
        return net
