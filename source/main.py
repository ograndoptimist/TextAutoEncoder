import torch.optim as optim

from source.model.autoencoder import AutoEncoder
from source.model.train_model import train_model


def run_main():
    model = AutoEncoder()

    optimizer = optim.Adam()


if __name__ == '__main__':
    run_main()
