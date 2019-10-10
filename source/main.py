import torch
import torch.nn as nn
import torch.optim as optim
import torchtext

from source.model.autoencoder import AutoEncoder
from source.model.train_model import train_model
from source.model.eval_model import evaluate


def get_data(data_path):
    return 1, 2, 3


def run_main(data_path,
             epochs):
    train_iterator, test_iterator, val_iterator = get_data(data_path)

    model = AutoEncoder(input_dim=42,
                        embedding_dim=15,
                        decoder_input=1)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        train_loss = train_model(model, train_iterator, optimizer, criterion)

        valid_loss = evaluate(model, val_iterator, criterion)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut1-model.pt')

        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f}')
        print()


if __name__ == '__main__':
    run_main(data_path='../data/tabular_data.csv',
             epochs=5)
