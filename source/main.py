import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim

from torchtext import data

from source.model.autoencoder import AutoEncoder
from source.model.train_model import train_model
from source.model.eval_model import evaluate


def split_dataset(data_path):
    dataset = pd.read_csv(data_path)

    X = [data_ for data_ in dataset['query_string']]

    train_X, test_X, train_Y, test_Y = train_test_split(X, X, test_size=0.3, random_state=42)

    test_X, val_X, test_Y, val_Y = train_test_split(test_X, test_Y, test_size=0.45, random_state=42)

    train = pd.concat([pd.Series(train_X), pd.Series(train_Y)], axis=1)
    test = pd.concat([pd.Series(test_X), pd.Series(test_Y)], axis=1)
    val = pd.concat([pd.Series(val_X), pd.Series(val_Y)], axis=1)

    train = train.rename(columns={0: 'input', 1: 'output'})
    test = test.rename(columns={0: 'input', 1: 'output'})
    val = val.rename(columns={0: 'input', 1: 'output'})

    train.to_csv("../data/train.csv", index=False)
    test.to_csv("../data/test.csv", index=False)
    val.to_csv("../data/val.csv", index=False)


def get_data(path):
    TEXT = data.Field()
    LABEL = data.Field(dtype=torch.float)

    device = torch.device('cpu')

    fields = [('input', TEXT), ('output', LABEL)]

    train_data, val_data, test_data = data.TabularDataset.splits(
        path=path,
        train='train.csv',
        validation='val.csv',
        test='test.csv',
        format='csv',
        fields=fields,
        skip_header=True
    )

    TEXT.build_vocab(train_data)
    LABEL.build_vocab(train_data)

    train_iterator, val_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, val_data, test_data),
        batch_size=256,
        device=device,
        sort_key=lambda x: len(x.input),
        sort_within_batch=False)

    return train_iterator, val_iterator, test_iterator, len(TEXT.vocab)


def run_main(data_path,
             epochs):
    print("Splitting initial dataset")
    split_dataset(data_path)

    print("Tokenizing data")
    train_iterator, test_iterator, val_iterator, input_dim = get_data(path='../data')

    print("Creating AutoEncoder")
    model = AutoEncoder(input_dim=input_dim,
                        embedding_dim=15,
                        encoder_output=12)

    optimizer = optim.Adam(model.parameters(),
                           lr=0.00001)

    criterion = nn.MSELoss()

    print("Initializing training and validation")
    best_valid_loss = float('inf')
    for epoch in range(epochs):
        print("\tTraining")
        train_loss = train_model(model, train_iterator, optimizer, criterion)

        print("\tValidation")
        valid_loss = evaluate(model, val_iterator, criterion)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut1-model.pt')

        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f}')
        print()


if __name__ == '__main__':
    run_main(data_path='../data/tabular_data.csv',
             epochs=50)
