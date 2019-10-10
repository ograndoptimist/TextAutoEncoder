"""
 Training the neural model.
"""
import torch


def train_model(model,
                dataset_iterator,
                optimizer,
                criterion):
    epoch_loss = 0

    model.train()
    for batch in dataset_iterator:
        optimizer.zero_grad()

        predictions = model(batch.input)
        predictions = torch.mean(predictions, -1)

        loss = criterion(predictions, batch.output)
        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataset_iterator)
