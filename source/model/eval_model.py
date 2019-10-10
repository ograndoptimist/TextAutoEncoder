import torch


def evaluate(model,
             iterator,
             criterion):
    epoch_loss = 0

    model.eval()
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.input).squeeze(1)

            loss = criterion(predictions, batch.output.type(torch.LongTensor))

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)