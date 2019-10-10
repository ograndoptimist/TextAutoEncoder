"""
 Training the neural model.
"""


def train_model(model,
                dataset_iterator,
                criterion,
                optimizer):
    model.train()
    for batch in dataset_iterator:
        predictions = model(batch.input).squeeze(1)

        loss = criterion(predictions, batch.output)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()


