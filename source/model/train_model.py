"""
 Training the neural model.
"""


def train_model(model,
                dataset,
                func_loss,
                optimizer,
                epochs):
    model.train()
    for epoch in epochs:
        for x, y in dataset:
            output = model(x)

            loss = func_loss(output, y)
            loss.backward()

            optimizer.step()
