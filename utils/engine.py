import torch 
import tqdm.auto import tqdm
from typing import Dict, List, Tuple


def train_step(model: torch.nn.Module,
                dataloader: torch.utils.data.dataloader,
                loss_fn: torch.nn.Module,
                optimizer: torch.optim.optimizer,
                device: torch.device) -> Tuple[float, float]:
    """Trains a Pytoch model for a single epoch.

    Turns a target Pytoch model to training mode and then runs
    through all of the required training steps (forward pass,
    loss calculation, optimizer step).

    Args:
    model: A Pytorch model to be trained.
    dataloader: A Dataloader instance for the model to trained on.
    loss_fn: A Pytorch loss function to minimize.
    optimizer: A Pytorch optimizer to help minimizer the loss funtion.
    device: A target decvice to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy).
    """

    #Put model in train mode
    model.train()

    #Setup train loss and train accuracy values
    train_loss, train_accuracy = 0, 0

    # Loop through data loader data batches
    for batch, (X,y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(decvice), y.to(decvice)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accuamulate accuaracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class==y).sum().item()/len(y_pred)

        return train_loss, train_acc


def test_step(model: torch.nn.Module,
                dataloader: torch.utils.data.Dataloader,
                loss_fn:torch.nn.Module,
                decvice: torch.decvice) -> Tuple[float, float]:
    """ Tests a Pytorch model for a single epoch.

    Turns a target Pytorch model to "eval"mode and then performs
    a forward pass on a testing dataset.

    Args:
    model: A Pytorch model to be tested.
    dataloader: A Dataloader instance for the model to be tested on.
    loss_fn: A Pytorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu")

    Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy).
    """

    # Put mode in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through Dataloader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = x.to(decvice), y.to(decvice)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels==y).sum().item())/len(test_pred_labels)

    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

def train(model: torch.nn.Module,
            train_dataloader: torch.utils.data.Dataloader,
            test_dataloader: torch.utils.data.Dataloader,
            optimizer: torch.optim.Optimizer,
            loss_fn: torch.nn.Module,
            epochs: int,
            decvice: torch.device) -> Dict[str, List]:
    """Trains and test a Pytorch model.

    Passes a target Pytorch models through train_step() and test_step()
    funcions for a number of epochs, training and testing the model in 
    the same epoch loop.

    Calculates, prints and stores evalution metrics throught.

    Args:
    model: A Pytorch model to be trained and tested.
    train_dataloader: A Dataloader instance for the model to be trained on.
    test_dataloader: A Dataloader instance for the model to be tested on.
    optimizer: A Pytorch optimizer to help minimize the loss function.
    loss_fn: A Pytorch loss function to calculate loss both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A dictionary of training and testing loss as well as training and testing 
    accuracy metrics. Each metric has a value in a list for each epoch.
    In the form: {train_loss: [...],
                train_acc: [...],
                test_loss: [...],
                test_acc: [...]}
    """
    results = {train_loss: [],
            train_acc: [],
            test_loss: [],
            test_acc: []}

    # Make sure model on target device
    model.to(decvice)

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                        dataloader=train_dataloader,
                                        loss_fn=loss_fn,
                                        optimizer=optimizer,
                                        decvice=decvice)
        
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        optimizer=optimizer,
                                        decvice=decvice)
        
        # Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} |"
            f"test_acc: {test_acc:.4f}"
        )

        # Update result dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results
