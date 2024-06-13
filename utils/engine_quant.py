import torch
import torch.quantization
from torch import nn
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from . import save

def set_seeds(seed: int=42):
    """Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)

def train_step(model: nn.Module, dataloader: torch.utils.data.DataLoader, 
               loss_fn: nn.Module, optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    model.train()
    train_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model, train_loss / len(dataloader)

def test_step(model: nn.Module, dataloader: torch.utils.data.DataLoader, 
              loss_fn: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            test_pred_logits = model(X)
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            test_acc += ((test_pred_logits.argmax(dim=1) == y).sum().item() / len(y))
    return test_loss / len(dataloader), test_acc / len(dataloader)

def train(model: nn.Module, train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer,
          loss_fn: nn.Module, epochs: int, device: torch.device) -> Dict[str, List]:
    results = {"train_loss": [], "test_loss": [], "test_acc": []}
    for epoch in tqdm(range(epochs)):
        model, train_loss = train_step(model, train_dataloader, loss_fn, optimizer, device)
        test_loss, test_acc = test_step(model, test_dataloader, loss_fn, device)
        print(f"Epoch: {epoch+1} | train_loss: {train_loss:.4f} | test_loss: {test_loss:.4f} | test_acc: {test_acc:.4f}")
        results["train_loss"].append(train_loss)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
    return results
