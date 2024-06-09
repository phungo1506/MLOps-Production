import torch
import torch.quantization
from torch import nn
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from . import save

def set_seeds(seed: int=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def train_step(model: nn.Module, dataloader: torch.utils.data.DataLoader, 
               loss_fn: nn.Module, optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    model.train()
    train_loss, train_acc = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        inputs = model.quant(X)
        outputs = model(inputs)
        y_pred = model.dequant(outputs)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return model, train_loss, train_acc

def test_step(model: nn.Module, dataloader: torch.utils.data.DataLoader, 
              loss_fn: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            test_pred_logits = model(X)
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))
    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
    return test_loss, test_acc

def train(model: nn.Module, train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer,
          loss_fn: nn.Module, epochs: int, work_dir: str, architecture: str,
          device: torch.device) -> Dict[str, List]:
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    model.to(device)
    best_metric = float('-inf')

    # Prepare model for quantization
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)

    for epoch in tqdm(range(epochs)):
        model, train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, device)
        test_loss, test_acc = test_step(model, test_dataloader, loss_fn, device)
        print(f"Epoch: {epoch+1} | train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | test_loss: {test_loss:.4f} | test_acc: {test_acc:.4f}")
        if test_acc > best_metric:
            best_metric = test_acc
            # Convert model to quantized version
            torch.quantization.convert(model, inplace=True)
            print(f'Check statistics of the various layers')
            print(model)
            # Save model using your save function
            save.save_model(model, work_dir, architecture)
            print(f'Saved best model with validation accuracy: {best_metric:.4f}')
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
    return results

# Example usage:
# Note: Replace `save.save_model` with your own function or method to save the model as needed.
