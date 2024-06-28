import torch 
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from . import save
import time
import os
import tempfile
import torch.quantization._numeric_suite as ns
from torch.quantization.quantize_fx import prepare_qat_fx, convert_fx
import copy

def set_seeds(seed: int=42):
    """Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)
    

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
    """
    # Put model in train mode
    model.train()
    # model.to(device)
    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)
  
        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of testing loss, testing accuracy metrics, and inference time.
    In the form (test_loss, test_accuracy, elapsed_time). For example:

    (0.0223, 0.8985, 120.0)
    """
    # Put model in eval mode
    model.to(device)
    model.eval() 

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0
    t0 = time.time()

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))

    elapsed = time.time() - t0

    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader) * 100

    return test_loss, test_acc, elapsed

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          work_dir: str,
          architecture: str,
          device: torch.device,
          qconfig_mapping,
          example_inputs) -> Dict[str, List]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for 
    each epoch.
    In the form: {train_loss: [...],
              train_acc: [...],
              test_loss: [...],
              test_acc: [...]} 
    """
    # Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
    }
    
    # Make sure model on target device
    model.to(device)

    best_metric = float('-inf')  # Initialize the best metric value

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)
        test_loss, test_acc, time_infer = test_step(model=model,
                                                    dataloader=test_dataloader,
                                                    loss_fn=loss_fn,
                                                    device=device)

        # Print out what's happening
        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.2f} | "
          f"time_infer: {time_infer:.4f}"
        )
           # # model.eval()  # Set the model to evaluation mode
    # # model.to('cpu')
    # mp.eval()
    # mp = mp.cpu()
        if test_acc > best_metric:
            best_metric = test_acc
            model_to_quantize = copy.deepcopy(model)
            model_to_quantize.eval()
            model_to_quantize = model.cpu()
            model_prepared = torch.quantization.quantize_fx.prepare_qat_fx(model_to_quantize, qconfig_mapping, example_inputs)
            model_quantized = torch.quantization.quantize_fx.convert_fx(model_prepared)

            # Save the model's state
            save.save_model(model_quantized, work_dir, architecture)
            print(f'Saved best model with validation accuracy: {best_metric:.4f}')
            model.to(device)
        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

def optimal_quant_strategy(model):
    from collections import Counter
    layer_counts = Counter([type(x).__name__ for x in model.modules()])
    print("Model consists of: ", layer_counts)

    dyn = [0, 0]
    stat = [0, 0]

    for m in model.modules():
        if hasattr(m, 'weight'):
            name = type(m).__name__
            params = m.weight.numel()
            if name in ['RNN', 'LSTM', 'GRU', 'LSTMCell', 'RNNCell', 'GRUCell', 'Linear']:
                dyn[0] += 1
                dyn[1] += params
            if 'Conv' in name or name == 'Linear':
                stat[0] += 1
                stat[1] += params
    print()
    print("Dynamic quantization")
    print("====================")
    print(f"Layers: {dyn[0]} || Parameters: {format(dyn[1], 'g')}")
    print()
    print("Static quantization")
    print("====================")
    print(f"Layers: {stat[0]} || Parameters: {format(stat[1], 'g')}")

def print_size_of_model(model):
    model_cpu = model.to('cpu')  # Ensure the model is on the CPU
    with tempfile.NamedTemporaryFile(delete=False) as f:
        temp_path = f.name
    try:
        torch.save(model_cpu.state_dict(), temp_path)
        size_mb = os.path.getsize(temp_path) / 1e6
        print('Size (MB):', size_mb)
    except Exception as e:
        print("Error in saving the model state_dict:", e)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def profile(model, dataloader, loss_fn, device):
    print_size_of_model(model)
    print("="*20)
     # Evaluate PyTorch model
    test_loss, test_acc, pytorch_inference_time = test_step(model, dataloader, loss_fn, device)
    print(f"Test Loss (PyTorch): {test_loss:.4f}, Test Accuracy (PyTorch): {test_acc:.2f}, Inference Time (PyTorch): {pytorch_inference_time:.2f} seconds")

def SNR(x, y):
    # Higher is better
    Ps = torch.norm(x)
    Pn = torch.norm(x-y)
    return 20 * torch.log10(Ps/Pn)

def compare_model_weights(float_model, quant_model):
    snr_dict = {}
    wt_compare_dict = ns.compare_weights(float_model.state_dict(), quant_model.state_dict())
    for param_name, weight in wt_compare_dict.items():
        snr = SNR(weight['float'], weight['quantized'].dequantize())
        snr_dict[param_name] = snr

    return snr_dict

def topk_sensitive_layers(snr_dict, k):
    snr_dict = dict(sorted(snr_dict.items(), key=lambda x:x[1]))
    snr_dict = {k.replace('.weight', ''):v for k,v in list(snr_dict.items())[:k]}
    return snr_dict