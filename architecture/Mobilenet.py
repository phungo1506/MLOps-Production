import torch
import torch.nn as nn
import torch.optim as optim
import torch.quantization
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from pathlib import Path
from torch import Tensor
import os
import torch
import torch.quantization
from torch import nn
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

# Model Components
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, act=nn.ReLU, groups=1, bn=True, bias=False):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels) if bn else nn.Identity()
        self.act = act

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x

class SeBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        C = in_channels
        r = C // 4
        self.globpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(C, r, bias=False)
        self.fc2 = nn.Linear(r, C, bias=False)
        self.relu = nn.ReLU()
        self.hsigmoid = nn.Hardsigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.globpool(x)
        f = torch.flatten(f, 1)
        f = self.relu(self.fc1(f))
        f = self.hsigmoid(self.fc2(f))
        f = f[:, :, None, None]
        scale = x * f
        return scale

class BNeck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, exp_size: int, se: bool, act: nn.Module, stride: int):
        super().__init__()
        self.add = in_channels == out_channels and stride == 1
        self.block = nn.Sequential(
            ConvBlock(in_channels, exp_size, 1, 1, act),
            ConvBlock(exp_size, exp_size, kernel_size, stride, act, exp_size),
            SeBlock(exp_size) if se else nn.Identity(),
            ConvBlock(exp_size, out_channels, 1, 1, act=nn.Identity())
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.block(x)
        if self.add:
            res += x
        return res

class MobileNetV3(nn.Module):
    def __init__(self, config_name : str, in_channels = 3, classes = 4):
        super().__init__()
        config = self.config(config_name)

        # First convolution(conv2d) layer.
        self.conv = ConvBlock(in_channels, 16, 3, 2, nn.Hardswish())
        # Bneck blocks in a list.
        self.blocks = nn.ModuleList([])
        for c in config:
            kernel_size, exp_size, in_channels, out_channels, se, nl, s = c
            self.blocks.append(BNeck(in_channels, out_channels, kernel_size, exp_size, se, nl, s))

        # Classifier
        last_outchannel = config[-1][3]
        last_exp = config[-1][1]
        out = 1280 if config_name == "large" else 1024
        self.classifier = nn.Sequential(
            ConvBlock(last_outchannel, last_exp, 1, 1, nn.Hardswish()),
            nn.AdaptiveAvgPool2d((1,1)),
            ConvBlock(last_exp, out, 1, 1, nn.Hardswish(), bn=False, bias=True),
            nn.Dropout(0.8),
            nn.Conv2d(out, classes, 1, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        for block in self.blocks:
            x = block(x)

        x = self.classifier(x)
        return torch.flatten(x, 1)


    def config(self, name):
        HE, RE = nn.Hardswish(), nn.ReLU()
        # [kernel, exp size, in_channels, out_channels, SEBlock(SE), activation function(NL), stride(s)]
        large = [
                [3, 16, 16, 16, False, RE, 1],
                [3, 64, 16, 24, False, RE, 2],
                [3, 72, 24, 24, False, RE, 1],
                [5, 72, 24, 40, True, RE, 2],
                [5, 120, 40, 40, True, RE, 1],
                [5, 120, 40, 40, True, RE, 1],
                [3, 240, 40, 80, False, HE, 2],
                [3, 200, 80, 80, False, HE, 1],
                [3, 184, 80, 80, False, HE, 1],
                [3, 184, 80, 80, False, HE, 1],
                [3, 480, 80, 112, True, HE, 1],
                [3, 672, 112, 112, True, HE, 1],
                [5, 672, 112, 160, True, HE, 2],
                [5, 960, 160, 160, True, HE, 1],
                [5, 960, 160, 160, True, HE, 1]
        ]

        small = [
                [3, 16, 16, 16, True, RE, 2],
                [3, 72, 16, 24, False, RE, 2],
                [3, 88, 24, 24, False, RE, 1],
                [5, 96, 24, 40, True, HE, 2],
                [5, 240, 40, 40, True, HE, 1],
                [5, 240, 40, 40, True, HE, 1],
                [5, 120, 40, 48, True, HE, 1],
                [5, 144, 48, 48, True, HE, 1],
                [5, 288, 48, 96, True, HE, 2],
                [5, 576, 96, 96, True, HE, 1],
                [5, 576, 96, 96, True, HE, 1]
        ]

        if name == "large": return large
        if name == "small": return small

# Dataloader creation
NUM_WORKERS = os.cpu_count()

def create_dataloaders(train_dir: str,
                    test_dir: str,
                    transform: transforms.Compose,
                    batch_size: int,
                    num_workers: int=NUM_WORKERS):
    """Creates training and testing DataLoaders.

    Takes in a training directory and testing directory path and turns
    them into Pyotrch Datasets and then into Pytorch DataLoaders.

    Args:
        train_dir: Path to training directory.
        test_dir: Path to testing directory.
        transform: torchvision transforms to perform on training and testing data.
        batch_size: Number of samples per batch in each of the DataLoaders.
        num_workers: An integer for number of workers per DataLoader.

    Returns:
    A tuple of (train_dataloader, test_dataloader, class_name).
    Where class_name is a list of the target classes.
    Example usage:
    train_dataloader, test_dataloader, class_names = create_dataloaders(
                                                        train_dir=path/to/train_dir,
                                                        test_dir=path/to/test_dir,
                                                        transform=some_transform,
                                                        batch_size=32,
                                                        num_workers=4) 
  """
    # Use ImageFolder to create dataset(s)
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    # Get class names
    class_names = train_data.classes

    # Turn images into data loaders
    train_dataloader = DataLoader(train_data,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers,
                                pin_memory=True)
                                
    test_dataloader = DataLoader(test_data,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                pin_memory=True)

    return train_dataloader, test_dataloader, class_names


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
          loss_fn: nn.Module, epochs: int, work_dir: str, architecture: str,
          device: torch.device) -> Dict[str, List]:
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    model.to(device)
    
    # Prepare model for Quantization-Aware Training
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    q_model = torch.quantization.prepare_qat(model, inplace=False)

    for epoch in tqdm(range(epochs)):
        q_model, train_loss = train_step(q_model, train_dataloader, loss_fn, optimizer, device)
        test_loss, test_acc = test_step(q_model, test_dataloader, loss_fn, device)
        
        print(f"Epoch: {epoch+1} | train_loss: {train_loss:.4f} | test_loss: {test_loss:.4f} | test_acc: {test_acc:.4f}")
        
        # Convert model to quantized version after training
        quantized_model = torch.quantization.convert(q_model.eval(), inplace=False)
        
        # Save the quantized model
        save_model(quantized_model, work_dir, architecture)
        
        results["train_loss"].append(train_loss)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
    
    return results

"""
Contains various utility functions for PyTorch model training and saving.
"""
import torch
from pathlib import Path

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

    Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                        exist_ok=True)

    # # Create model save path
    model_save_path = str(target_dir_path) + '/' + model_name + '_best.pth'
    assert model_save_path.endswith(".pth") or model_save_path.endswith(".pt"), "model_name should end with '.pt' or '.pth'"

    # Save the model state_dict()
    torch.save(model.state_dict(), model_save_path)
    
# Main Function
def main():
    parser = ArgumentParser(description='Train classification')
    parser.add_argument('--work-dir', default='models', help='the dir to save logs and models')
    parser.add_argument("--train-folder", default='data/train', type=str)
    parser.add_argument("--valid-folder", default='data/test', type=str)
    parser.add_argument('--architecture', default='MobileNetv3', help='MobileNetv3, ResNet', type=str)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--img-size", default=112, type=int)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument('--lr', default=0.01, type=float)

    args = parser.parse_args()

    devices = "cuda" if torch.cuda.is_available() else "cpu"

    manual_transforms = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.46295794, 0.46194877, 0.4847407), (0.19444681, 0.19439201, 0.19383532))
    ])

    train_dataloader, test_dataloader, class_names = create_dataloaders(args.train_folder, args.valid_folder, manual_transforms, args.batch_size)

    model = MobileNetV3('small')
    # Fuse operations for better quantization
    model.apply(torch.quantization.fuse_modules, [['conv', 'bn', 'act']])

    # Prepare model for Quantization-Aware Training
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    q_model = torch.quantization.prepare_qat(model, inplace=False)

    q_model.to(devices)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(q_model.parameters(), lr=args.lr, momentum=0.9)

    train(q_model, train_dataloader, test_dataloader, optimizer, loss_fn, args.epochs, devices)

    # Convert the model to quantized form after training
    quantized_model = torch.quantization.convert(q_model.eval(), inplace=False)

    # Save the quantized model
    save_model(quantized_model, args.work_dir, model_name=f"{args.architecture}_quantized.pth")

if __name__ == "__main__":
    main()