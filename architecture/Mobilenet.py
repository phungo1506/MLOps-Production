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
    def __init__(self, config_name : str, in_channels = 3, num_classes = 4):
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
            nn.Conv2d(out, num_classes, 1, 1)
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

        small_KD = [
                [3, 16, 16, 16, True, RE, 2],
                [3, 72, 16, 48, False, RE, 2], 
                # [3, 88, 24, 24, False, RE, 1], #2
                # [5, 96, 24, 40, True, HE, 2], #2
                # [5, 240, 40, 40, True, HE, 1], #1
                # [5, 240, 40, 40, True, HE, 1], #1
                [5, 120, 48, 96, True, HE, 1],
                # [5, 144, 48, 96, True, HE, 2],
                # [5, 144, 96, 96, True, HE, 1],
                # [5, 120, 40, 48, True, HE, 1],
                # [5, 144, 48, 48, True, HE, 1], #1
                # [5, 288, 48, 96, True, HE, 2],
                # [5, 576, 96, 96, True, HE, 1], #1
                # [5, 576, 96, 96, True, HE, 1]
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
        if name == "small_KD": return small_KD
