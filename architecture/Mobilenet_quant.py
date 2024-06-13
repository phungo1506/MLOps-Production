import torch
import torch.nn as nn
from torch import Tensor
from torch.quantization import QuantStub, DeQuantStub, fuse_modules

# Model Components
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, act=nn.ReLU, groups=1, bn=True, bias=False):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels) if bn else nn.Identity()
        self.act = act()  # Fixed this line

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
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
            ConvBlock(exp_size, exp_size, kernel_size, stride, act, groups=exp_size),
            SeBlock(exp_size) if se else nn.Identity(),
            ConvBlock(exp_size, out_channels, 1, 1, nn.Identity)
        )
        self.act = act() if act is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.block(x)
        if self.add:
            res += x
        return self.act(res)

class MobileNetV3(nn.Module):
    def __init__(self, config_name: str, in_channels=3, num_classes=4):
        super().__init__()
        config = self.config(config_name)

        self.conv = ConvBlock(in_channels, 16, 3, 2, nn.Hardswish)
        self.blocks = nn.ModuleList([])
        for c in config:
            kernel_size, exp_size, in_channels, out_channels, se, nl, s = c
            self.blocks.append(BNeck(in_channels, out_channels, kernel_size, exp_size, se, nl, s))

        last_outchannel = config[-1][3]
        last_exp = config[-1][1]
        out = 1280 if config_name == "large" else 1024
        self.classifier = nn.Sequential(
            ConvBlock(last_outchannel, last_exp, 1, 1, nn.Hardswish),
            nn.AdaptiveAvgPool2d((1, 1)),
            ConvBlock(last_exp, out, 1, 1, nn.Hardswish, bn=False, bias=True),
            nn.Dropout(0.8),
            nn.Conv2d(out, num_classes, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)

    def config(self, name):
        HE, RE = nn.Hardswish, nn.ReLU
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

def fuse_hardswish(*modules):
    if len(modules) == 3:
        conv, bn, hs = modules
        fused = nn.Sequential(conv, bn, hs)
    elif len(modules) == 2:
        conv, bn = modules
        fused = nn.Sequential(conv, bn)
    else:
        raise ValueError("Unexpected number of modules to fuse")
    return fused


def fuse_model(model):
    fuse_list = []
    for name, m in model.named_modules():
        if isinstance(m, ConvBlock):
            if hasattr(m, 'conv') and hasattr(m, 'bn') and hasattr(m, 'act'):
                fuse_list.append((f'{name}.conv', f'{name}.bn', f'{name}.act'))
        elif isinstance(m, BNeck):
            if hasattr(m.block[0], 'conv') and hasattr(m.block[0], 'bn') and hasattr(m.block[0], 'act'):
                fuse_list.append((f'{name}.block.0.conv', f'{name}.block.0.bn', f'{name}.block.0.act'))
            if hasattr(m.block[1], 'conv') and hasattr(m.block[1], 'bn') and hasattr(m.block[1], 'act'):
                fuse_list.append((f'{name}.block.1.conv', f'{name}.block.1.bn', f'{name}.block.1.act'))
            if isinstance(m.block[2], SeBlock):
                if hasattr(m.block[3], 'conv') and hasattr(m.block[3], 'bn'):
                    fuse_list.append((f'{name}.block.3.conv', f'{name}.block.3.bn'))
            else:
                if hasattr(m.block[2], 'conv') and hasattr(m.block[2], 'bn'):
                    fuse_list.append((f'{name}.block.2.conv', f'{name}.block.2.bn'))

    for fusion in fuse_list:
        modules_to_fuse = [model.get_submodule(name) for name in fusion]
        fused = fuse_hardswish(*modules_to_fuse)
        parent_name, attr_name = '.'.join(fusion[0].split('.')[:-1]), fusion[0].split('.')[-1]
        setattr(model.get_submodule(parent_name), attr_name, fused)
    
    return model