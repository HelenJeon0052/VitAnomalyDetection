"""
key ideas:
1. 3D UNet is a convolutional encoder-decoder with skip connections
2. map : volume > dense per-voxel prediction (segmentation/logits)
3. not inherently a diffusion model

Run:
    python unet3d.py --help
    python unet3d.py --device cpu --steps 50
    python unet3d.py --device cuda --steps 10000

"""


from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import List, Optional, Tuple

from torch.utils.data import Dataset, DataLoader

import argparse
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------
# Utils
# ---------------------------------------

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # backends.cudnn.deterministic = True
    # backends.cudnn.benchmark = False

def _choose_groupnorm_groups(num_channels: int, max_groups: int) -> int:
    # num_ch % g == 0
    # fallback to 1

    g = min(max_groups, num_channels)
    while g > 1 and (num_channels % g) !=0:
        g -= 1
    return g

# ---------------------------------------
# SyntheticBlobs 3D : Datasets
# ---------------------------------------
class SyntheticBlobs3D(Dataset):
    """
    Input : (1, D, L, W)
    Target : binary mask of random spheres (blobs)
    Training uses supervised loss (BCE, Dice)
    """

    def __init__(
        self,
        n_samples: int = 1024,
        size: Tuple[int, int, int] = (64, 64, 64),
        n_blobs_range: Tuple[int, int] = (1, 4),
        radius_range: Tuple[float, float] = (6.0, 14.0),
        noise_std: float = 0.25,
        seed: int = 123,
    ) ->  None:
        super().__init__()
        self.n_samples = n_samples
        self.size = size
        self.n_blobs_range = n_blobs_range
        self.radius_range = radius_range
        self.noise_std = noise_std
        self.rng = random.Random(seed)

        self.torch_gen = torch.Generator().manual_seed(seed)

        # normalized coordinate grid (D, L, W, 3)
        D, L, W = size
        zs = torch.linspace(-1.0, 1.0, D)
        ys = torch.linspace(-1.0, 1.0, L)
        xs = torch.linspace(-1.0, 1.0, W)
        zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing='ij')
        self.grid = torch.stack([zz, yy, xx], dim=-1) # (D, L, W, 3)

    def __len__(self) -> int:
        return self.n_samples

    def _rand_uniform(self, a: float, b: float) -> float:
        return a + (b - a) * self.rng.random()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        D, L, W = self.size

        # empty mask
        mask = torch.zeros((D, L, W), dtype=torch.float32)

        # add a new spherical blobs
        n_blobs = self.rng.randint(self.n_blobs_range[0], self.n_blobs_range[1])

        for _ in range(n_blobs):
            cz = self._rand_uniform(-.6, .6)
            cy = self._rand_uniform(-.6, .6)
            cx = self._rand_uniform(-.6, .6)
            radius = self._rand_uniform(self.radius_range[0], self.radius_range[1])

            # normalized scale ~ 2 / size_axis
            avg_axis = (D + L + W) / 3.0
            r_norm = radius * (2.0 / avg_axis)

            center = torch.tensor([cz, cy, cx], dtype=torch.float32)
            dist = torch.norm(self.grid - center, dim=-1) # (D, L, W)
            blob = (dist <= r_norm).float()
            mask = torch.maximum(mask, blob)

        # build intensity image correlated with the mask
        # brighter foreground and noise added
        img = .15 * torch.randn((D, L, W), generator=self.torch_gen)
        img = img + 1.0 * mask
        img = img + self.noise_std * torch.randn((D, L, W), generator=self.torch_gen)

        # add channel dim for input/output : (C, D, L, W)
        img = img.unsqueeze(0) # (1, D, L, W)
        mask = mask.unsqueeze(0) # (1, D, L, W)

        return img, mask


class ConvNormAct3d(nn.Module):
    """
    Conv3D > GroupNorm > SiLU
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int =3,
            padding: int =1,
            groups: int = 8,
            dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        g = _choose_groupnorm_groups(out_channels, groups)
        self.norm = nn.GroupNorm(num_groups=g, num_channels=out_channels)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)

        return x

class ResNetBlock3D(nn.Module):
    """
    two ConvNormAct3d blocks
    residual skip with 1x1 conv if channels changes
    """

    def __init__(self, in_channels:int, out_channels:int, dropout: float = 0.0, groups:int = 8) -> None:
        super().__init__()
        self.block1 = ConvNormAct3d(in_channels, out_channels, dropout=dropout, groups=groups)
        self.block2 = ConvNormAct3d(out_channels, out_channels, dropout=dropout, groups=groups)
        self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) ->  torch.Tensor:
        l = self.block1(x)
        l = self.block2(l)

        return l + self.skip(x)

class Downsample3D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.down = nn.Conv3d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)

class Upsample3D(nn.Module):
    """
    Trilinear upsample > ConvNormAct3d
    """
    def __init__(self, in_channels: int, out_channels: int, scale: int = 2, dropout: float = 0.0, groups: int = 8):
        super().__init__()
        self.scale = scale
        self.post = ConvNormAct3d(in_channels, out_channels, dropout=dropout, groups=groups)

    def forward(self, x: torch.Tensor, size: Optional[Tuple[int, int, int]] = None) -> torch.Tensor:
        if size is None:
            x = F.interpolate(x, scale_factor=self.scale, mode='trilinear', align_corners=False)
        else:
            x = F.interpolate(x, size=size, mode='trilinear', align_corners=False)
        x = self.post(x)

        return x

# ---------------------------------------
# UNet3D
# ---------------------------------------
class UNet3D(nn.Module):
    """
    Lightweight 3D Unet with ResNet blocks and GroupNorm
    output: logits (no sigmoid), BCEWithLogitsLoss
    """
    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 32,
        num_levels: int = 4,
        dropout: float = 0.0,
        groups: int = 8,
        out_channels: int = 4,
    ) -> None:
        super().__init__()
        assert num_levels >= 2
        
        channels = [base_channels * (2 ** i) for i in range(num_levels)]
        
        #
        self.in_conv = ResNetBlock3D(in_channels, channels[0], dropout=dropout, groups=groups)
        self.downs = nn.ModuleList()
        self.encs = nn.ModuleList()
        
        for i in range(1, num_levels):
            self.downs.append(Downsample3D(channels[i-1]))
            self.encs.append(ResNetBlock3D(channels[i-1], channels[i], dropout=dropout, groups=groups))
        
        self.bottleneck = ResNetBlock3D(channels[-1], channels[-1], dropout=dropout, groups=groups)
        
        # decoder
        self.ups = nn.ModuleList()
        self.decs = nn.ModuleList()
        for i in reversed(range(1, num_levels)):
            self.ups.append(Upsample3D(channels[i], channels[i-1], dropout=dropout, groups=groups))
            self.decs.append(ResNetBlock3D(channels[i-1] * 2, channels[i-1], dropout=dropout, groups=groups))

        self.out = nn.Conv3d(channels[0], out_channels, kernel_size=1)
    
    def encode(self, x: torch.Tensor):
        skips = []
        l = self.in_conv(x)
        skips.append(l)

        for down, enc in zip(self.downs, self.encs):
            l = down(l)
            l = enc(l)
            skips.append(l)
        
        feat = self.bottleneck(l)
        return feat, skips

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        feat, _ = self.encode(x)
        print(f"forward features output shape: {feat.shape}")

        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        l, skips = self.encode(x)
        skips = list(skips)

        # decode : pop skips from end (the last = the deepest skip)
        skips.pop()
        for up, dec in zip(self.ups, self.decs):
            skip = skips.pop()
            l = up(l, size=skip.shape[-3:])
            l = torch.cat([l, skip], dim=1)
            l = dec(l)
            
        return self.out(l)

    
# ---------------------------------------
# Losses
# ---------------------------------------

def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor, n_dim = 2 , num_classes = 4, eps: float = 1e-6) -> torch.Tensor:
    """
    for train
    logits : [b, 1, D, L, W]
    targets: [b, 1, D, L, W] in {0, 1}
    """


    
    probs = torch.sigmoid(logits)
    probs = probs.flatten(n_dim)
    targets = F.one_hot(targets, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
    targets = targets.flatten(n_dim)
    
    inter = (probs * targets).sum(dim=n_dim)
    denom = probs.sum(dim=n_dim) + targets.sum(dim=n_dim)
    dice = (2.0 * inter + eps) / (denom + eps)

    if probs.ndim == 1:
        dice_mean = 1.0 - dice.mean()
    else:
        dice_mean = 1.0 - dice[:, 1:].mean() 
    
    return dice_mean

def dice_score(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> float:
    """
    for eval
    compute dice score for evaluation
    """
    probs = torch.sigmoid(logits)
    preds = probs.float()
    
    preds = preds.flatten(1)
    targets = targets.flatten(1)
    
    inter = (preds * targets).sum(dim=1)
    denom = preds.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * inter + eps) / (denom + eps)
    
    return float(dice.mean())