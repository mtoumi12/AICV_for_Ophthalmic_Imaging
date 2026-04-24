"""Small U-Net for corneal endothelial cell-boundary segmentation.

Single-input (grayscale), single-output (boundary probability). Kept small on
purpose: on a CPU the training demo in `train.py` finishes in a few minutes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_ch: int = 1, base: int = 16):
        super().__init__()
        self.d1 = ConvBlock(in_ch, base)
        self.d2 = ConvBlock(base, base * 2)
        self.d3 = ConvBlock(base * 2, base * 4)
        self.d4 = ConvBlock(base * 4, base * 8)
        self.bott = ConvBlock(base * 8, base * 8)

        self.up4 = nn.ConvTranspose2d(base * 8, base * 8, 2, stride=2)
        self.u4 = ConvBlock(base * 16, base * 4)
        self.up3 = nn.ConvTranspose2d(base * 4, base * 4, 2, stride=2)
        self.u3 = ConvBlock(base * 8, base * 2)
        self.up2 = nn.ConvTranspose2d(base * 2, base * 2, 2, stride=2)
        self.u2 = ConvBlock(base * 4, base)
        self.up1 = nn.ConvTranspose2d(base, base, 2, stride=2)
        self.u1 = ConvBlock(base * 2, base)

        self.out = nn.Conv2d(base, 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.d1(x)
        d2 = self.d2(self.pool(d1))
        d3 = self.d3(self.pool(d2))
        d4 = self.d4(self.pool(d3))
        b = self.bott(self.pool(d4))

        u4 = self.u4(torch.cat([self.up4(b), d4], dim=1))
        u3 = self.u3(torch.cat([self.up3(u4), d3], dim=1))
        u2 = self.u2(torch.cat([self.up2(u3), d2], dim=1))
        u1 = self.u1(torch.cat([self.up1(u2), d1], dim=1))

        return self.out(u1)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    num = 2.0 * (probs * target).sum(dim=(2, 3))
    den = probs.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps
    return (1.0 - num / den).mean()


def combined_loss(logits: torch.Tensor, target: torch.Tensor, bce_weight: float = 1.0) -> torch.Tensor:
    # Boundary pixels are ~5% of total -> pos_weight for BCE to counter imbalance.
    pos = target.sum()
    neg = target.numel() - pos
    pos_weight = (neg / (pos + 1.0)).clamp(max=20.0).to(logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    return bce_weight * bce + dice_loss(logits, target)
