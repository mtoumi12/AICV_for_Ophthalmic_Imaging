"""ResNet-18 classifier for 4-class fundus disease classification.

Prefers ImageNet-pretrained weights. Falls back to a fresh (randomly
initialized) ResNet-18 if pretrained weights cannot be downloaded in the
sandbox. On the synthetic data, even a fresh ResNet trained for 3 epochs
gets high accuracy because the class cues are obvious.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


def build_classifier(num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    """ResNet-18 with a new classification head."""
    try:
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            net = models.resnet18(weights=weights)
        else:
            net = models.resnet18(weights=None)
    except Exception:
        net = models.resnet18(weights=None)
    in_f = net.fc.in_features
    net.fc = nn.Linear(in_f, num_classes)
    return net
