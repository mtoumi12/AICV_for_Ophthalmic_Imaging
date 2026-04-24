"""Preprocessing for color fundus images.

Standard pipeline used on real fundus photography:
    1. Circular crop — removes dark camera border (prevents the model from
       cheating by memorising per-site border shape).
    2. CLAHE on the green channel — green gives the highest vessel/lesion
       contrast; we apply CLAHE to G and pass G/R/G back as channels.
       (A lot of production fundus CNNs use green-only + 2 CLAHE variants.)
    3. Resize to 224x224.
    4. ImageNet normalization — ResNet-18 was pretrained with these statistics.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch


_CLAHE = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def circular_crop(image_u8: np.ndarray) -> np.ndarray:
    h, w = image_u8.shape[:2]
    cy, cx = h // 2, w // 2
    r = int(min(h, w) * 0.48)
    ys, xs = np.ogrid[:h, :w]
    mask = ((ys - cy) ** 2 + (xs - cx) ** 2) <= r * r
    out = image_u8.copy()
    out[~mask] = 0
    return out


def fundus_preprocess(image_u8: np.ndarray, size: int = 224) -> np.ndarray:
    """(H, W, 3) uint8 -> (3, size, size) float32, ImageNet-normalized."""
    img = circular_crop(image_u8)
    green = img[..., 1]
    green_eq = _CLAHE.apply(green)
    img = img.copy()
    img[..., 1] = green_eq
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    x = img.astype(np.float32) / 255.0
    x = (x - _IMAGENET_MEAN) / _IMAGENET_STD
    x = np.transpose(x, (2, 0, 1))
    return x


class FundusDataset(torch.utils.data.Dataset):
    def __init__(self, samples: list[dict], augment: bool = True, seed: int = 0):
        self.samples = samples
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.samples[idx]
        img = s["image"]
        if self.augment:
            if self.rng.random() < 0.5:
                img = np.flip(img, axis=1).copy()
        x = fundus_preprocess(img)
        return torch.from_numpy(x), torch.tensor(s["label"], dtype=torch.long)
