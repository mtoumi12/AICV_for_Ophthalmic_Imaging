"""Preprocessing for corneal specular-microscopy images.

Pipeline:
    1. CLAHE (contrast-limited adaptive histogram equalization) on the raw uint8
       image — this mirrors what commercial cell-analysis software does as a
       front-end step, boosting boundary contrast without amplifying global bias.
    2. Zero-mean / unit-std normalization.
    3. Optional random flips & 90-degree rotations for training-time augmentation.
       (We avoid arbitrary-angle rotation so cell boundaries stay sharp.)
"""

from __future__ import annotations

import cv2
import numpy as np
import torch


_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def clahe_normalize(image_u8: np.ndarray) -> np.ndarray:
    """uint8 -> float32 in roughly standard-normal range."""
    eq = _CLAHE.apply(image_u8)
    x = eq.astype(np.float32) / 255.0
    x = (x - x.mean()) / (x.std() + 1e-6)
    return x


def augment(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if rng.random() < 0.5:
        image = np.flip(image, axis=0).copy()
        mask = np.flip(mask, axis=0).copy()
    if rng.random() < 0.5:
        image = np.flip(image, axis=1).copy()
        mask = np.flip(mask, axis=1).copy()
    k = int(rng.integers(0, 4))
    if k:
        image = np.rot90(image, k=k).copy()
        mask = np.rot90(mask, k=k).copy()
    return image, mask


def to_tensors(image: np.ndarray, mask: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    img_t = torch.from_numpy(image).unsqueeze(0).float()
    mask_t = torch.from_numpy(mask).unsqueeze(0).float()
    return img_t, mask_t


class CorneaDataset(torch.utils.data.Dataset):
    """In-memory dataset over a list of generated samples."""

    def __init__(self, samples: list[dict], augment_on: bool = True, seed: int = 0):
        self.samples = samples
        self.augment_on = augment_on
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.samples[idx]
        img = clahe_normalize(s["image"])
        mask = s["mask"].astype(np.float32)
        if self.augment_on:
            img, mask = augment(img, mask, self.rng)
        return to_tensors(img, mask)
