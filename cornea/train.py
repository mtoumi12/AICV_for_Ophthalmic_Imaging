"""End-to-end demo: synthesize data -> train U-Net -> evaluate morphometry.

Runs on CPU in a few minutes. Produces:
    outputs/cornea_demo.png      -- qualitative panel
    outputs/cornea_metrics.txt   -- Dice + MAE on ECD / CV / HEX%
"""

from __future__ import annotations

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from generate_data import generate_sample, morphometry_from_ground_truth
from preprocess import CorneaDataset, clahe_normalize
from model import UNet, combined_loss
from morphometry import morphometry_from_mask


def build_dataset(n: int, seed_base: int, vary: bool = True) -> list[dict]:
    samples = []
    rng = np.random.default_rng(seed_base)
    for i in range(n):
        if vary:
            spacing = float(rng.uniform(22.0, 34.0))
            jitter = float(rng.uniform(2.5, 6.0))
        else:
            spacing, jitter = 28.0, 4.5
        samples.append(generate_sample(size=256, mean_cell_spacing=spacing,
                                       spacing_jitter=jitter, seed=seed_base + i))
    return samples


def dice_score(logits: torch.Tensor, target: torch.Tensor) -> float:
    probs = (torch.sigmoid(logits) > 0.5).float()
    num = 2.0 * (probs * target).sum().item()
    den = probs.sum().item() + target.sum().item() + 1e-6
    return num / den


def main(
    n_train: int = 200,
    n_test: int = 40,
    epochs: int = 15,
    batch_size: int = 8,
    lr: float = 1e-3,
    device: str = "cpu",
) -> None:
    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)

    print("Generating synthetic data...")
    train_samples = build_dataset(n_train, seed_base=1000)
    test_samples = build_dataset(n_test, seed_base=9000)

    train_ds = CorneaDataset(train_samples, augment_on=True, seed=0)
    test_ds = CorneaDataset(test_samples, augment_on=False, seed=1)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = UNet(in_ch=1, base=16).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Training U-Net on {n_train} images for {epochs} epochs...")
    for ep in range(epochs):
        model.train()
        running = 0.0
        for img, mask in tqdm(train_loader, desc=f"epoch {ep+1}/{epochs}", leave=False):
            img, mask = img.to(device), mask.to(device)
            logits = model(img)
            loss = combined_loss(logits, mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * img.size(0)
        print(f"  epoch {ep+1:02d}  train_loss={running / len(train_ds):.4f}")

    print("Evaluating...")
    model.eval()
    dices = []
    mae_ecd, mae_cv, mae_hex = [], [], []
    for img, mask in test_loader:
        img, mask = img.to(device), mask.to(device)
        with torch.no_grad():
            logits = model(img)
        dices.append(dice_score(logits, mask))

    for s in test_samples:
        gt = morphometry_from_ground_truth(s)
        x = clahe_normalize(s["image"])
        x_t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).float().to(device)
        with torch.no_grad():
            prob = torch.sigmoid(model(x_t)).squeeze().cpu().numpy()
        pred = morphometry_from_mask(prob)
        mae_ecd.append(abs(pred["ecd_cells_mm2"] - gt["ecd_cells_mm2"]))
        mae_cv.append(abs(pred["cv_area"] - gt["cv_area"]))
        mae_hex.append(abs(pred["hex_percent"] - gt["hex_percent"]))

    summary = (
        f"Dice (boundary): {np.mean(dices):.3f}\n"
        f"MAE ECD (cells/mm2): {np.mean(mae_ecd):.1f}\n"
        f"MAE CV (ratio):      {np.mean(mae_cv):.3f}\n"
        f"MAE HEX%:            {np.mean(mae_hex):.2f}\n"
    )
    print("\n" + summary)
    with open(os.path.join(ROOT, "outputs", "cornea_metrics.txt"), "w") as f:
        f.write(summary)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 3, figsize=(9, 9))
        for col in range(3):
            s = test_samples[col]
            x = clahe_normalize(s["image"])
            x_t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).float().to(device)
            with torch.no_grad():
                prob = torch.sigmoid(model(x_t)).squeeze().cpu().numpy()
            axes[0, col].imshow(s["image"], cmap="gray")
            axes[0, col].set_title("input (synthetic)")
            axes[0, col].axis("off")
            axes[1, col].imshow(s["mask"], cmap="gray")
            axes[1, col].set_title("ground truth boundary")
            axes[1, col].axis("off")
            axes[2, col].imshow(prob, cmap="magma")
            axes[2, col].set_title("predicted boundary prob")
            axes[2, col].axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(ROOT, "outputs", "cornea_demo.png"), dpi=110)
        print("Saved outputs/cornea_demo.png")
    except Exception as e:
        print(f"(skipping figure: {e})")


if __name__ == "__main__":
    main()
