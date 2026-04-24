"""End-to-end demo: synthesize fundus data -> train ResNet-18 -> evaluate.

Runs on CPU in ~3-5 minutes for the demo sizes. Produces:
    outputs/retina_metrics.txt       per-class F1, confusion matrix
    outputs/retina_demo.png          a panel of test images with predictions
"""

from __future__ import annotations

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from generate_data import CLASS_NAMES, generate_sample
from preprocess import FundusDataset, fundus_preprocess
from model import build_classifier


def build_dataset(n_per_class: int, seed_base: int) -> list[dict]:
    samples = []
    for c in range(len(CLASS_NAMES)):
        for i in range(n_per_class):
            samples.append(generate_sample(size=256, cls=c, seed=seed_base + c * 10_000 + i))
    return samples


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> np.ndarray:
    cm = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def per_class_f1(cm: np.ndarray) -> np.ndarray:
    k = cm.shape[0]
    f1 = np.zeros(k)
    for i in range(k):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        p = tp / (tp + fp + 1e-9)
        r = tp / (tp + fn + 1e-9)
        f1[i] = 2 * p * r / (p + r + 1e-9)
    return f1


def main(
    n_train_per_class: int = 60,
    n_test_per_class: int = 20,
    epochs: int = 8,
    batch_size: int = 16,
    lr: float = 1e-3,
    device: str = "cpu",
    pretrained: bool = True,
) -> None:
    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)

    print("Generating synthetic fundus data...")
    train_samples = build_dataset(n_train_per_class, seed_base=1000)
    test_samples = build_dataset(n_test_per_class, seed_base=9000)

    train_ds = FundusDataset(train_samples, augment=True, seed=0)
    test_ds = FundusDataset(test_samples, augment=False, seed=1)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Building ResNet-18 (pretrained={pretrained})...")
    model = build_classifier(num_classes=len(CLASS_NAMES), pretrained=pretrained).to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Training on {len(train_ds)} images for {epochs} epochs...")
    for ep in range(epochs):
        model.train()
        running = 0.0
        n = 0
        for x, y in tqdm(train_loader, desc=f"epoch {ep+1}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
            n += x.size(0)
        print(f"  epoch {ep+1}  train_loss={running/n:.4f}")

    print("Evaluating...")
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()
            y_true.extend(y.numpy().tolist())
            y_pred.extend(preds.tolist())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = (y_true == y_pred).mean()
    cm = confusion_matrix(y_true, y_pred, k=len(CLASS_NAMES))
    f1 = per_class_f1(cm)

    lines = [f"Accuracy: {acc*100:.2f}%"]
    for i, name in enumerate(CLASS_NAMES):
        lines.append(f"  F1[{name:>6}] = {f1[i]:.3f}")
    lines.append("")
    lines.append("Confusion matrix (rows = true, cols = predicted)")
    header = "          " + "  ".join(f"{n:>6}" for n in CLASS_NAMES)
    lines.append(header)
    for i, name in enumerate(CLASS_NAMES):
        row = "  ".join(f"{v:>6d}" for v in cm[i])
        lines.append(f"{name:>8}  {row}")

    summary = "\n".join(lines)
    print("\n" + summary)
    with open(os.path.join(ROOT, "outputs", "retina_metrics.txt"), "w") as f:
        f.write(summary)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        rng = np.random.default_rng(0)
        idx = rng.choice(len(test_samples), size=8, replace=False)
        for k, i in enumerate(idx):
            s = test_samples[i]
            x = fundus_preprocess(s["image"])
            with torch.no_grad():
                p = model(torch.from_numpy(x).unsqueeze(0).to(device))
                pc = int(p.argmax(dim=1).item())
            ax = axes[k // 4, k % 4]
            ax.imshow(s["image"])
            ax.set_title(f"T:{CLASS_NAMES[s['label']]}  P:{CLASS_NAMES[pc]}",
                         color=("green" if pc == s['label'] else "red"))
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(ROOT, "outputs", "retina_demo.png"), dpi=110)
        print("Saved outputs/retina_demo.png")
    except Exception as e:
        print(f"(skipping figure: {e})")


if __name__ == "__main__":
    pretrained = os.environ.get("USE_PRETRAINED", "1") != "0"
    main(pretrained=pretrained)
