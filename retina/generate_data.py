"""Synthetic color-fundus generator for 4-class classification.

Generates RGB images that look reminiscent of a fundus photograph: orange-red
background, bright optic disc, branching vessels, and class-specific lesions.
It is deliberately simple — the goal is to demonstrate the full CV pipeline
(preprocessing, classifier, metrics), not to pass clinical validation.

Classes:
    0 = Normal     (no lesions)
    1 = DR         (microaneurysms + small hemorrhages, scattered)
    2 = AMD        (drusen near macula — yellowish small dots)
    3 = DME        (hard exudates near macula — bright yellow clusters)
"""

from __future__ import annotations

import numpy as np

CLASS_NAMES = ["Normal", "DR", "AMD", "DME"]


def _base_fundus(size: int, rng: np.random.Generator) -> np.ndarray:
    """Orange-red background with radial gradient, inside a circular FOV."""
    h = w = size
    cy, cx = h // 2, w // 2
    ys, xs = np.ogrid[:h, :w]
    r = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    radius = size * 0.48
    inside = r <= radius

    base = np.zeros((h, w, 3), dtype=np.float32)
    base[..., 0] = 0.70 + 0.12 * rng.normal(0, 1)
    base[..., 1] = 0.25 + 0.05 * rng.normal(0, 1)
    base[..., 2] = 0.12 + 0.04 * rng.normal(0, 1)

    falloff = np.clip(1.0 - (r / radius) ** 2, 0.0, 1.0)
    base *= (0.55 + 0.45 * falloff[..., None])
    base[~inside] = 0.0
    return base


def _optic_disc(img: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
    h, w, _ = img.shape
    angle = rng.uniform(-0.4, 0.4)
    dx = int(0.28 * w * np.cos(angle))
    dy = int(0.10 * h * np.sin(angle))
    cy, cx = h // 2 + dy, w // 2 + dx
    disc_r = int(0.07 * h)
    ys, xs = np.ogrid[:h, :w]
    disc_mask = ((ys - cy) ** 2 + (xs - cx) ** 2) <= disc_r ** 2
    img[disc_mask, 0] = 0.95
    img[disc_mask, 1] = 0.92
    img[disc_mask, 2] = 0.78
    return cy, cx


def _draw_vessel(img: np.ndarray, y: float, x: float, dy: float, dx: float,
                 length: float, width: float, rng: np.random.Generator, depth: int) -> None:
    h, w, _ = img.shape
    if depth > 5 or length < 6 or width < 0.5:
        return
    steps = int(length)
    for _ in range(steps):
        dy += rng.normal(0, 0.10)
        dx += rng.normal(0, 0.10)
        n = np.sqrt(dy * dy + dx * dx) + 1e-6
        dy /= n
        dx /= n
        y += dy
        x += dx
        if not (0 <= y < h and 0 <= x < w):
            return
        r = max(1, int(round(width)))
        yy, xx = np.ogrid[max(0, int(y) - r):min(h, int(y) + r + 1),
                          max(0, int(x) - r):min(w, int(x) + r + 1)]
        mask = (yy - y) ** 2 + (xx - x) ** 2 <= r * r
        patch = img[max(0, int(y) - r):min(h, int(y) + r + 1),
                    max(0, int(x) - r):min(w, int(x) + r + 1)]
        patch[mask] = np.array([0.32, 0.03, 0.03], dtype=np.float32)

    if rng.random() < 0.6:
        a = rng.uniform(0.5, 1.1)
        _draw_vessel(img, y, x, dy * np.cos(a) - dx * np.sin(a),
                     dy * np.sin(a) + dx * np.cos(a),
                     length * 0.55, width * 0.8, rng, depth + 1)
    if rng.random() < 0.6:
        a = rng.uniform(0.5, 1.1)
        _draw_vessel(img, y, x, dy * np.cos(-a) - dx * np.sin(-a),
                     dy * np.sin(-a) + dx * np.cos(-a),
                     length * 0.55, width * 0.8, rng, depth + 1)


def _vessels(img: np.ndarray, disc_yx: tuple[int, int], rng: np.random.Generator) -> None:
    cy, cx = disc_yx
    h, w, _ = img.shape
    for _ in range(6):
        angle = rng.uniform(0, 2 * np.pi)
        _draw_vessel(img, cy, cx, np.sin(angle), np.cos(angle),
                     length=rng.uniform(40, 80), width=rng.uniform(1.3, 2.2),
                     rng=rng, depth=0)


def _add_blob(img: np.ndarray, y: int, x: int, r: int, color: np.ndarray) -> None:
    h, w, _ = img.shape
    yy, xx = np.ogrid[max(0, y - r):min(h, y + r + 1), max(0, x - r):min(w, x + r + 1)]
    mask = (yy - y) ** 2 + (xx - x) ** 2 <= r * r
    img[max(0, y - r):min(h, y + r + 1), max(0, x - r):min(w, x + r + 1)][mask] = color


def _lesions(img: np.ndarray, cls: int, rng: np.random.Generator) -> None:
    h, w, _ = img.shape
    macula_y, macula_x = h // 2, w // 2
    if cls == 0:
        return
    if cls == 1:
        for _ in range(rng.integers(15, 30)):
            y = int(rng.integers(int(0.15 * h), int(0.85 * h)))
            x = int(rng.integers(int(0.15 * w), int(0.85 * w)))
            r = int(rng.integers(1, 3))
            _add_blob(img, y, x, r, np.array([0.55, 0.05, 0.05], dtype=np.float32))
        for _ in range(rng.integers(3, 8)):
            y = int(rng.integers(int(0.2 * h), int(0.8 * h)))
            x = int(rng.integers(int(0.2 * w), int(0.8 * w)))
            r = int(rng.integers(3, 6))
            _add_blob(img, y, x, r, np.array([0.45, 0.04, 0.04], dtype=np.float32))
    elif cls == 2:
        for _ in range(rng.integers(25, 45)):
            y = macula_y + int(rng.normal(0, 0.10 * h))
            x = macula_x + int(rng.normal(0, 0.10 * w))
            r = int(rng.integers(1, 3))
            _add_blob(img, y, x, r, np.array([0.92, 0.86, 0.60], dtype=np.float32))
    elif cls == 3:
        for _ in range(rng.integers(5, 10)):
            y = macula_y + int(rng.normal(0, 0.04 * h))
            x = macula_x + int(rng.normal(0, 0.04 * w))
            r = int(rng.integers(4, 8))
            _add_blob(img, y, x, r, np.array([1.00, 0.90, 0.10], dtype=np.float32))


def generate_sample(size: int = 256, cls: int = 0, seed: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    img = _base_fundus(size, rng)
    disc_yx = _optic_disc(img, rng)
    _vessels(img, disc_yx, rng)
    _lesions(img, cls, rng)
    img += rng.normal(0, 0.02, img.shape).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)
    image_u8 = (img * 255).astype(np.uint8)
    return {"image": image_u8, "label": int(cls), "class_name": CLASS_NAMES[cls]}


if __name__ == "__main__":
    for c in range(4):
        s = generate_sample(cls=c, seed=c)
        print(f"class={s['class_name']:>6}  image={s['image'].shape}")
