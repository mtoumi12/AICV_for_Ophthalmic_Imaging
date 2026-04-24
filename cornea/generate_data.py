"""Synthetic specular-microscopy data generator.

Generates grayscale images that look like corneal endothelium: roughly
hexagonal cells separated by thin dark boundaries, with plausible noise.

Each sample comes with:
    image      : (H, W) uint8, the microscopy-like image
    mask       : (H, W) uint8, 1 on cell boundaries, 0 elsewhere (ground truth)
    cell_areas : list of cell areas (pixels), excluding border-touching cells
    n_sides    : list of neighbor counts per cell

The synthesis recipe (Voronoi tessellation of jittered hex-grid seeds) is a
standard surrogate for corneal endothelium. It captures the three clinical
morphometry features we care about: cell density, coefficient of variation
of cell area, and percent hexagonal cells.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import Voronoi
from skimage.draw import polygon_perimeter
from skimage.filters import gaussian


def _hex_grid_seeds(h: int, w: int, spacing: float, jitter: float, rng: np.random.Generator) -> np.ndarray:
    """Seeds arranged on a hexagonal grid, jittered to simulate biological variation."""
    seeds = []
    dy = spacing * np.sqrt(3) / 2.0
    row = 0
    y = 0.0
    while y < h + spacing:
        x0 = (spacing / 2.0) if row % 2 == 1 else 0.0
        x = x0
        while x < w + spacing:
            jx = rng.normal(0.0, jitter)
            jy = rng.normal(0.0, jitter)
            seeds.append([x + jx, y + jy])
            x += spacing
        y += dy
        row += 1
    return np.asarray(seeds, dtype=np.float64)


def _rasterize_voronoi_boundaries(
    vor: Voronoi, shape: tuple[int, int]
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return (boundary_mask, list_of_cell_polygons_inside_image)."""
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cells: list[np.ndarray] = []
    for region_idx in vor.point_region:
        verts_idx = vor.regions[region_idx]
        if len(verts_idx) == 0 or -1 in verts_idx:
            continue
        pts = vor.vertices[verts_idx]
        if np.any(pts[:, 0] < 0) or np.any(pts[:, 0] >= w):
            continue
        if np.any(pts[:, 1] < 0) or np.any(pts[:, 1] >= h):
            continue
        rr, cc = polygon_perimeter(pts[:, 1], pts[:, 0], shape=(h, w))
        mask[rr, cc] = 1
        cells.append(pts)
    return mask, cells


def generate_sample(
    size: int = 256,
    mean_cell_spacing: float = 28.0,
    spacing_jitter: float = 4.5,
    seed: int | None = None,
) -> dict:
    """Generate one synthetic specular-microscopy image + ground truth.

    Parameters
    ----------
    size : int
        Image height = width (pixels).
    mean_cell_spacing : float
        Approximate distance between neighboring cell centers, in pixels.
        Larger spacing -> fewer, bigger cells (simulates low ECD).
    spacing_jitter : float
        Random displacement of hex-grid seeds. Larger jitter -> more polymegathism
        (higher CV) and fewer hexagonal cells (lower HEX%).
    seed : int | None
        RNG seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    h = w = size
    seeds = _hex_grid_seeds(h, w, mean_cell_spacing, spacing_jitter, rng)
    vor = Voronoi(seeds)

    boundary_mask, cells = _rasterize_voronoi_boundaries(vor, (h, w))
    boundary_mask = np.clip(gaussian(boundary_mask.astype(np.float32), sigma=0.6) * 2.2, 0, 1)
    boundary_mask = (boundary_mask > 0.4).astype(np.uint8)

    base = 0.82 * np.ones((h, w), dtype=np.float32)
    image = base - 0.55 * boundary_mask.astype(np.float32)

    image += rng.normal(0.0, 0.04, size=image.shape).astype(np.float32)
    image = gaussian(image, sigma=0.7)

    xs, ys = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    vignette = 1.0 - 0.15 * (xs**2 + ys**2)
    image = image * vignette
    image = np.clip(image, 0.0, 1.0)
    image_u8 = (image * 255).astype(np.uint8)

    cell_areas, n_sides = _per_cell_stats(cells, (h, w))
    return {
        "image": image_u8,
        "mask": boundary_mask,
        "cell_areas": np.asarray(cell_areas, dtype=np.float32),
        "n_sides": np.asarray(n_sides, dtype=np.int32),
    }


def _per_cell_stats(cells: list[np.ndarray], shape: tuple[int, int]) -> tuple[list[float], list[int]]:
    """Compute polygon area (shoelace) and #sides, excluding image-border cells."""
    h, w = shape
    areas: list[float] = []
    sides: list[int] = []
    for pts in cells:
        if (pts[:, 0].min() < 2 or pts[:, 0].max() > w - 3 or
                pts[:, 1].min() < 2 or pts[:, 1].max() > h - 3):
            continue
        x = pts[:, 0]
        y = pts[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        areas.append(float(area))
        sides.append(int(len(pts)))
    return areas, sides


def morphometry_from_ground_truth(sample: dict, fov_area_mm2: float = 0.25) -> dict:
    """Reference ECD / CV / HEX% computed from the ground-truth geometry."""
    areas = sample["cell_areas"]
    sides = sample["n_sides"]
    if len(areas) == 0:
        return {"ecd_cells_mm2": 0.0, "cv_area": 0.0, "hex_percent": 0.0, "num_cells": 0}
    ecd = len(areas) / fov_area_mm2
    cv = float(areas.std() / areas.mean()) if areas.mean() > 0 else 0.0
    hex_pct = float((sides == 6).mean() * 100.0)
    return {
        "ecd_cells_mm2": float(ecd),
        "cv_area": cv,
        "hex_percent": hex_pct,
        "num_cells": int(len(areas)),
    }


if __name__ == "__main__":
    sample = generate_sample(seed=0)
    print("image  :", sample["image"].shape, sample["image"].dtype)
    print("mask   :", sample["mask"].shape, "boundary pixels =", int(sample["mask"].sum()))
    print("morph  :", morphometry_from_ground_truth(sample))
