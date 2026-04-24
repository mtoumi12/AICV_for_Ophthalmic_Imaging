"""Extract ECD / CV / HEX% from a predicted boundary mask.

Given a predicted boundary probability map, we:
    1. Threshold to a binary boundary.
    2. Distance-transform the interior (1 - boundary).
    3. Seed from local maxima, then watershed -> per-cell instance labels.
    4. For each cell, compute area, and estimate neighbor count from the
       shared-boundary touching labels.
    5. Drop cells that touch the image border (partial cells corrupt stats).
    6. Aggregate.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.segmentation import watershed


def instances_from_boundary(prob: np.ndarray, thresh: float = 0.5) -> np.ndarray:
    """prob in [0,1] -> instance-label map (0 = background/boundary)."""
    boundary = prob > thresh
    interior = ~boundary
    distance = ndi.distance_transform_edt(interior)
    coords = peak_local_max(distance, min_distance=6, threshold_abs=2.0)
    markers = np.zeros_like(distance, dtype=np.int32)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i
    markers = ndi.label(markers)[0]
    labels = watershed(-distance, markers, mask=interior)
    return labels


def _neighbor_count(labels: np.ndarray, label_id: int) -> int:
    """Number of distinct neighboring labels sharing a boundary with this cell."""
    mask = labels == label_id
    dil = ndi.binary_dilation(mask, iterations=1)
    ring = dil & ~mask
    neigh = np.unique(labels[ring])
    return int(np.sum((neigh > 0) & (neigh != label_id)))


def morphometry_from_mask(prob: np.ndarray, fov_area_mm2: float = 0.25, thresh: float = 0.5) -> dict:
    labels = instances_from_boundary(prob, thresh=thresh)
    h, w = labels.shape
    areas: list[float] = []
    sides: list[int] = []
    for rp in regionprops(labels):
        if rp.area < 25:
            continue
        minr, minc, maxr, maxc = rp.bbox
        if minr <= 1 or minc <= 1 or maxr >= h - 1 or maxc >= w - 1:
            continue
        areas.append(float(rp.area))
        sides.append(_neighbor_count(labels, rp.label))
    if not areas:
        return {"ecd_cells_mm2": 0.0, "cv_area": 0.0, "hex_percent": 0.0, "num_cells": 0}
    a = np.asarray(areas)
    s = np.asarray(sides)
    return {
        "ecd_cells_mm2": float(len(a) / fov_area_mm2),
        "cv_area": float(a.std() / a.mean()),
        "hex_percent": float((s == 6).mean() * 100.0),
        "num_cells": int(len(a)),
    }
