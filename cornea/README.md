# Cornea — Endothelial Cell Morphometry

## The clinical problem

The corneal endothelium is a single layer of hexagonal cells on the inner
surface of the cornea. These cells do not divide. Their job is to pump fluid
out of the cornea; if the cell count drops below ~500 cells/mm² the cornea
swells and vision is lost.

Reading centers quantify endothelial health with three numbers:

| Feature | Meaning                                         | Typical range   |
|---------|-------------------------------------------------|-----------------|
| **ECD** | Endothelial cell density (cells/mm²)            | 2000–3500 healthy; < 500 is decompensation risk |
| **CV**  | Coefficient of variation of cell **area** (SD/mean) | < 0.30 healthy; > 0.40 stressed |
| **HEX%**| Percentage of cells with exactly 6 neighbors    | > 60% healthy; < 50% stressed |

These are computed from a specular microscopy image after segmenting every
individual cell in the field of view.

## The CV problem

**Input.** `(H, W)` grayscale specular microscopy image; cells look like a
honeycomb of roughly hexagonal shapes separated by thin dark boundaries.

**Output.** A per-pixel binary **boundary mask** + per-image scalars
`ECD`, `CV`, `HEX%`.

**Approach in this module.**

1. **`generate_data.py`** — synthesize 2D Voronoi tessellations with ~40 to 80
   cells per 256×256 image. Simulates the appearance of specular microscopy
   (dark cell boundaries on a light background) + adds noise and slight blur.
   Each image comes with a ground-truth boundary mask and the true
   per-image ECD/CV/HEX%.

2. **`preprocess.py`** — CLAHE, zero-mean/unit-std normalization, on-the-fly
   flips/rotations. Returns `(image_tensor, mask_tensor)` pairs.

3. **`model.py`** — small U-Net (4 down / 4 up, 16→128 base channels).
   Loss: `BCE + Dice` on the boundary mask. Boundary pixels are ~5% of total,
   so Dice is required to counter the imbalance.

4. **`morphometry.py`** — given a predicted mask:
   - `1 - mask` → distance transform → local peaks as seeds → watershed
   - `regionprops` for area, perimeter, neighbor count per cell
   - Discard cells that touch the image border
   - Aggregate to `ECD`, `CV`, `HEX%`

5. **`train.py`** — generates 200 train + 50 test images, trains U-Net for
   20 epochs, and on the test set reports:
   - Dice on the boundary mask
   - Mean absolute error for `ECD`, `CV`, `HEX%` vs ground truth
   - Saves a qualitative overlay to `outputs/cornea_demo.png`

## Features the model exposes to the reading center

| Feature         | Source                                              |
|-----------------|-----------------------------------------------------|
| `ecd_cells_mm2` | `N_valid_cells / FOV_area_mm2`                      |
| `cv_area`       | `std(areas) / mean(areas)`                          |
| `hex_percent`   | `100 * mean(neighbor_count == 6)`                   |
| `mean_area_um2` | `mean(areas) * pixel_to_um^2`                       |
| `num_cells`     | `N_valid_cells`                                     |
| `confidence`    | fraction of pixels in high-confidence mask range    |

`FOV_area_mm2` is a scope calibration constant — the real reading center knows
this per-device. Here we set it to an illustrative 0.25 mm².
