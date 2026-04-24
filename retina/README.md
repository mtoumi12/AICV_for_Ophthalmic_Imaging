# Retina — Fundus Disease Classification

## The clinical problem

The Eye Image Analysis Reading Center's retina arm grades fundus photography
and fluorescein angiography for:

- **Diabetic retinopathy (DR)** — microaneurysms, hemorrhages, exudates
- **Diabetic macular edema (DME)** — retinal thickening + hard exudates near fovea
- **Age-related macular degeneration (AMD)** — drusen (dry), CNV (wet)
- **Fuchs' endothelial dystrophy** (cornea-adjacent; typically imaged with specular/confocal on cornea side)
- **Uveitis, glaucoma, vitreoretinal interface disorders**

In this demo we solve a 4-class classification problem on synthetic color fundus
photographs:

| Class   | Synthetic cue                                                 |
|---------|---------------------------------------------------------------|
| Normal  | no lesions                                                    |
| DR      | small red dots (microaneurysms) + small red blobs (hemorrhages) |
| AMD     | yellowish dots near macula (drusen)                           |
| DME     | bright yellow clusters near macula (hard exudates)            |

A real reading center uses multi-modal data (OCT + fundus + FA + clinical
metadata) and grades on continuous scales. This demo stays at fundus-only,
4-class — enough to show the pipeline end-to-end.

## The CV problem

**Input.** `(3, H, W)` RGB fundus image.
**Output.** Class probabilities over {Normal, DR, AMD, DME}.

## Pipeline

1. **`generate_data.py`** — synthesize fundus-like RGB images:
   - reddish-orange background with a radial gradient
   - bright optic disc (off-center)
   - branching vessel tree (randomized recursion)
   - class-specific lesions at a location consistent with clinical anatomy
     (macula vs periphery)

2. **`preprocess.py`**:
   - circular crop (remove black corners — same as real fundus cameras)
   - CLAHE on the green channel (green has the highest vessel/lesion contrast)
   - resize to 224×224
   - ImageNet normalization (ResNet was pretrained this way)

3. **`model.py`** — ResNet-18 pretrained on ImageNet, final FC replaced with 4
   classes. A randomly-initialized alternative is provided for offline use when
   pretrained weights cannot be downloaded.

4. **`train.py`** — generates balanced train/test set, trains for a few epochs,
   reports accuracy + per-class F1 + confusion matrix.

## Features the model uses

The CNN learns its features from pixels, but at the preprocessing stage we
engineer:

| Feature                | Purpose                                           |
|------------------------|---------------------------------------------------|
| green channel CLAHE    | maximize lesion–background contrast               |
| circular mask          | remove camera-border artefacts that would leak class info |
| ImageNet normalization | align with pretrained ResNet statistics           |

## What "good" looks like

On this synthetic data, a few minutes of training should hit 90%+ accuracy.
On real clinical data you'd expect 75–90% macro-F1 depending on severity
distribution and camera variability.
