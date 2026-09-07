# Method

This document describes the methodology implemented in `vpr-overlap-training`.

## 1. Visual Place Recognition model

The project addresses **Visual Place Recognition (VPR)**: given a query image, the model retrieves database images that represent the same place.

The VPR network contains:

- a **ResNet50** backbone initialized from ImageNet weights;
- **GeM pooling**;
- a linear projection to a 512-dimensional descriptor;
- L2-normalized global descriptors;
- Euclidean retrieval distance;
- a **Triplet Loss** training objective.

For query descriptor `x_q`, positive descriptor `x_p`, negative descriptor `x_n`, and margin `m`:

```text
L_base = max(0, d(x_q, x_p) - d(x_q, x_n) + m)
```

The MSLS training pipeline uses hard-negative mining and multiple negatives per query.

## 2. Motivation for overlap-aware training

MSLS positives are defined from geographic proximity. Nearby images can nevertheless observe different parts of the environment because of camera orientation, road geometry, scene structure, temporal changes, or occlusions.

The project therefore uses **visual overlap** as an additional confidence signal for geographically positive query-positive pairs.

## 3. VGGT-based geometric overlap

The main overlap estimator uses **VGGT** to obtain the geometry required to compare two views.

For each reference-positive pair:

1. VGGT estimates camera intrinsics, camera extrinsics, and depth.
2. Source-image pixels are unprojected into 3D.
3. The 3D points are transformed into the target camera frame.
4. The points are projected into the target image.
5. A source pixel is valid when the transformed point is in front of the target camera and its projection lies inside the target image.
6. Directional overlap is computed as the fraction of valid source pixels.

The computation is performed in both directions:

```text
o_ref_to_pos = overlap(reference -> positive)
o_pos_to_ref = overlap(positive -> reference)
```

The final overlap is:

```text
o = min(o_ref_to_pos, o_pos_to_ref)
```

Implementation:

```text
src/overlap/compute_vggt_overlap.py
```

## 4. Training configurations

### Baseline

```text
L = L_base
```

Implementation:

```text
src/training/train_baseline.py
```

### Continuous VGGT overlap

```text
L = o * L_base
```

Implementation:

```text
src/training/train_overlap_continuous.py
```

### Binary VGGT overlap

```text
w = 0  if o = 0
w = 1  if o > 0
L = w * L_base
```

Implementation:

```text
src/training/train_overlap_binary.py
```

### 2D field-of-view overlap

The comparison method:

1. uses VGGT camera intrinsics and extrinsics;
2. computes each camera center and forward direction;
3. constructs a horizontal field-of-view sector on the world XZ plane;
4. limits the field of view to 25 m;
5. computes sector intersection and union with Shapely;
6. uses field-of-view IoU as the training overlap value.

```text
o_2d = area(intersection) / area(union)
```

Overlap implementation:

```text
src/overlap/compute_2d_overlap.py
```

Training implementation:

```text
src/training/train_overlap_2d.py
```

## 5. Positive-pair generation

`src/data/generate_msls_positives.py` generates direct query-to-database positives independently for each MSLS city.

The implementation uses:

- individual images (`seq_length = 1`);
- panorama exclusion;
- a 25 m geographic positive threshold;
- local query and database indices for every city.

Each `positives_<split>.npy` contains:

```text
city
mode
qIdx
dbIdx
pIdx
query_paths
database_paths
query_place_id
```

## 6. Validation

Validation is performed independently for each city selected through `VAL_CITIES`.

The default configuration used by the main experiment is:

```text
VAL_CITIES=cph,sf
```

For each selected city:

1. database descriptors are computed;
2. query descriptors are computed;
3. each query is compared only with the database of the same city;
4. Recall@1, Recall@5, and Recall@10 are calculated from the city's local positive indices.

The global validation result aggregates the selected city-level evaluations.

The city list is configurable without editing the Python source. For example:

```bash
VAL_CITIES=cph python src/training/train_overlap_continuous.py
```

This configuration is used by the Copenhagen-only variability experiment.

## 7. Early stopping

The main experiment uses an early-stopping patience of 10 epochs without sufficient Recall@1 improvement.

The patience is configurable through:

```text
EARLY_STOPPING_PATIENCE
```

For example:

```bash
EARLY_STOPPING_PATIENCE=20 python src/training/train_overlap_binary.py
```

This configuration is used by the longer-training variability experiment.

## 8. Qualitative evaluation

`src/evaluation/generate_qualitative_retrievals.py` supports qualitative Top-K inspection, including:

- correct Top-1 retrievals;
- incorrect Top-1 retrievals;
- city-specific cases;
- Top-10 recovery cases;
- city-level evaluation summaries.

The generated qualitative artifacts are not included under `results/`.
