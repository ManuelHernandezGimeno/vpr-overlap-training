# Method

This document summarizes the methodology implemented in `vpr-overlap-training`.

## 1. Visual Place Recognition model

The project addresses **Visual Place Recognition (VPR)**: given a query image, the model retrieves database images that represent the same place.

The VPR network used in the experiments contains:

- a **ResNet50** backbone initialized from ImageNet weights;
- **GeM pooling** to aggregate convolutional features into a global descriptor;
- L2-normalized global image descriptors;
- Euclidean distance in descriptor space;
- a **Triplet Loss** training objective.

For an anchor/query descriptor `x_a`, a positive descriptor `x_p`, a negative descriptor `x_n`, and margin `m`, the baseline objective is:

```text
L_base = max(0, d(x_a, x_p) - d(x_a, x_n) + m)
```

The training pipeline uses hard-negative mining through the MSLS training dataset.

## 2. Motivation for overlap-aware training

In MSLS, positive images are selected using geographic proximity. Two geographically close images can nevertheless observe different parts of the environment because of camera orientation, road geometry, temporal changes, or scene structure.

The project therefore introduces **visual overlap** as an additional confidence signal for positive pairs.

## 3. VGGT-based geometric overlap

The main proposed overlap estimator uses **VGGT** to infer camera geometry.

For a reference-positive image pair:

1. VGGT estimates camera intrinsics, camera extrinsics, and depth.
2. Pixels from the source image are unprojected to 3D.
3. The 3D points are transformed into the target camera frame.
4. The points are projected into the target image.
5. A source pixel is considered valid when the projected point is in front of the target camera and falls inside the target image bounds.
6. Directional overlap is computed as the fraction of valid source pixels.

The overlap is computed in both directions:

```text
o_ref_to_pos = overlap(reference -> positive)
o_pos_to_ref = overlap(positive -> reference)
```

The final value is conservative:

```text
o = min(o_ref_to_pos, o_pos_to_ref)
```

This prevents a highly asymmetric projection from being interpreted as strong mutual overlap.

## 4. Training configurations

Four configurations are compared.

### 4.1 Baseline

```text
L = L_base
```

### 4.2 Continuous VGGT overlap

```text
L = o * L_base
```

The overlap acts as a confidence weight without changing the internal relation between positive distance, negative distance, and margin.

### 4.3 Binary VGGT overlap

```text
w = 0  if o = 0
w = 1  if o > 0
L = w * L_base
```

This formulation removes only positive pairs for which VGGT estimates no visual overlap.

### 4.4 2D field-of-view overlap

The comparison method in `src/overlap/compute_2d_overlap.py`:

1. uses VGGT camera intrinsics and extrinsics;
2. computes the camera center and forward direction;
3. constructs a horizontal field-of-view sector on the world XZ plane;
4. limits the sector to a maximum distance of 25 m;
5. computes the intersection and union of the two sectors with Shapely;
6. uses field-of-view IoU as the training overlap value:

```text
o_2d = area(intersection) / area(union)
```

## 5. Positive-pair generation

`src/data/generate_msls_positives.py` generates direct query-to-database positives for each MSLS city.

The current implementation uses:

- individual images (`seq_length = 1`);
- panorama exclusion;
- a 25 m geographic positive threshold;
- separate local query and database indices for each city.

The resulting `positives_<split>.npy` files contain fields including:

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

Validation is performed independently for each city.

For each validation city:

1. database descriptors are computed;
2. query descriptors are computed;
3. each query is compared only against the database of the same city;
4. Recall@1, Recall@5, and Recall@10 are calculated using local positive indices.

Global validation metrics are obtained by aggregating city-level results according to the number of evaluated queries.

## 7. Qualitative evaluation

`src/evaluation/generate_qualitative_retrievals.py` generates qualitative Top-K retrieval examples and city-level metric summaries, including correct and incorrect Top-1 cases and Top-10 recovery cases.
