# Data

## 1. Mapillary Street-Level Sequences

The project uses **Mapillary Street-Level Sequences (MSLS)**.

The dataset is not distributed with this repository. It must be obtained separately and used according to its applicable terms.

The dataset root is configured with:

```bash
export MSLS_ROOT=/path/to/msls
```

Inside the provided container workflow, MSLS is mounted at:

```text
/data/msls
```

## 2. External repositories

The code depends on two external repositories:

- Mapillary SLS support code;
- VGGT.

For a local installation, configure:

```bash
export MAPILLARY_SLS_ROOT=/path/to/mapillary_sls
export VGGT_ROOT=/path/to/vggt
```

Inside the provided container they are available at:

```text
/opt/mapillary_sls
/opt/vggt
```

Mapillary SLS requires the compatibility patch documented under [`patches/`](../patches/README.md).

## 3. Positive-pair files

Generate MSLS positives with:

```bash
python src/data/generate_msls_positives.py
```

The generated layout is:

```text
positives/
├── train/
│   ├── <city>/
│   │   ├── query_nodes_train.csv
│   │   ├── database_nodes_train.csv
│   │   ├── positives_train.npy
│   │   └── summary_train.csv
│   └── ...
└── val/
    ├── cph/
    │   ├── query_nodes_val.csv
    │   ├── database_nodes_val.csv
    │   ├── positives_val.npy
    │   └── summary_val.csv
    └── sf/
        └── ...
```

Each `positives_<split>.npy` stores a dictionary containing:

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

`pIdx[i]` contains the local database indices considered positive for query `i`.

## 4. VGGT overlap outputs

Generate the 3D geometric overlap with:

```bash
python src/overlap/compute_vggt_overlap.py
```

The generated layout is:

```text
overlaps/
└── vggt/
    ├── <city>/
    │   ├── place_<id>/
    │   │   ├── i_<image>.npy
    │   │   ├── o_<positive>.npy
    │   │   ├── d_<image>.png
    │   │   └── DONE.txt
    │   └── ...
    └── ...
```

Depth-map PNG files are saved only when depth saving is enabled in the overlap script.

A pair-level `o_*.npy` file stores information including:

```text
reference_image
reference_path
positive_image
positive_path
block_index
overlap_ref_to_pos
overlap_pos_to_ref
overlap
```

The main training weight is:

```text
overlap = min(overlap_ref_to_pos, overlap_pos_to_ref)
```

Intrinsic matrices are stored in `i_*.npy` files.

## 5. 2D overlap outputs

Generate the 2D field-of-view overlap with:

```bash
python src/overlap/compute_2d_overlap.py
```

The generated layout is:

```text
overlaps/
└── 2d/
    └── <city>/
        └── place_<id>/
            ├── i_<image>.npy
            ├── o_<positive>.npy
            └── DONE.txt
```

The pair-level files store the 2D overlap values and auxiliary geometric information. The value used for training is the field-of-view IoU.

## 6. Analysis cache

The overlap-analysis scripts generate city-level caches under:

```text
overlap_cache/
├── overlaps_<city>.pkl
├── overlaps_<city>.csv
├── summary_overlaps_<city>.pkl
├── summary_overlaps_<city>.csv
├── global_analysis/
└── plots/
```

These files are generated analysis artifacts and are not tracked by Git.

## 7. Published quantitative results

The repository includes a lightweight set of quantitative results under:

```text
results/
```

This directory contains:

- epoch-level metric histories for the main experiment;
- the available metric files for the variability experiments;
- compact CSV summary tables;
- a city-level VGGT overlap summary.

See [`results/README.md`](../results/README.md) for the exact structure and metric conventions.

## 8. Files excluded from the repository

The repository does not include:

- the MSLS dataset;
- VGGT checkpoints;
- trained VPR checkpoints;
- full pair-level overlap collections;
- depth-map collections;
- overlap caches;
- SLURM logs;
- container images;
- qualitative retrieval outputs containing dataset image paths or MSLS images;
- credentials or private storage links.
