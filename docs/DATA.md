# Data

## 1. Mapillary Street-Level Sequences

The project uses **Mapillary Street-Level Sequences (MSLS)**.

The dataset is not distributed with this repository. Users must obtain it from the official source and comply with the applicable terms.

Set the dataset root with:

```bash
export MSLS_ROOT=/path/to/msls
```

Inside the public container configuration, the recommended mount point is:

```text
/data/msls
```

## 2. External repositories

The code also depends on:

- Mapillary SLS support code;
- VGGT.

Recommended variables:

```bash
export MAPILLARY_SLS_ROOT=/path/to/mapillary_sls
export VGGT_ROOT=/path/to/vggt
```

Inside the container:

```text
/opt/mapillary_sls
/opt/vggt
```

## 3. Positive-pair files

Generate positives with:

```bash
python src/data/generate_msls_positives.py
```

Recommended layout:

```text
positives/
├── train/
│   ├── amsterdam/
│   │   ├── query_nodes_train.csv
│   │   ├── database_nodes_train.csv
│   │   ├── positives_train.npy
│   │   └── summary_train.csv
│   └── ...
└── val/
    ├── cph/
    └── sf/
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

`pIdx[i]` contains local database positive indices for query `i`.

## 4. VGGT overlap outputs

Generate them with:

```bash
python src/overlap/compute_vggt_overlap.py
```

Recommended layout:

```text
overlaps/
└── vggt/
    ├── amsterdam/
    │   ├── place_0/
    │   │   ├── i_<image>.npy
    │   │   ├── o_<positive>.npy
    │   │   └── DONE.txt
    │   └── ...
    └── ...
```

A pair-level `o_*.npy` dictionary contains fields such as:

```text
reference_image
reference_path
positive_image
positive_path
block_index
relative_translation
relative_rotation
overlap_ref_to_pos
overlap_pos_to_ref
overlap
```

The final `overlap` is the minimum of the two directional overlaps.

## 5. 2D overlap outputs

Generate them with:

```bash
python src/overlap/compute_2d_overlap.py
```

Recommended layout:

```text
overlaps/
└── 2d/
    └── <city>/
        └── place_<id>/
            ├── i_<image>.npy
            ├── o_<positive>.npy
            └── DONE.txt
```

## 6. Analysis cache

Recommended generated cache layout:

```text
overlap_cache/
├── overlaps_<city>.pkl
├── overlaps_<city>.csv
├── summary_overlaps_<city>.pkl
├── summary_overlaps_<city>.csv
└── plots/
```

## 7. Files that should not be committed

Do not commit:

- MSLS data;
- VGGT checkpoints;
- full model checkpoints;
- full overlap collections;
- depth maps;
- caches;
- SLURM logs;
- container images;
- credentials or private storage links.

Final lightweight CSV tables and selected figures can be copied to `results/`.
