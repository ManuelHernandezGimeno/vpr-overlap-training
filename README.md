# Overlap-Aware Training for Visual Place Recognition

Code and documentation developed for a Bachelor's Thesis on **Visual Place Recognition (VPR)** at the University of Zaragoza.

The project studies whether the visual overlap between geographically positive image pairs can be used as an additional training signal for VPR. The main idea is to estimate geometric overlap with **VGGT**, then use that overlap to weight the contribution of positive pairs in a **Triplet Loss** training pipeline based on **ResNet50 + GeM**.

## Overview

In large-scale VPR datasets such as **Mapillary Street-Level Sequences (MSLS)**, positive pairs are commonly defined using geographic proximity. Two nearby images, however, may observe different parts of the scene and share little or no visual content.

This repository investigates four training configurations:

1. **Baseline** — standard Triplet Loss without overlap information.
2. **Continuous overlap** — the complete triplet loss is weighted by the VGGT-based overlap value.
3. **Binary overlap** — triplets with zero overlap receive zero weight; all non-zero-overlap triplets keep full weight.
4. **2D overlap** — a field-of-view-based overlap formulation used as a comparison method.

The VGGT-based overlap is estimated from predicted depth maps and camera parameters by unprojecting pixels from one image, transforming the resulting 3D points, and reprojecting them into the other view. Overlap is computed in both directions and the minimum directional value is used as a conservative estimate.

## Main contributions

- A pipeline for generating MSLS positive pairs used by the overlap computation.
- A 3D geometric overlap estimator based on VGGT predictions.
- Continuous and binary overlap-aware variants of Triplet Loss.
- A 2D overlap baseline for comparison.
- Training and validation pipelines for ResNet50 + GeM VPR models.
- Recall@1, Recall@5 and Recall@10 evaluation.
- Checkpointing, training resumption and early stopping.
- Per-city validation to avoid mixing database indices across MSLS cities.
- Quantitative and qualitative analysis of overlap values and VPR retrievals.
- SLURM scripts used to run the large-scale experiments on a DGX-1 cluster.

## Repository structure

```text
vpr-overlap-training/
├── README.md
├── LICENSE
├── CITATION.cff
├── THIRD_PARTY.md
├── CONTRIBUTING.md
├── environment.yml
├── .gitignore
├── config/
│   └── env.example
├── src/
│   ├── data/
│   │   └── generate_msls_positives.py
│   ├── overlap/
│   │   ├── compute_vggt_overlap.py
│   │   ├── compute_2d_overlap.py
│   │   ├── analyze_overlaps.py
│   │   └── overlap_io.py
│   ├── training/
│   │   ├── train_baseline.py
│   │   ├── train_overlap_continuous.py
│   │   ├── train_overlap_binary.py
│   │   └── train_overlap_2d.py
│   └── evaluation/
│       └── generate_qualitative_retrievals.py
├── cluster/
│   ├── submit_train.sh
│   ├── run_train_job.sh
│   ├── run_analysis_job.sh
│   └── container/
├── docs/
│   ├── METHOD.md
│   ├── DATA.md
│   ├── REPRODUCIBILITY.md
│   └── RESULTS.md
└── results/
    └── README.md
```

Large datasets, VGGT predictions, overlap files, model checkpoints and SLURM logs are intentionally not tracked by Git.

## Installation

### 1. Create the Python environment

```bash
conda env create -f environment.yml
conda activate vpr-overlap
```

### 2. Install MSLS support code

Clone the official Mapillary Street-Level Sequences repository outside this repository:

```bash
git clone https://github.com/mapillary/mapillary_sls.git
```

Set `MAPILLARY_SLS_ROOT` to the cloned directory.

### 3. Install VGGT

Clone the official VGGT repository outside this repository and follow its installation instructions:

```bash
git clone https://github.com/facebookresearch/vggt.git
```

Set `VGGT_ROOT` to the cloned directory.

> This repository does not redistribute VGGT code, VGGT weights, or the MSLS dataset. Users are responsible for complying with the corresponding licenses and dataset terms.

## Dataset

This project uses **Mapillary Street-Level Sequences (MSLS)**. The dataset itself is not included in this repository.

Set the relevant paths before running the code. An example is provided in [`config/env.example`](config/env.example).

See [`docs/DATA.md`](docs/DATA.md) for the expected data layout and the generated positive/overlap file formats.

## Workflow

### 1. Generate positive pairs

```bash
python src/data/generate_msls_positives.py
```

The generated files contain query paths, database paths, positive indices and place identifiers for each city.

### 2. Compute VGGT geometric overlap

```bash
python src/overlap/compute_vggt_overlap.py
```

For each query-positive pair, overlap is computed in both projection directions. The final value is:

```text
overlap = min(overlap_query_to_positive, overlap_positive_to_query)
```

### 3. Compute the 2D overlap baseline

```bash
python src/overlap/compute_2d_overlap.py
```

### 4. Train the VPR models

```bash
python src/training/train_baseline.py
python src/training/train_overlap_continuous.py
python src/training/train_overlap_binary.py
python src/training/train_overlap_2d.py
```

The models use a **ResNet50 backbone**, **GeM pooling**, L2-normalized global descriptors and Triplet Loss.

### 5. Evaluate and inspect retrievals

Validation reports **Recall@1**, **Recall@5** and **Recall@10**. Qualitative Top-K retrieval examples can be generated with:

```bash
python src/evaluation/generate_qualitative_retrievals.py
```

## Cluster execution

The final large-scale experiments were run with SLURM on an NVIDIA DGX-1 system. Cluster-specific launcher scripts are provided under [`cluster/`](cluster/).

The public version of these scripts should use environment variables rather than personal or institution-specific absolute paths.

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Results

The experiments show that overlap-aware training is competitive with the baseline and with the 2D overlap comparison. The continuous formulation tends to reach competitive performance faster, while the binary formulation can benefit from longer training and may achieve stronger final results in some experiments.

The analysis also shows that overlap estimates can help identify geographically positive pairs that share little visual content. At the same time, VGGT-based estimates can fail in difficult conditions, which makes the quality of the geometric model an important limitation of the approach.

A summary of the experimental conclusions is provided in [`docs/RESULTS.md`](docs/RESULTS.md). Machine-readable metrics and selected figures can be stored under [`results/`](results/).

## Reproducibility

Training scripts save:

- one checkpoint per epoch,
- the best model according to validation Recall@1,
- training loss,
- Recall@1 / Recall@5 / Recall@10,
- JSON and CSV metric histories.

The code also supports resuming training from the latest saved epoch.

More details are available in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Citation

If you use this repository, please cite it using the metadata in [`CITATION.cff`](CITATION.cff).

This repository accompanies a Bachelor's Thesis carried out at the University of Zaragoza in 2026.

## Third-party software and data

This project depends on external research software and data, including VGGT and MSLS. Their original licenses and terms remain applicable.

See [`THIRD_PARTY.md`](THIRD_PARTY.md) for details.

## License

The original code in this repository is released under the MIT License unless a file explicitly states otherwise. Third-party components are not relicensed by this repository.

See [`LICENSE`](LICENSE) and [`THIRD_PARTY.md`](THIRD_PARTY.md).
