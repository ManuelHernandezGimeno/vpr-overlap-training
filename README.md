# Training Visual Place Recognition Models Using View Overlap

Code, documentation, and quantitative results developed for a Bachelor's Thesis on **Visual Place Recognition (VPR)** at the University of Zaragoza.

The project studies whether visual overlap between geographically positive image pairs can be used as an additional training signal for VPR. Geometric overlap is estimated with **VGGT** and incorporated into a **ResNet50 + GeM** Triplet Loss training pipeline on **Mapillary Street-Level Sequences (MSLS)**.

## Overview

MSLS positive pairs are defined from geographic proximity, but geographically close images can observe different parts of the environment and share little visible content.

This repository compares four training configurations:

1. **Baseline** — standard Triplet Loss without overlap information.
2. **Continuous overlap** — the complete Triplet Loss is weighted by the continuous VGGT overlap value.
3. **Binary overlap** — pairs with zero VGGT overlap receive zero weight; non-zero pairs retain full weight.
4. **2D overlap** — a field-of-view-based overlap formulation used as a comparison method.

The VGGT-based overlap is computed in both projection directions and the final pair value is:

```text
overlap = min(overlap_reference_to_positive, overlap_positive_to_reference)
```

## Main contributions

- generation of direct MSLS query-positive pairs by city;
- VGGT-based 3D geometric overlap estimation;
- continuous and binary overlap-aware Triplet Loss variants;
- a 2D field-of-view overlap comparison;
- ResNet50 + GeM VPR training and validation;
- configurable validation cities and early-stopping patience;
- Recall@1, Recall@5, and Recall@10 evaluation;
- checkpointing, training resumption, and early stopping;
- overlap-analysis utilities;
- qualitative Top-K retrieval inspection;
- Docker, Enroot, and SLURM execution scripts;
- machine-readable quantitative results.

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
├── .dockerignore
│
├── config/
│   └── env.example
│
├── patches/
│   ├── README.md
│   └── mapillary_sls_numpy_object.patch
│
├── scripts/
│   └── apply_msls_patch.sh
│
├── src/
│   ├── data/
│   ├── overlap/
│   ├── analysis/
│   ├── training/
│   └── evaluation/
│
├── cluster/
│   ├── README.md
│   ├── submit_train.sh
│   ├── run_train_job.sh
│   ├── submit_analysis.sh
│   ├── run_analysis_job.sh
│   └── container/
│
├── docs/
│   ├── METHOD.md
│   ├── DATA.md
│   ├── EXPERIMENTS.md
│   ├── REPRODUCIBILITY.md
│   └── RESULTS.md
│
└── results/
    ├── README.md
    ├── metrics/
    ├── summaries/
    └── overlap_analysis/
```

## Installation

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate tfg_vpr
```

Clone the external dependencies:

```bash
git clone https://github.com/mapillary/mapillary_sls.git /path/to/mapillary_sls
git clone https://github.com/facebookresearch/vggt.git /path/to/vggt
```

Apply the Mapillary SLS compatibility patch:

```bash
export MAPILLARY_SLS_ROOT=/path/to/mapillary_sls
bash scripts/apply_msls_patch.sh
```

Create a local configuration:

```bash
cp config/env.example config/env.sh
```

Edit it and load it:

```bash
source config/env.sh
```

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the complete workflow.

## Dataset

This project uses **Mapillary Street-Level Sequences (MSLS)**. The dataset is not included.

Configure its location with:

```bash
export MSLS_ROOT=/path/to/msls
```

See [`docs/DATA.md`](docs/DATA.md) for the generated data layouts.

## Main workflow

Generate positive pairs:

```bash
python src/data/generate_msls_positives.py
```

Compute VGGT overlap:

```bash
python src/overlap/compute_vggt_overlap.py
```

Compute 2D overlap:

```bash
python src/overlap/compute_2d_overlap.py
```

Analyze overlaps:

```bash
python src/analysis/read_overlaps.py
python src/analysis/global_summary.py
```

Train the four main models:

```bash
python src/training/train_baseline.py
python src/training/train_overlap_continuous.py
python src/training/train_overlap_binary.py
python src/training/train_overlap_2d.py
```

## Configurable training settings

The training scripts read two experiment-level environment variables:

```text
VAL_CITIES
EARLY_STOPPING_PATIENCE
```

The main-experiment defaults are:

```bash
export VAL_CITIES="cph,sf"
export EARLY_STOPPING_PATIENCE="10"
```

A Copenhagen-only run can be launched with:

```bash
VAL_CITIES=cph python src/training/train_overlap_binary.py
```

A longer early-stopping run can be launched with:

```bash
EARLY_STOPPING_PATIENCE=20 python src/training/train_overlap_binary.py
```

See [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) for the complete main and variability configurations, including separate output directories.

## Cluster execution

The large-scale experiments were executed with SLURM on an NVIDIA DGX-1 system.

The Docker, Enroot, and SLURM workflow is documented in:

[`cluster/README.md`](cluster/README.md)

The same `VAL_CITIES` and `EARLY_STOPPING_PATIENCE` variables can be exported before submitting a training job.

## Results

Machine-readable metric histories and summary tables are available under [`results/`](results/).

### Main validation

| Method | Best epoch | R@1 (%) | R@5 (%) | R@10 (%) |
|---|---:|---:|---:|---:|
| Baseline | 18 | 50.0 | 58.8 | 60.1 |
| Continuous overlap | 13 | 51.5 | 59.9 | 62.6 |
| Binary overlap | 23 | 51.5 | 59.5 | 62.7 |
| 2D overlap | 27 | **51.6** | **60.7** | **63.5** |

Detailed results are available in [`docs/RESULTS.md`](docs/RESULTS.md), and the published CSV files are documented in [`results/README.md`](results/README.md).

## Reproducibility

The training scripts maintain:

- CSV and JSON metric histories;
- the best model according to validation Recall@1;
- recent epoch checkpoints;
- training resumption from the latest saved epoch.

The selected validation cities and early-stopping patience are stored in checkpoint metadata.

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Documentation

- [`docs/METHOD.md`](docs/METHOD.md) — model, overlap definitions, losses, and validation.
- [`docs/DATA.md`](docs/DATA.md) — MSLS and generated file layouts.
- [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) — main and variability configurations.
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) — end-to-end execution.
- [`docs/RESULTS.md`](docs/RESULTS.md) — quantitative results.
- [`cluster/README.md`](cluster/README.md) — cluster workflow.
- [`results/README.md`](results/README.md) — published result files.

## Citation

If you use this repository in academic work, cite it using [`CITATION.cff`](CITATION.cff).

This repository accompanies a Bachelor's Thesis carried out at the University of Zaragoza in 2026.

## Third-party software and data

The project depends on VGGT, Mapillary SLS, MSLS, PyTorch, and other external components.

Their original licenses and usage terms remain applicable. See [`THIRD_PARTY.md`](THIRD_PARTY.md).

## License

The original code in this repository is released under the MIT License unless a file explicitly states otherwise.
