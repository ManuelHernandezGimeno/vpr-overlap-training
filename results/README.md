# Results

This directory contains the lightweight quantitative results associated with the Bachelor's Thesis on overlap-aware training for Visual Place Recognition.

The purpose of this directory is to provide the numerical results reported in the thesis without distributing the MSLS dataset, VGGT outputs, model checkpoints, or other large generated artifacts.

## Directory structure

```text
results/
├── README.md
│
├── metrics/
│   ├── main/
│   │   ├── baseline_metrics_history.csv
│   │   ├── continuous_overlap_metrics_history.csv
│   │   ├── binary_overlap_metrics_history.csv
│   │   └── overlap_2d_metrics_history.csv
│   │
│   └── variability/
│       ├── patience_20/
│       │   ├── baseline_metrics_history.csv
│       │   ├── continuous_overlap_metrics_history.csv
│       │   ├── binary_overlap_metrics_history.csv
│       │   └── overlap_2d_metrics_history.csv
│       │
│       └── copenhagen_only/
│           ├── baseline_best_model_val_metrics.csv
│           ├── continuous_overlap_best_model_val_metrics.csv
│           ├── binary_overlap_best_model_val_metrics.csv
│           └── overlap_2d_best_model_val_metrics.csv
│
├── summaries/
│   ├── main_validation_results.csv
│   ├── city_validation_results.csv
│   ├── variability_results.csv
│   └── training_time_results.csv
│
└── overlap_analysis/
    └── overlap_city_summary.csv
```

## Metric conventions

The raw metric files store Recall@K values as fractions in the range `[0, 1]`.

The compact summary CSV files use percentages in the range `[0, 100]`.

For example:

```text
raw value:     0.699
summary value: 69.9
```

## Main experiment

The files under `metrics/main/` contain the complete epoch-by-epoch histories of the four main training configurations:

- `baseline_metrics_history.csv`: standard Triplet Loss without overlap weighting;
- `continuous_overlap_metrics_history.csv`: Triplet Loss weighted by the continuous VGGT overlap;
- `binary_overlap_metrics_history.csv`: binary VGGT overlap weighting;
- `overlap_2d_metrics_history.csv`: Triplet Loss weighted by the 2D field-of-view overlap.

The selected best-model results are summarized in `summaries/main_validation_results.csv`.

| Method | Best epoch | R@1 (%) | R@5 (%) | R@10 (%) |
|---|---:|---:|---:|---:|
| Baseline | 18 | 50.0 | 58.8 | 60.1 |
| Continuous overlap | 13 | 51.5 | 59.9 | 62.6 |
| Binary overlap | 23 | 51.5 | 59.5 | 62.7 |
| 2D overlap | 27 | **51.6** | **60.7** | **63.5** |

## City-level validation

The city-level values associated with the main experiment are stored in `summaries/city_validation_results.csv`.

### Copenhagen

| Method | R@1 (%) | R@5 (%) | R@10 (%) |
|---|---:|---:|---:|
| Baseline | 65.5 | 77.3 | 78.7 |
| Continuous overlap | 67.9 | 79.3 | 83.1 |
| Binary overlap | **68.3** | 78.7 | 83.1 |
| 2D overlap | 67.9 | **80.1** | **83.9** |

### San Francisco

| Method | R@1 (%) | R@5 (%) | R@10 (%) |
|---|---:|---:|---:|
| Baseline | **18.2** | **20.7** | **21.9** |
| Continuous overlap | 17.8 | 19.8 | 20.2 |
| Binary overlap | 16.9 | 19.8 | 20.7 |
| 2D overlap | **18.2** | **20.7** | 21.5 |

## Variability analysis

The thesis includes two additional experiments evaluated on Copenhagen. Their final results are collected in:

```text
summaries/variability_results.csv
```

The amount of raw data available is different for the two experiments.

### Experiment 1 — longer early-stopping patience

The complete epoch-by-epoch histories for Experiment 1 are available under:

```text
metrics/variability/patience_20/
```

This experiment uses a longer early-stopping patience and preserves the complete training history for all four methods.

| Method | R@1 (%) | R@5 (%) | R@10 (%) |
|---|---:|---:|---:|
| Baseline | 65.5 | 77.3 | 78.7 |
| Continuous overlap | 68.1 | 79.3 | 83.7 |
| Binary overlap | **68.5** | 78.5 | 81.7 |
| 2D overlap | 67.9 | **80.1** | **83.9** |

### Experiment 2 — Copenhagen-focused configuration

For Experiment 2, only the exported validation metrics of the selected best model are available. They are stored under:

```text
metrics/variability/copenhagen_only/
```

Each file contains the validation metrics saved for the selected model. The `cph` row is the one used in the variability table reported in the thesis.

| Method | R@1 (%) | R@5 (%) | R@10 (%) |
|---|---:|---:|---:|
| Baseline | 64.9 | 79.1 | 82.7 |
| Continuous overlap | 69.7 | 80.9 | **85.1** |
| Binary overlap | **69.9** | 79.1 | 82.5 |
| 2D overlap | 68.7 | **81.3** | 84.5 |

No epoch-by-epoch training-history files are available for Experiment 2.

## Summary tables

The `summaries/` directory contains compact tables used to reproduce the quantitative tables in the thesis.

### `main_validation_results.csv`

Main validation results for the four methods:

```text
method,best_epoch,recall_at_1,recall_at_5,recall_at_10
```

### `city_validation_results.csv`

City-level results for the main validation experiment:

```text
method,city,recall_at_1,recall_at_5,recall_at_10
```

### `variability_results.csv`

Final Copenhagen results for the two variability experiments:

```text
experiment,method,recall_at_1,recall_at_5,recall_at_10
```

### `training_time_results.csv`

Training-time comparison for the main experiment:

```text
method,best_epoch,time_per_epoch_hours,total_training_time_days
```

The recorded values are:

| Method | Best epoch | Time per epoch (h) | Total training time (days) |
|---|---:|---:|---:|
| Baseline | 18 | 6.48 | 7.55 |
| Continuous overlap | 13 | 6.67 | 6.39 |
| Binary overlap | 23 | 6.70 | 9.22 |
| 2D overlap | 27 | 6.68 | 10.31 |

## Overlap analysis

`overlap_analysis/overlap_city_summary.csv` contains the final city-level statistics obtained from the VGGT overlap analysis.

The file covers:

- 22 MSLS training cities;
- 8,578,944 query-positive pairs;
- 3,027,526 zero-overlap pairs;
- 35.29% zero-overlap pairs overall;
- a pair-weighted mean overlap of approximately 0.1296.

The CSV contains the following columns:

```text
city
num_places
num_query_positive_pairs
num_zero_overlap_pairs
zero_overlap_percentage
weighted_mean_overlap
num_places_with_zero_mean_overlap
```

Pair-level overlap files, per-place caches, depth maps, and camera intrinsics are intentionally not included.

## Files intentionally excluded

The following artifacts are not stored under `results/`:

- MSLS images or archives;
- VGGT checkpoints;
- trained VPR checkpoints;
- epoch checkpoints;
- `best_model.ckpt`;
- `best_metrics.npy`;
- pair-level overlap `.npy` files;
- depth maps and camera intrinsics;
- overlap `.pkl` caches;
- qualitative retrieval CSV files containing image paths;
- qualitative retrieval figures containing MSLS images;
- SLURM logs;
- container images.

## Relationship with the thesis

The summary CSV files are intended to reproduce the quantitative tables reported in the final thesis.

For the variability analysis:

- Experiment 1 includes complete epoch-by-epoch metric histories.
- Experiment 2 includes only the validation metrics exported from the selected best models.

The difference in available raw files reflects the artifacts preserved from each experiment and does not affect the summary values reported in the thesis.
