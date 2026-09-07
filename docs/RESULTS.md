# Results

This document summarizes the quantitative results included in the repository. Machine-readable metric histories and summary tables are available under [`results/`](../results/).

Recall values in the summary tables are expressed as percentages.

## 1. Main validation

| Method | Best epoch | R@1 (%) | R@5 (%) | R@10 (%) |
|---|---:|---:|---:|---:|
| Baseline | 18 | 50.0 | 58.8 | 60.1 |
| Continuous VGGT overlap | 13 | 51.5 | 59.9 | 62.6 |
| Binary VGGT overlap | 23 | 51.5 | 59.5 | 62.7 |
| 2D overlap | 27 | **51.6** | **60.7** | **63.5** |

All three overlap-aware configurations are competitive with the baseline in the main validation. The 2D configuration obtains the highest global Recall@1, Recall@5, and Recall@10 in this experiment, although the differences between overlap-aware methods are small.

The complete main metric histories are stored in:

```text
results/metrics/main/
```

The compact table is stored in:

```text
results/summaries/main_validation_results.csv
```

## 2. Copenhagen

| Method | R@1 (%) | R@5 (%) | R@10 (%) |
|---|---:|---:|---:|
| Baseline | 65.5 | 77.3 | 78.7 |
| Continuous VGGT overlap | 67.9 | 79.3 | 83.1 |
| Binary VGGT overlap | **68.3** | 78.7 | 83.1 |
| 2D overlap | 67.9 | **80.1** | **83.9** |

Copenhagen shows a clearer improvement from overlap-aware training. Binary VGGT overlap obtains the highest Recall@1, while the 2D overlap configuration obtains the highest Recall@5 and Recall@10.

## 3. San Francisco

| Method | R@1 (%) | R@5 (%) | R@10 (%) |
|---|---:|---:|---:|
| Baseline | **18.2** | **20.7** | **21.9** |
| Continuous VGGT overlap | 17.8 | 19.8 | 20.2 |
| Binary VGGT overlap | 16.9 | 19.8 | 20.7 |
| 2D overlap | **18.2** | **20.7** | 21.5 |

San Francisco does not show the same gain as Copenhagen. The baseline remains strongest at Recall@10, while the 2D configuration matches the baseline at Recall@1 and Recall@5.

The city-level table for the main experiment is stored in:

```text
results/summaries/city_validation_results.csv
```

## 4. Variability analysis

Two additional configurations are reported for Copenhagen.

### 4.1 Experiment 1 — longer early-stopping patience

Complete epoch-level histories are available in:

```text
results/metrics/variability/patience_20/
```

| Method | R@1 (%) | R@5 (%) | R@10 (%) |
|---|---:|---:|---:|
| Baseline | 65.5 | 77.3 | 78.7 |
| Continuous VGGT overlap | 68.1 | 79.3 | 83.7 |
| Binary VGGT overlap | **68.5** | 78.5 | 81.7 |
| 2D overlap | 67.9 | **80.1** | **83.9** |

The longer training window allows the binary formulation to reach the highest Recall@1 in this experiment.

### 4.2 Experiment 2 — Copenhagen-focused configuration

For Experiment 2, the repository contains the exported validation metrics of the selected best models:

```text
results/metrics/variability/copenhagen_only/
```

The `cph` row from each file is used in the table below.

| Method | R@1 (%) | R@5 (%) | R@10 (%) |
|---|---:|---:|---:|
| Baseline | 64.9 | 79.1 | 82.7 |
| Continuous VGGT overlap | 69.7 | 80.9 | **85.1** |
| Binary VGGT overlap | **69.9** | 79.1 | 82.5 |
| 2D overlap | 68.7 | **81.3** | 84.5 |

Binary overlap obtains the highest Recall@1, continuous overlap obtains the highest Recall@10, and the 2D configuration obtains the highest Recall@5.

The combined variability table is stored in:

```text
results/summaries/variability_results.csv
```

## 5. Training time

| Method | Best epoch | Time / epoch (h) | Total time (days) |
|---|---:|---:|---:|
| Baseline | 18 | 6.48 | 7.55 |
| Continuous VGGT overlap | 13 | 6.67 | 6.39 |
| Binary VGGT overlap | 23 | 6.70 | 9.22 |
| 2D overlap | 27 | 6.68 | 10.31 |

The per-epoch times are similar because the overlap values are precomputed before training.

The continuous formulation reaches its selected best model earlier in the main experiment. The variability analysis also shows that the relative ranking between overlap-aware variants can change when the training configuration is modified.

The machine-readable table is stored in:

```text
results/summaries/training_time_results.csv
```

## 6. VGGT overlap analysis

The city-level overlap summary covers:

- 22 MSLS training cities;
- 8,578,944 query-positive pairs;
- 3,027,526 zero-overlap pairs;
- 35.29% zero-overlap pairs overall;
- a pair-weighted mean overlap of approximately 0.1296.

These values show that a substantial fraction of geographically positive pairs receive zero geometric overlap under the VGGT-based estimator.

Zero overlap is not interpreted automatically as an incorrect geographic label. Low or zero values can also be caused by viewpoint, illumination, scene geometry, temporal changes, or limitations of the estimated geometry.

The city-level table is stored in:

```text
results/overlap_analysis/overlap_city_summary.csv
```

## 7. Qualitative observations

The qualitative analysis performed during the project identified three recurring types of examples:

1. geographically close images that genuinely share little visible content;
2. visually related pairs for which the VGGT-based geometry produces very low overlap;
3. difficult urban scenes in which substantial overlap is still recovered despite dynamic objects and appearance changes.

The qualitative retrieval images and path-level retrieval CSV files are not included in the public `results/` directory.
