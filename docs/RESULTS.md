# Results

> Verify every quantitative value below against the final submitted thesis and the final metric-history files before creating the public `v1.0.0` release.

## 1. Main validation

| Method | R@1 (%) | R@5 (%) | R@10 (%) |
|---|---:|---:|---:|
| Baseline | 50.0 | 58.8 | 60.1 |
| Continuous VGGT overlap | 51.5 | 59.9 | 62.6 |
| Binary VGGT overlap | 51.5 | 59.5 | 62.7 |
| 2D overlap | **51.6** | **60.7** | **63.5** |

## 2. Copenhagen

| Method | R@1 (%) | R@5 (%) | R@10 (%) |
|---|---:|---:|---:|
| Baseline | 65.5 | 77.3 | 78.7 |
| Continuous VGGT overlap | 67.9 | 79.3 | 83.1 |
| Binary VGGT overlap | **68.3** | 78.7 | 83.1 |
| 2D overlap | 67.9 | **80.1** | **83.9** |

## 3. San Francisco

| Method | R@1 (%) | R@5 (%) | R@10 (%) |
|---|---:|---:|---:|
| Baseline | **18.2** | **20.7** | **21.9** |
| Continuous VGGT overlap | 17.8 | 19.8 | 20.2 |
| Binary VGGT overlap | 16.9 | 19.8 | 20.7 |
| 2D overlap | **18.2** | **20.7** | 21.5 |

## 4. Training time

| Method | Best epoch | Time / epoch (h) | Total time (days) |
|---|---:|---:|---:|
| Baseline | 18 | 6.48 | 7.55 |
| Continuous VGGT overlap | 13 | 6.67 | 6.39 |
| Binary VGGT overlap | 23 | 6.70 | 9.22 |
| 2D overlap | 27 | 6.68 | 10.31 |

The continuous formulation reaches its best model earlier in the main experiment. Additional longer-training experiments indicate that binary weighting can benefit from a more permissive stopping criterion.

## 5. Overlap analysis

The geometric-overlap analysis covers approximately 8.58 million query-positive pairs.

Main observations:

- approximately 35.29% of the analyzed pairs have zero final overlap;
- the global pair-weighted mean overlap is approximately 0.13;
- overlap behavior varies substantially across cities;
- zero overlap is not automatically equivalent to an incorrect geographic positive;
- difficult lighting, viewpoint, open-scene geometry, and VGGT failures can also produce low overlap.

The VGGT overlap is intentionally conservative:

```text
overlap = min(overlap_ref_to_pos, overlap_pos_to_ref)
```

## 6. Qualitative conclusions

The qualitative analysis identifies:

1. geographically close pairs with genuinely low shared visual content;
2. failure cases caused by imperfect VGGT geometry;
3. high-overlap examples in dynamic urban scenes.

The public repository should also include the verified lightweight CSV metric histories and selected figures under `results/`.
