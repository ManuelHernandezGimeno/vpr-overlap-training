# Experiments

## 1. Main comparison

The repository contains four training scripts:

```text
src/training/train_baseline.py
src/training/train_overlap_continuous.py
src/training/train_overlap_binary.py
src/training/train_overlap_2d.py
```

The intended comparison changes the overlap treatment while keeping the remaining setup as consistent as possible.

## 2. Main training configuration

The current public scripts use:

| Parameter | Value |
|---|---:|
| Backbone | ResNet50 |
| Aggregation | GeM |
| Number of negatives | 5 |
| Positive distance threshold | 25 m |
| Negative distance threshold | 25 m |
| Cached queries | 5000 |
| Cached negatives | 5000 |
| Optimizer | Adam |
| Learning rate | 1e-5 |
| Weight decay | 1e-6 |
| Maximum epochs | 100 |
| Early-stopping patience | 10 |
| Minimum R@1 improvement | 0.001 |
| Training batch size | 32 |
| Descriptor batch size | 512 |
| Distance batch size | 512 |
| Training workers | 16 |
| Validation workers | 16 |
| Validation cities | Copenhagen (`cph`), San Francisco (`sf`) |

These values should be checked once more against the final submitted thesis before creating `v1.0.0`.

## 3. Image transformations

Training:

```text
Resize(256, 256)
RandomResizedCrop(224, scale=(0.7, 1.0))
ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
RandomGrayscale(p=0.1)
GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
ImageNet normalization
```

Validation uses deterministic resizing to 224x224 followed by ImageNet normalization.

## 4. Main commands

```bash
python src/training/train_baseline.py
python src/training/train_overlap_continuous.py
python src/training/train_overlap_binary.py
python src/training/train_overlap_2d.py
```

Recommended outputs:

```text
outputs/
├── baseline/
├── continuous/
├── binary/
└── overlap_2d/
```

## 5. Validation metrics

```text
Recall@1
Recall@5
Recall@10
```

Validation is performed independently for each city and then aggregated.

## 6. Variability experiments

### 6.1 Longer early stopping

The thesis also studies a patience value of 20 epochs without sufficient R@1 improvement.

For the final public release, this parameter should be configurable, for example:

```bash
EARLY_STOPPING_PATIENCE=20 python src/training/train_overlap_binary.py
```

### 6.2 Copenhagen-only validation

The thesis also includes an analysis based on Copenhagen without the influence of San Francisco.

For the final public release, validation cities should be configurable, for example:

```bash
VAL_CITIES=cph python src/training/train_overlap_binary.py
```

## 7. Cluster execution

The large-scale experiments were executed through SLURM on an NVIDIA DGX-1 system.

Recommended public container paths:

```text
Repository: /workspace/vpr-overlap-training
MSLS:       /data/msls
VGGT:       /opt/vggt
Mapillary:  /opt/mapillary_sls
```
