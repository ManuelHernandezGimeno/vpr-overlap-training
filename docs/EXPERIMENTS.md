# Experiments

This document describes the experimental configurations implemented by the training scripts and the quantitative results published in this repository.

## 1. Compared methods

The project compares four VPR training configurations:

```text
src/training/train_baseline.py
src/training/train_overlap_continuous.py
src/training/train_overlap_binary.py
src/training/train_overlap_2d.py
```

The comparison changes the use of visual overlap while keeping the remaining training setup aligned across methods.

## 2. Main training configuration

The main experiment uses:

| Parameter | Value |
|---|---:|
| Backbone | ResNet50 |
| Aggregation | GeM |
| Descriptor dimension | 512 |
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

The default values used by the training scripts are:

```text
EARLY_STOPPING_PATIENCE=10
VAL_CITIES=cph,sf
```

Both settings can be changed through environment variables without editing the Python source files.

## 3. Image transformations

Training images use:

```text
Resize(256, 256)
RandomResizedCrop(224, scale=(0.7, 1.0))
ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
RandomGrayscale(p=0.1)
GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
ImageNet normalization
```

Validation uses deterministic resizing to 224x224 followed by ImageNet normalization.

## 4. Main experiment

The default settings reproduce the main training configuration:

```bash
python src/training/train_baseline.py
python src/training/train_overlap_continuous.py
python src/training/train_overlap_binary.py
python src/training/train_overlap_2d.py
```

The default output directories are:

```text
outputs/
├── baseline/
├── continuous/
├── binary/
└── overlap_2d/
```

Each run stores:

- CSV and JSON metric histories;
- the best model according to validation Recall@1;
- recent epoch checkpoints.

The epoch-level metric histories selected for publication are available in:

```text
results/metrics/main/
```

## 5. Validation

Validation is performed independently for each city in `VAL_CITIES`.

The default value is:

```bash
VAL_CITIES=cph,sf
```

The reported metrics are:

```text
Recall@1
Recall@5
Recall@10
```

City-level retrieval is computed using only the database and local positive indices associated with that city. The global validation result is then aggregated from the city-level evaluations.

## 6. Configurable experiment variables

The four training scripts read the following experiment-level environment variables:

```text
VAL_CITIES
EARLY_STOPPING_PATIENCE
```

### Validation cities

`VAL_CITIES` is a comma-separated list.

Examples:

```bash
VAL_CITIES=cph,sf python src/training/train_overlap_continuous.py
```

```bash
VAL_CITIES=cph python src/training/train_overlap_continuous.py
```

### Early-stopping patience

`EARLY_STOPPING_PATIENCE` is an integer.

Example:

```bash
EARLY_STOPPING_PATIENCE=20 python src/training/train_overlap_binary.py
```

If these variables are not defined, the scripts use the main-experiment defaults:

```text
VAL_CITIES=cph,sf
EARLY_STOPPING_PATIENCE=10
```

## 7. Variability experiments

Two additional configurations are included in the thesis.

### 7.1 Experiment 1 — longer early-stopping patience

Experiment 1 uses the same validation cities as the main experiment but increases the early-stopping patience from 10 to 20 epochs without sufficient Recall@1 improvement.

The configuration is:

```text
VAL_CITIES=cph,sf
EARLY_STOPPING_PATIENCE=20
```

To avoid mixing the generated files with the main experiment, use a separate output directory for each method.

#### Baseline

```bash
EARLY_STOPPING_PATIENCE=20 VAL_CITIES=cph,sf OUTPUTNEW2_ROOT="${PROJECT_ROOT}/outputs/variability/patience_20/baseline" python src/training/train_baseline.py
```

#### Continuous overlap

```bash
EARLY_STOPPING_PATIENCE=20 VAL_CITIES=cph,sf OUTPUT_OVERLAP2_ROOT="${PROJECT_ROOT}/outputs/variability/patience_20/continuous" python src/training/train_overlap_continuous.py
```

#### Binary overlap

```bash
EARLY_STOPPING_PATIENCE=20 VAL_CITIES=cph,sf OUTPUT_OVERLAP_BINARIO2_ROOT="${PROJECT_ROOT}/outputs/variability/patience_20/binary" python src/training/train_overlap_binary.py
```

#### 2D overlap

```bash
EARLY_STOPPING_PATIENCE=20 VAL_CITIES=cph,sf OUTPUT_OVERLAP_2D2_ROOT="${PROJECT_ROOT}/outputs/variability/patience_20/overlap_2d" python src/training/train_overlap_2d.py
```

Complete epoch-level histories for this experiment are published under:

```text
results/metrics/variability/patience_20/
```

### 7.2 Experiment 2 — Copenhagen-only model selection

Experiment 2 selects the best model using Copenhagen without the influence of San Francisco.

The configuration is:

```text
VAL_CITIES=cph
EARLY_STOPPING_PATIENCE=10
```

Again, separate output directories are recommended.

#### Baseline

```bash
VAL_CITIES=cph EARLY_STOPPING_PATIENCE=10 OUTPUTNEW2_ROOT="${PROJECT_ROOT}/outputs/variability/copenhagen_only/baseline" python src/training/train_baseline.py
```

#### Continuous overlap

```bash
VAL_CITIES=cph EARLY_STOPPING_PATIENCE=10 OUTPUT_OVERLAP2_ROOT="${PROJECT_ROOT}/outputs/variability/copenhagen_only/continuous" python src/training/train_overlap_continuous.py
```

#### Binary overlap

```bash
VAL_CITIES=cph EARLY_STOPPING_PATIENCE=10 OUTPUT_OVERLAP_BINARIO2_ROOT="${PROJECT_ROOT}/outputs/variability/copenhagen_only/binary" python src/training/train_overlap_binary.py
```

#### 2D overlap

```bash
VAL_CITIES=cph EARLY_STOPPING_PATIENCE=10 OUTPUT_OVERLAP_2D2_ROOT="${PROJECT_ROOT}/outputs/variability/copenhagen_only/overlap_2d" python src/training/train_overlap_2d.py
```

For Experiment 2, the repository contains the exported validation metrics of the selected best models under:

```text
results/metrics/variability/copenhagen_only/
```

The `cph` row from these files is used in the variability table reported in the thesis.

## 8. Published variability results

The final Copenhagen results from both variability experiments are available in:

```text
results/summaries/variability_results.csv
```

Experiment 1 also includes complete training histories. For Experiment 2, only the available best-model validation metric files are published.

## 9. Training-time comparison

The main experiment records:

- the epoch of the selected best model;
- the approximate time per epoch;
- the total training time.

These values are available in:

```text
results/summaries/training_time_results.csv
```

## 10. Cluster execution

The same experiment variables can be used for SLURM jobs because the submission environment is propagated to the training process.

For example, Experiment 1 can be submitted with:

```bash
cd /path/to/vpr-overlap-training

export MSLS_HOST_ROOT=/path/to/msls
export EARLY_STOPPING_PATIENCE=20
export VAL_CITIES=cph,sf
export OUTPUT_OVERLAP_BINARIO2_ROOT="${PWD}/outputs/variability/patience_20/binary"

sbatch cluster/submit_train.sh     src/training/train_overlap_binary.py
```

For Experiment 2:

```bash
cd /path/to/vpr-overlap-training

export MSLS_HOST_ROOT=/path/to/msls
export EARLY_STOPPING_PATIENCE=10
export VAL_CITIES=cph
export OUTPUT_OVERLAP_BINARIO2_ROOT="${PWD}/outputs/variability/copenhagen_only/binary"

sbatch cluster/submit_train.sh     src/training/train_overlap_binary.py
```

The Docker, Enroot, and SLURM workflow is documented in [`../cluster/README.md`](../cluster/README.md).
