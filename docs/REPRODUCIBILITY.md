# Reproducibility

This document describes the end-to-end workflow for reproducing positive-pair generation, overlap computation, VPR training, validation, and quantitative analysis.

## 1. Clone the repository

```bash
git clone https://github.com/ManuelHernandezGimeno/vpr-overlap-training.git
cd vpr-overlap-training
```

## 2. Create the Conda environment

The repository uses the root `environment.yml`:

```bash
conda env create -f environment.yml
conda activate tfg_vpr
```

## 3. Clone the external repositories

For local execution, clone Mapillary SLS and VGGT outside this repository:

```bash
git clone https://github.com/mapillary/mapillary_sls.git /path/to/mapillary_sls
git clone https://github.com/facebookresearch/vggt.git /path/to/vggt
```

Configure their locations with:

```bash
export MAPILLARY_SLS_ROOT=/path/to/mapillary_sls
export VGGT_ROOT=/path/to/vggt
```

Inside the container they are available at:

```text
/opt/mapillary_sls
/opt/vggt
```

## 4. Apply the Mapillary SLS compatibility patch

The project scripts do not modify Mapillary SLS at runtime.

For a local installation:

```bash
export MAPILLARY_SLS_ROOT=/path/to/mapillary_sls
bash scripts/apply_msls_patch.sh
```

The patch changes the conversion of the variable-length `pIdx` and `nonNegIdx` collections to NumPy object arrays.

The Dockerfile applies the same patch during image construction.

## 5. Configure paths

Create a local configuration:

```bash
cp config/env.example config/env.sh
```

Edit `config/env.sh`, then load it:

```bash
source config/env.sh
```

`config/env.sh` is excluded from Git because it contains machine-specific paths.

The principal path variables are:

```text
PROJECT_ROOT
MSLS_ROOT
MAPILLARY_SLS_ROOT
VGGT_ROOT
POSITIVES_ROOT
TRAIN_POSITIVES_ROOT
OVERLAP_ROOT
OVERLAP2D_ROOT
CACHE_DIR
PLOTS_DIR
```

The training-output variables are:

```text
OUTPUTNEW2_ROOT
OUTPUT_OVERLAP2_ROOT
OUTPUT_OVERLAP_BINARIO2_ROOT
OUTPUT_OVERLAP_2D2_ROOT
```

## 6. Training configuration variables

The four training scripts also read two experiment-level environment variables:

```text
VAL_CITIES
EARLY_STOPPING_PATIENCE
```

The defaults reproduce the main experiment:

```bash
export VAL_CITIES="cph,sf"
export EARLY_STOPPING_PATIENCE="10"
```

`VAL_CITIES` is a comma-separated list. For example:

```bash
export VAL_CITIES="cph"
```

selects Copenhagen as the only validation city.

`EARLY_STOPPING_PATIENCE` controls the number of consecutive epochs without sufficient Recall@1 improvement before early stopping.

For example:

```bash
export EARLY_STOPPING_PATIENCE="20"
```

uses a patience of 20 epochs.

These values can also be supplied for a single command:

```bash
VAL_CITIES=cph EARLY_STOPPING_PATIENCE=20 python src/training/train_overlap_binary.py
```

## 7. Generate MSLS positive pairs

```bash
python src/data/generate_msls_positives.py
```

The generated files are written under:

```text
positives/train/
positives/val/
```

See [`DATA.md`](DATA.md) for their structure.

## 8. Compute VGGT overlap

```bash
python src/overlap/compute_vggt_overlap.py
```

The generated files are written under:

```text
overlaps/vggt/
```

The final pair value is the minimum of the two directional projection overlaps.

## 9. Compute the 2D overlap

```bash
python src/overlap/compute_2d_overlap.py
```

The generated files are written under:

```text
overlaps/2d/
```

## 10. Analyze overlap values

Build the city-level caches with:

```bash
python src/analysis/read_overlaps.py
```

Generate the global city summary with:

```bash
python src/analysis/global_summary.py
```

Additional utilities are available under:

```text
src/analysis/
```

## 11. Train the main VPR configurations

With the default experiment variables:

```text
VAL_CITIES=cph,sf
EARLY_STOPPING_PATIENCE=10
```

run:

```bash
python src/training/train_baseline.py
python src/training/train_overlap_continuous.py
python src/training/train_overlap_binary.py
python src/training/train_overlap_2d.py
```

The default output directories are:

```text
outputs/baseline/
outputs/continuous/
outputs/binary/
outputs/overlap_2d/
```

## 12. Reproduce the variability experiments

### Experiment 1

Experiment 1 uses:

```text
VAL_CITIES=cph,sf
EARLY_STOPPING_PATIENCE=20
```

Example:

```bash
EARLY_STOPPING_PATIENCE=20 VAL_CITIES=cph,sf OUTPUT_OVERLAP_BINARIO2_ROOT="${PROJECT_ROOT}/outputs/variability/patience_20/binary" python src/training/train_overlap_binary.py
```

### Experiment 2

Experiment 2 uses:

```text
VAL_CITIES=cph
EARLY_STOPPING_PATIENCE=10
```

Example:

```bash
VAL_CITIES=cph EARLY_STOPPING_PATIENCE=10 OUTPUT_OVERLAP_BINARIO2_ROOT="${PROJECT_ROOT}/outputs/variability/copenhagen_only/binary" python src/training/train_overlap_binary.py
```

The complete commands for the four methods are listed in [`EXPERIMENTS.md`](EXPERIMENTS.md).

Separate output directories are used so that variability experiments do not resume from or overwrite the main experiment.

## 13. Training resumption and model selection

Each training script stores:

- CSV and JSON metric histories;
- the best model according to validation Recall@1;
- recent epoch checkpoints.

When a run is restarted with existing epoch checkpoints in the selected output directory, training resumes from the latest saved epoch.

Only the six most recent epoch checkpoints are retained. The selected best model is stored separately.

The checkpoint metadata records the validation-city configuration and early-stopping patience used by the run.

## 14. Validation

Validation is performed independently for every city listed in `VAL_CITIES`.

For each city:

1. database descriptors are computed;
2. query descriptors are computed;
3. queries are matched only against that city's database;
4. Recall@1, Recall@5, and Recall@10 are calculated using local positive indices.

The main experiment uses:

```text
cph,sf
```

The Copenhagen-only variability configuration uses:

```text
cph
```

## 15. Qualitative retrieval inspection

Qualitative retrievals can be generated with:

```bash
python src/evaluation/generate_qualitative_retrievals.py
```

These generated files are not included in the public `results/` directory.

## 16. Docker

Build the image with:

```bash
bash cluster/container/build.sh
```

For an interactive session:

```bash
export MSLS_HOST_ROOT=/path/to/msls
bash cluster/container/run.sh
```

The container uses:

```text
Project:       /workspace/vpr-overlap-training
MSLS:          /data/msls
VGGT:          /opt/vggt
Mapillary SLS: /opt/mapillary_sls
Conda env:     tfg_vpr
```

## 17. SLURM

The full cluster workflow is documented in [`../cluster/README.md`](../cluster/README.md).

The experiment variables can be exported before `sbatch`.

Main experiment:

```bash
cd /path/to/vpr-overlap-training

export MSLS_HOST_ROOT=/path/to/msls
export VAL_CITIES=cph,sf
export EARLY_STOPPING_PATIENCE=10

sbatch cluster/submit_train.sh     src/training/train_overlap_continuous.py
```

Experiment 1:

```bash
export VAL_CITIES=cph,sf
export EARLY_STOPPING_PATIENCE=20
export OUTPUT_OVERLAP2_ROOT="${PWD}/outputs/variability/patience_20/continuous"

sbatch cluster/submit_train.sh     src/training/train_overlap_continuous.py
```

Experiment 2:

```bash
export VAL_CITIES=cph
export EARLY_STOPPING_PATIENCE=10
export OUTPUT_OVERLAP2_ROOT="${PWD}/outputs/variability/copenhagen_only/continuous"

sbatch cluster/submit_train.sh     src/training/train_overlap_continuous.py
```

## 18. Published results

The quantitative artifacts selected for publication are available under:

```text
results/
```

They include:

- main epoch-level metric histories;
- complete histories for Variability Experiment 1;
- available best-model validation metrics for Variability Experiment 2;
- compact summary CSV tables;
- the city-level VGGT overlap summary.

See [`results/README.md`](../results/README.md) for the exact structure and metric conventions.

## 19. Stochasticity

The training scripts do not set one global seed across Python, NumPy, and PyTorch, so independent training runs can follow slightly different trajectories.

The qualitative retrieval script uses a fixed NumPy generator seed for selecting examples.
