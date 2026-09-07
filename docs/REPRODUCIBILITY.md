# Reproducibility

## 1. Clone

```bash
git clone https://github.com/ManuelHernandezGimeno/vpr-overlap-training.git
cd vpr-overlap-training
```

## 2. Environment

```bash
conda env create -f environment.yml
conda activate vpr-overlap
```

The final `environment.yml` should list all direct dependencies, including `scikit-learn` and `shapely`.

## 3. External repositories

```bash
git clone https://github.com/mapillary/mapillary_sls.git /path/to/mapillary_sls
git clone https://github.com/facebookresearch/vggt.git /path/to/vggt
```

For the final release, document and pin the exact commits used for the thesis experiments.

## 4. Path configuration

Recommended local variables:

```bash
export PROJECT_ROOT=/path/to/vpr-overlap-training
export MSLS_ROOT=/path/to/msls
export MAPILLARY_SLS_ROOT=/path/to/mapillary_sls
export VGGT_ROOT=/path/to/vggt

export POSITIVES_ROOT=${PROJECT_ROOT}/positives
export TRAIN_POSITIVES_ROO=${POSITIVES_ROOT}/train

export OVERLAP_ROOT=${PROJECT_ROOT}/overlaps/vggt
export OVERLAP2D_ROOT=${PROJECT_ROOT}/overlaps/2d

export CACHE_DIR=${PROJECT_ROOT}/overlap_cache
export PLOTS_DIR=${CACHE_DIR}/plots

export OUTPUTNEW2_ROOT=${PROJECT_ROOT}/outputs/baseline
export OUTPUT_OVERLAP2_ROOT=${PROJECT_ROOT}/outputs/continuous
export OUTPUT_OVERLAP_BINARIO2_ROOT=${PROJECT_ROOT}/outputs/binary
export OUTPUT_OVERLAP_2D2_ROOT=${PROJECT_ROOT}/outputs/overlap_2d
```

The historical environment-variable names above are kept for compatibility with the current Python scripts.

## 5. Execution order

```bash
python src/data/generate_msls_positives.py
python src/overlap/compute_vggt_overlap.py
python src/overlap/compute_2d_overlap.py
python src/analysis/read_overlaps.py
python src/analysis/global_summary.py

python src/training/train_baseline.py
python src/training/train_overlap_continuous.py
python src/training/train_overlap_binary.py
python src/training/train_overlap_2d.py

python src/evaluation/generate_qualitative_retrievals.py
```

## 6. Training resumption

Training scripts save metric histories, the best model, and recent epoch checkpoints.

When a run restarts, the latest epoch checkpoint is loaded and training continues from the next epoch.

The current scripts retain only the most recent epoch checkpoints to limit storage use.

## 7. City-wise validation

Validation must remain city-wise so that local database indices from different MSLS cities are never mixed.

## 8. Docker

The public container should use:

```text
/workspace/vpr-overlap-training
```

as the project working directory.

Recommended dataset mount:

```text
/data/msls
```

External repositories:

```text
/opt/vggt
/opt/mapillary_sls
```

The Docker build must use the repository root as build context because `environment.yml` is stored there.

## 9. SLURM

The public SLURM launchers should mount:

```text
<host-repository-root> -> /workspace/vpr-overlap-training
<host-msls-root>       -> /data/msls
```

Derive the repository root from the script location:

```bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
```

Provide the dataset path explicitly:

```bash
export MSLS_HOST_ROOT=/path/to/msls
```

## 10. Reproducibility limitations

GPU training contains stochastic components.

If the final thesis experiments did not use a fixed global random seed, document that fact rather than assigning one retroactively.

The qualitative retrieval script does use a fixed NumPy generator seed for selecting examples.

## 11. Final-release checklist

- [ ] remove personal absolute paths;
- [ ] align Docker and SLURM with the public repository layout;
- [ ] verify the VGGT images-per-block value;
- [ ] verify training parameters against the final thesis;
- [ ] list all direct dependencies;
- [ ] translate comments and console messages to English;
- [ ] avoid silently modifying third-party source code at runtime;
- [ ] publish lightweight metrics under `results/`;
- [ ] verify all README links;
- [ ] run smoke tests for the major pipeline stages.
