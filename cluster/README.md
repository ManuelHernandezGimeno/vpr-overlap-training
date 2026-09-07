# Cluster execution

This directory contains the Docker, Enroot, and SLURM scripts used for the large-scale experiments on an NVIDIA DGX-1 system.

## Directory structure

```text
cluster/
├── README.md
├── submit_train.sh
├── run_train_job.sh
├── submit_analysis.sh
├── run_analysis_job.sh
└── container/
    ├── Dockerfile
    ├── build.sh
    ├── enroot.sh
    └── run.sh
```

## Container paths

```text
Project:       /workspace/vpr-overlap-training
MSLS dataset:  /data/msls
VGGT:          /opt/vggt
Mapillary SLS: /opt/mapillary_sls
Conda env:     tfg_vpr
```

## Build the container

```bash
bash cluster/container/build.sh
bash cluster/container/enroot.sh
```

The Enroot image is created as:

```text
cluster/container/vpr-overlap-training+latest.sqsh
```

## Interactive execution

```bash
export MSLS_HOST_ROOT=/path/to/msls
bash cluster/container/run.sh
```

## Submit the main experiment

Submit from the repository root:

```bash
cd /path/to/vpr-overlap-training

export MSLS_HOST_ROOT=/path/to/msls
export VAL_CITIES=cph,sf
export EARLY_STOPPING_PATIENCE=10
```

Example:

```bash
sbatch cluster/submit_train.sh     src/training/train_overlap_continuous.py
```

The training scripts read `VAL_CITIES` and `EARLY_STOPPING_PATIENCE` directly from the environment.

## Experiment 1 — patience 20

Example for binary overlap:

```bash
export VAL_CITIES=cph,sf
export EARLY_STOPPING_PATIENCE=20
export OUTPUT_OVERLAP_BINARIO2_ROOT="${PWD}/outputs/variability/patience_20/binary"

sbatch cluster/submit_train.sh     src/training/train_overlap_binary.py
```

Use the corresponding output-root variable for the other methods:

```text
Baseline:    OUTPUTNEW2_ROOT
Continuous:  OUTPUT_OVERLAP2_ROOT
Binary:      OUTPUT_OVERLAP_BINARIO2_ROOT
2D:          OUTPUT_OVERLAP_2D2_ROOT
```

## Experiment 2 — Copenhagen-only model selection

Example for binary overlap:

```bash
export VAL_CITIES=cph
export EARLY_STOPPING_PATIENCE=10
export OUTPUT_OVERLAP_BINARIO2_ROOT="${PWD}/outputs/variability/copenhagen_only/binary"

sbatch cluster/submit_train.sh     src/training/train_overlap_binary.py
```

Using separate output directories prevents a variability run from resuming from or overwriting the main experiment.

## Analysis jobs

```bash
sbatch cluster/submit_analysis.sh     src/analysis/read_overlaps.py
```

## Training-job resources

`submit_train.sh` requests:

```text
Nodes:           1
Tasks:           1
CPUs per task:   16
Memory:          96 GB
GPUs:            1
```

## Analysis-job resources

`submit_analysis.sh` requests:

```text
Nodes:           1
Tasks:           1
CPUs per task:   8
Memory:          32 GB
```

## Generated files

Cluster jobs write generated artifacts under:

```text
outputs/
positives/
overlaps/
overlap_cache/
slurm_jobs/
.cache/
```

These generated directories are excluded from Git.

The lightweight quantitative results selected for publication are stored separately under:

```text
results/
```

## Training resumption

The training scripts resume from the latest epoch checkpoint found in the selected output directory.

They retain the six most recent epoch checkpoints and store the best model according to validation Recall@1 separately.
