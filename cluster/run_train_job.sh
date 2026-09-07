#!/bin/bash
set -e

# Expected workdir from SLURM: /workspace/vpr-overlap-training

source /opt/conda/etc/profile.d/conda.sh
conda activate tfg_vpr

PROJECT_ROOT=/workspace/vpr-overlap-training
cd "${PROJECT_ROOT}"

# Paths used by the Python scripts
export PROJECT_ROOT=/workspace/vpr-overlap-training
export MSLS_ROOT=/data/msls

export POSITIVES_ROOT=${PROJECT_ROOT}/positives
export TRAIN_POSITIVES_ROOT=${PROJECT_ROOT}/positives/train

export OVERLAP_ROOT=${PROJECT_ROOT}/overlaps/vggt
export OVERLAP2D_ROOT=${PROJECT_ROOT}/overlaps/2d

export OUTPUTNEW2_ROOT=${PROJECT_ROOT}/outputs/baseline
export OUTPUT_OVERLAP2_ROOT=${PROJECT_ROOT}/outputs/continuous
export OUTPUT_OVERLAP_BINARIO2_ROOT=${PROJECT_ROOT}/outputs/binary
export OUTPUT_OVERLAP_2D2_ROOT=${PROJECT_ROOT}/outputs/overlap_2d

export HF_HOME=${PROJECT_ROOT}/.cache/huggingface
export TORCH_HOME=${PROJECT_ROOT}/.cache/torch

export VGGT_ROOT=/opt/vggt
export MAPILLARY_SLS_ROOT=/opt/mapillary_sls

export PYTHONPATH=/opt/vggt:/opt/mapillary_sls:${PROJECT_ROOT}/src:${PROJECT_ROOT}/src/analysis:${PROJECT_ROOT}:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1

mkdir -p \
    slurm_jobs \
    outputs/baseline \
    outputs/continuous \
    outputs/binary \
    outputs/overlap_2d \
    positives/train \
    positives/val \
    overlaps/vggt \
    overlaps/2d \
    overlap_cache \
    overlap_cache/plots \
    .cache/huggingface \
    .cache/torch


# Script to run. Default can be changed, or passed from sbatch:
# sbatch cluster/submit_train.sh src/training/train_overlap_continuous.py
if [ "$#" -gt 0 ]; then
    TRAIN_SCRIPT="$1"
    shift
else
    TRAIN_SCRIPT="src/training/train_baseline.py"
fi

if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "ERROR: training script not found: $TRAIN_SCRIPT"
    echo "Current folder: $(pwd)"
    echo "Available files in code/:"
    ls -lah src || true
    exit 1
fi

# Save a copy of the executed script for reproducibility
cp "$TRAIN_SCRIPT" "slurm_jobs/$(basename "$TRAIN_SCRIPT" .py)_${SLURM_JOB_ID}.py" || true

# Quick environment check
python --version
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu:', torch.cuda.get_device_name(0))
PY

# -u avoids buffering, so prints appear in the SLURM log while the job runs
python -u "$TRAIN_SCRIPT" "$@"
