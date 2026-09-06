#!/bin/bash
set -euo pipefail

# Expected workdir from SLURM: /workspace/mhernang/VPR

source /opt/conda/etc/profile.d/conda.sh
conda activate tfg_vpr

cd /workspace/vpr-overlap-training

# Paths used by the Python scripts
export PROJECT_ROOT=/workspace/vpr-overlap-training
export MSLS_ROOT=/data/msls

export POSITIVES_ROOT=${PROJECT_ROOT}/positives
export TRAIN_POSITIVES_ROOT=${PROJECT_ROOT}/positives/train

export OVERLAP_ROOT=${PROJECT_ROOT}/overlaps/vggt
export OVERLAP2D_ROOT=${PROJECT_ROOT}/overlaps/2d

export CACHE_DIR=${PROJECT_ROOT}/overlap_cache
export PLOTS_DIR=${PROJECT_ROOT}/overlap_cache/plots

export HF_HOME=${PROJECT_ROOT}/.cache/huggingface
export TORCH_HOME=${PROJECT_ROOT}/.cache/torch

export VGGT_ROOT=/opt/vggt
export MAPILLARY_SLS_ROOT=/opt/mapillary_sls

export PYTHONPATH=/opt/vggt:/opt/mapillary_sls:${PROJECT_ROOT}/src:${PROJECT_ROOT}/src/analysis:${PROJECT_ROOT}:${PYTHONPATH:-}

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

mkdir -p \
    /workspace/mhernang/VPR/slurm_jobs \
    "${CACHE_DIR}" \
    "${PLOTS_DIR}"

# Script to run. Default can be changed, or passed from sbatch:
# sbatch analysis.sh code/Analisis_overlaps/Leer_overlaps.py
if [ "$#" -gt 0 ]; then
    ANALYSIS_SCRIPT="$1"
    shift
else
    ANALYSIS_SCRIPT="src/analysis/read_overlaps.py"
fi

if [ ! -f "${ANALYSIS_SCRIPT}" ]; then
    echo "ERROR: no se encuentra el script: ${ANALYSIS_SCRIPT}"
    echo "Archivos disponibles:"
    ls -lah src/analysis || true
    exit 1
fi

if [ ! -f "src/analysis/overlap_io.py" ]; then
    echo "ERROR: falta src/analysis/overlap_io.py"
    exit 1
fi

# Save a copy of the executed script for reproducibility
cp "${ANALYSIS_SCRIPT}" \
    "slurm_jobs/$(basename "${ANALYSIS_SCRIPT}" .py)_${SLURM_JOB_ID}.py" \
    || true

cp "src/analysis/overlap_io.py" \
   "slurm_jobs/overlap_io_${SLURM_JOB_ID}.py" \
   || true

echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodo: ${HOSTNAME}"
echo "Script: ${ANALYSIS_SCRIPT}"
echo "OVERLAP_ROOT: ${OVERLAP_ROOT}"
echo "CACHE_DIR: ${CACHE_DIR}"
echo "Inicio: $(date)"
echo "========================================"

# Quick environment check
python --version
python - <<'PY'
import numpy
import pandas
import matplotlib
import tqdm

print("numpy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("matplotlib:", matplotlib.__version__)
print("tqdm:", tqdm.__version__)
PY

# -u avoids buffering, so prints appear in the SLURM log while the job runs
python -u "${ANALYSIS_SCRIPT}" "$@"

echo "========================================"
echo "Fin: $(date)"
echo "========================================"
