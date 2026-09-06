#!/bin/bash
set -euo pipefail

# Expected workdir from SLURM: /workspace/mhernang/VPR

source /opt/conda/etc/profile.d/conda.sh
conda activate tfg_vpr

cd /workspace/mhernang/VPR

# Paths used by the Python scripts
export PROJECT_ROOT=/workspace/mhernang/VPR
export MSLS_ROOT=/workspace/mhernang/VPR/data/mapillary
export POSITIVES_ROOT=/workspace/mhernang/VPR/positives
export POSITIVESTRAIN_ROOT=/workspace/mhernang/VPR/positives/train
export OVERLAP_ROOT=/workspace/mhernang/VPR/overlaps/VGGT_Overlap
export CACHE_DIR=/workspace/mhernang/VPR/Overlap_cache
export PLOTS_DIR=/workspace/mhernang/VPR/Overlap_cache/plots
export OVERLAP2D_ROOT=/workspace/mhernang/VPR/overlaps/Overlap_2D
export HF_HOME=/workspace/mhernang/VPR/.cache/huggingface
export TORCH_HOME=/workspace/mhernang/VPR/.cache/torch
export VGGT_ROOT=/opt/vggt
export MAPILLARY_SLS_ROOT=/opt/mapillary_sls
export PYTHONPATH=/opt/vggt:/opt/mapillary_sls:/workspace/mhernang/VPR/code/Analisis_overlaps:/workspace/mhernang/VPR/code:/workspace/mhernang/VPR:${PYTHONPATH:-}
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
    ANALYSIS_SCRIPT="code/Analisis_overlaps/Leer_overlaps.py"
fi

if [ ! -f "${ANALYSIS_SCRIPT}" ]; then
    echo "ERROR: no se encuentra el script: ${ANALYSIS_SCRIPT}"
    echo "Archivos disponibles:"
    ls -lah code || true
    exit 1
fi

if [ ! -f "code/Analisis_overlaps/overlap_io.py" ]; then
    echo "ERROR: falta code/Analisis_overlaps/overlap_io.py"
    exit 1
fi

# Save a copy of the executed script for reproducibility
cp "${ANALYSIS_SCRIPT}" \
    "slurm_jobs/$(basename "${ANALYSIS_SCRIPT}" .py)_${SLURM_JOB_ID}.py" \
    || true

cp "code/Analisis_overlaps/overlap_io.py" \
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
