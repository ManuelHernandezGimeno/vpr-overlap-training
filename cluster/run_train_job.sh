#!/bin/bash
set -e

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
export OUTPUT_ROOT=/workspace/mhernang/VPR/outputs/Checkpoints
export OUTPUTNEW_ROOT=/workspace/mhernang/VPR/outputs/Checkpoints_new
export OUTPUTNEW2_ROOT=/workspace/mhernang/VPR/outputs/Resultados
export OUTPUT_OVERLAP_ROOT=/workspace/mhernang/VPR/outputs/Checkpoints_O
export OUTPUT_OVERLAP2_ROOT=/workspace/mhernang/VPR/outputs/Resultados_O
export OUTPUT_OVERLAPNEW_ROOT=/workspace/mhernang/VPR/outputs/Checkpoints_O_new
export OUTPUT_OVERLAP_BINARIO_ROOT=/workspace/mhernang/VPR/outputs/Checkpoints_O_binario
export OUTPUT_OVERLAP_BINARIO2_ROOT=/workspace/mhernang/VPR/outputs/Resultados_O_binario
export OUTPUT_OVERLAP_2D_ROOT=/workspace/mhernang/VPR/outputs/Checkpoints_O_2D
export OUTPUT_OVERLAP_2D2_ROOT=/workspace/mhernang/VPR/outputs/Resultados_O_2D
export HF_HOME=/workspace/mhernang/VPR/.cache/huggingface
export TORCH_HOME=/workspace/mhernang/VPR/.cache/torch
export VGGT_ROOT=/opt/vggt
export MAPILLARY_SLS_ROOT=/opt/mapillary_sls
export PYTHONPATH=/opt/vggt:/opt/mapillary_sls:/workspace/mhernang/VPR/code:/workspace/mhernang/VPR:${PYTHONPATH}
export PYTHONUNBUFFERED=1

mkdir -p slurm_jobs outputs positives/train positives/val overlaps/VGGT_Overlap overlaps/Overlap_2D data .cache/huggingface .cache/torch

# Script to run. Default can be changed, or passed from sbatch:
# sbatch train.sh code/train_overlap_continuo.py
if [ "$#" -gt 0 ]; then
    TRAIN_SCRIPT="$1"
    shift
else
    TRAIN_SCRIPT="code/calcular_positivos_msls.py"
fi

if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "ERROR: training script not found: $TRAIN_SCRIPT"
    echo "Current folder: $(pwd)"
    echo "Available files in code/:"
    ls -lah code || true
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