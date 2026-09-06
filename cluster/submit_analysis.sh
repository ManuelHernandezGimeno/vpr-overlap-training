#!/bin/bash

#SBATCH -N 1
#SBATCH --job-name="overlap_analysis"
#SBATCH --ntasks=1
#SBATCH --output=/raid/ropert/mhernang/VPR/slurm_jobs/slurm_overlap_analysis_%j.out
#SBATCH --error=/raid/ropert/mhernang/VPR/slurm_jobs/slurm_overlap_analysis_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

# Root folders in the DGX host
USER_VPR_ROOT=/raid/ropert/mhernang/VPR
CONTAINER_IMAGE=${USER_VPR_ROOT}/docker/vpr_train_mhernang+latest.sqsh

mkdir -p ${USER_VPR_ROOT}/slurm_jobs

if [ ! -f "${CONTAINER_IMAGE}" ]; then
    echo "ERROR: no se encuentra la imagen:"
    echo "${CONTAINER_IMAGE}"
    exit 1
fi

srun \
    --container-mounts=${USER_VPR_ROOT}:/workspace/mhernang/VPR \
    --container-workdir=/workspace/mhernang/VPR \
    --container-image=${CONTAINER_IMAGE} \
    bash /workspace/mhernang/VPR/analysis_job.sh "$@"