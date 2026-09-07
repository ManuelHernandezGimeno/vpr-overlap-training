#!/bin/bash

#SBATCH -N 1
#SBATCH --job-name="overlap_analysis"
#SBATCH --ntasks=1
#SBATCH --output=slurm_overlap_analysis_%j.out
#SBATCH --error=slurm_overlap_analysis_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

# Root folders in the DGX host
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONTAINER_PROJECT_ROOT=/workspace/vpr-overlap-training

CONTAINER_IMAGE="${CONTAINER_IMAGE:-${REPO_ROOT}/cluster/container/vpr-overlap-training+latest.sqsh}"

mkdir -p "${REPO_ROOT}/slurm_jobs"

if [ ! -f "${CONTAINER_IMAGE}" ]; then
    echo "ERROR: no se encuentra la imagen:"
    echo "${CONTAINER_IMAGE}"
    exit 1
fi

srun \
    --container-mounts="${REPO_ROOT}:${CONTAINER_PROJECT_ROOT}" \
    --container-workdir="${CONTAINER_PROJECT_ROOT}" \
    --container-image="${CONTAINER_IMAGE}" \
    bash "${CONTAINER_PROJECT_ROOT}/cluster/run_analysis_job.sh" "$@"
