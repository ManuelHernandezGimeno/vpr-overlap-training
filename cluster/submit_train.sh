#!/bin/bash

#SBATCH -N 1
#SBATCH --job-name="vpr_overlap_train"
#SBATCH --ntasks=1
#SBATCH --output=slurm_train_%j.out
#SBATCH --error=slurm_train_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=disable-binding

set -e

# Root folders in the DGX host
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"

MSLS_HOST_ROOT="${MSLS_HOST_ROOT:?Set MSLS_HOST_ROOT before submitting the job}"

CONTAINER_IMAGE="${CONTAINER_IMAGE:-${REPO_ROOT}/cluster/container/vpr-overlap-training+latest.sqsh}"

CONTAINER_PROJECT_ROOT=/workspace/vpr-overlap-training
CONTAINER_MSLS_ROOT=/data/msls


mkdir -p "${REPO_ROOT}/slurm_jobs"

if [ ! -d "${MSLS_HOST_ROOT}" ]; then
    echo "ERROR: MSLS dataset not found at:"
    echo "${MSLS_HOST_ROOT}"
    exit 1
fi

if [ ! -f "${CONTAINER_IMAGE}" ]; then
    echo "ERROR: no se encuentra la imagen del contenedor en ${CONTAINER_IMAGE}"
    echo "Ejecuta primero docker/build.sh y docker/enroot.sh"
    exit 1
fi

if [ ! -d "${REPO_ROOT}/src" ]; then
    echo "ERROR: repository root not found at ${REPO_ROOT}"
    echo "Submit the job from the repository root or set REPO_ROOT explicitly."
    exit 1
fi

srun \
    --container-mounts="${REPO_ROOT}:${CONTAINER_PROJECT_ROOT},${MSLS_HOST_ROOT}:${CONTAINER_MSLS_ROOT}" \
    --container-workdir="${CONTAINER_PROJECT_ROOT}" \
    --container-image="${CONTAINER_IMAGE}" \
    bash "${CONTAINER_PROJECT_ROOT}/cluster/run_train_job.sh" "$@"
