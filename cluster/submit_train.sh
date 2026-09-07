#!/bin/bash

#SBATCH -N 1
#SBATCH --job-name="vpr_train_mhernang"
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
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_VPR_LINK=${USER_VPR_ROOT}/data_vpr
MSLS_HOST_ROOT="${MSLS_HOST_ROOT:?Set MSLS_HOST_ROOT before submitting the job}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-${REPO_ROOT}/cluster/container/vpr-overlap-training+latest.sqsh}"
CONTAINER_PROJECT_ROOT=/workspace/vpr-overlap-training
CONTAINER_MSLS_ROOT=/data/msls

mkdir -p ${USER_VPR_ROOT}/slurm_jobs
mkdir -p ${USER_VPR_ROOT}/data/mapillary

if [ ! -d "${MAPILLARY_ROOT}" ]; then
    echo "ERROR: no se encuentra Mapillary en ${MSLS_HOST_ROOT}"
    echo "Comprueba el enlace: ${DATA_VPR_LINK}"
    exit 1
fi

if [ ! -f "${CONTAINER_IMAGE}" ]; then
    echo "ERROR: no se encuentra la imagen del contenedor en ${CONTAINER_IMAGE}"
    echo "Ejecuta primero docker/build.sh y docker/enroot.sh"
    exit 1
fi


srun \
    --container-mounts="${REPO_ROOT}:${CONTAINER_PROJECT_ROOT},${MSLS_HOST_ROOT}:${CONTAINER_MSLS_ROOT}" \
    --container-workdir="${CONTAINER_PROJECT_ROOT}" \
    --container-image="${CONTAINER_IMAGE}" \
    bash "${CONTAINER_PROJECT_ROOT}/cluster/run_train_job.sh" "$@"
