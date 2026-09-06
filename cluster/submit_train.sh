#!/bin/bash

#SBATCH -N 1
#SBATCH --job-name="vpr_train_mhernang"
#SBATCH --ntasks=1
#SBATCH --output=/raid/ropert/mhernang/VPR/slurm_jobs/slurm_train_%j.out
#SBATCH --error=/raid/ropert/mhernang/VPR/slurm_jobs/slurm_train_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=disable-binding

set -e

# Root folders in the DGX host
USER_VPR_ROOT=/raid/ropert/mhernang/VPR
DATA_VPR_LINK=${USER_VPR_ROOT}/data_vpr
MAPILLARY_ROOT=$(readlink -f ${DATA_VPR_LINK}/mapillary)
CONTAINER_IMAGE=${USER_VPR_ROOT}/docker/vpr_train_mhernang+latest.sqsh

mkdir -p ${USER_VPR_ROOT}/slurm_jobs
mkdir -p ${USER_VPR_ROOT}/data/mapillary

if [ ! -d "${MAPILLARY_ROOT}" ]; then
    echo "ERROR: no se encuentra Mapillary en ${MAPILLARY_ROOT}"
    echo "Comprueba el enlace: ${DATA_VPR_LINK}"
    exit 1
fi

if [ ! -f "${CONTAINER_IMAGE}" ]; then
    echo "ERROR: no se encuentra la imagen del contenedor en ${CONTAINER_IMAGE}"
    echo "Ejecuta primero docker/build.sh y docker/enroot.sh"
    exit 1
fi


srun \
--container-mounts=${USER_VPR_ROOT}:/workspace/mhernang/VPR,${MAPILLARY_ROOT}:/workspace/mhernang/VPR/data/mapillary \
--container-workdir=/workspace/mhernang/VPR \
--container-image=${CONTAINER_IMAGE} \
bash /workspace/mhernang/VPR/train_job.sh "$@"