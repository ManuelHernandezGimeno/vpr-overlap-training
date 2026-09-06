#!/bin/bash
set -e

USER_VPR_ROOT=/raid/ropert/mhernang/VPR
DATA_VPR_LINK=${USER_VPR_ROOT}/data_vpr
MAPILLARY_ROOT=$(readlink -f ${DATA_VPR_LINK}/mapillary)

# Optional interactive test. For real training, use SLURM with ../train.sh.
docker run --gpus all --ipc=host -it \
  -v ${USER_VPR_ROOT}:/workspace/mhernang/VPR \
  -v ${MAPILLARY_ROOT}:/workspace/mhernang/VPR/data/mapillary \
  -w /workspace/mhernang/VPR \
  vpr_train_mhernang:latest bash
