#!/bin/bash
set -e

USER_VPR_ROOT=REPO_ROOT
DATA_VPR_LINK=${USER_VPR_ROOT}/data_vpr
MAPILLARY_ROOT=$(readlink -f ${DATA_VPR_LINK}/mapillary)

# Optional interactive test. For real training, use SLURM with ../train.sh.
docker run --gpus all --ipc=host -it \
  -v ${USER_VPR_ROOT}:/workspace/vpr-overlap-training \
  -v ${MAPILLARY_ROOT}:/workspace/vpr-overlap-training/data/mapillary \
  -w /workspace/vpr-overlap-training \
  vpr-overlap-training:latest bash
