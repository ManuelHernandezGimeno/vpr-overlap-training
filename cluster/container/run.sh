#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MSLS_HOST_ROOT="${MSLS_HOST_ROOT:?Set MSLS_HOST_ROOT to the MSLS dataset directory}"

CONTAINER_PROJECT_ROOT=/workspace/vpr-overlap-training
CONTAINER_MSLS_ROOT=/data/msls

docker run \
    --gpus all \
    --ipc=host \
    -it \
    -v "${REPO_ROOT}:${CONTAINER_PROJECT_ROOT}" \
    -v "${MSLS_HOST_ROOT}:${CONTAINER_MSLS_ROOT}:ro" \
    -e PROJECT_ROOT="${CONTAINER_PROJECT_ROOT}" \
    -e MSLS_ROOT="${CONTAINER_MSLS_ROOT}" \
    -w "${CONTAINER_PROJECT_ROOT}" \
    vpr-overlap-training:latest \
    bash
