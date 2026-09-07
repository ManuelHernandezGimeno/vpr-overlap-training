#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

docker build \
    -f "${SCRIPT_DIR}/Dockerfile" \
    -t vpr-overlap-training:latest \
    "${REPO_ROOT}"
