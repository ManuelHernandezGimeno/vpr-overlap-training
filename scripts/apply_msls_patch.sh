#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MAPILLARY_SLS_ROOT="${MAPILLARY_SLS_ROOT:?Set MAPILLARY_SLS_ROOT to the cloned mapillary_sls repository}"
PATCH_FILE="${REPO_ROOT}/patches/mapillary_sls_numpy_object.patch"

if [ ! -d "${MAPILLARY_SLS_ROOT}/.git" ]; then
    echo "ERROR: MAPILLARY_SLS_ROOT is not a Git repository:"
    echo "  ${MAPILLARY_SLS_ROOT}"
    exit 1
fi

if [ ! -f "${PATCH_FILE}" ]; then
    echo "ERROR: patch file not found:"
    echo "  ${PATCH_FILE}"
    exit 1
fi

# If the reverse patch applies cleanly, the patch is already present.
if git -C "${MAPILLARY_SLS_ROOT}" apply --reverse --check "${PATCH_FILE}" >/dev/null 2>&1; then
    echo "Mapillary SLS compatibility patch is already applied."
    exit 0
fi

# Validate before modifying the external repository.
git -C "${MAPILLARY_SLS_ROOT}" apply --check "${PATCH_FILE}"

git -C "${MAPILLARY_SLS_ROOT}" apply "${PATCH_FILE}"

echo "Mapillary SLS compatibility patch applied successfully."
