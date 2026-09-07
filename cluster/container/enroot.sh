#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

rm -f vpr-overlap-training+latest.sqsh

enroot import dockerd://vpr-overlap-training:latest
