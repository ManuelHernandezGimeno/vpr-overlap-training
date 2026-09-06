#!/bin/bash
set -e

# Run this from REPO_ROOT/workspace/vpr-overlap-training/cluster/container
rm -f vpr-overlap-training+latest.sqsh

enroot import dockerd://vpr-overlap-training:latest
