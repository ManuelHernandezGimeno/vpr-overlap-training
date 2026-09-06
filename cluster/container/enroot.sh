#!/bin/bash
set -e

# Run this from /raid/ropert/mhernang/VPR/docker
rm -f vpr_train_mhernang+latest.sqsh

enroot import dockerd://vpr_train_mhernang:latest