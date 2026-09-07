FROM nvcr.io/nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# --------------------------------------------------------
# Basic utilities
# --------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget \
        ca-certificates \
        git \
        curl \
        vim \
        build-essential \
        libglib2.0-0 \
        libgl1 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------
# Miniconda
# --------------------------------------------------------
RUN wget -q \
        https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh

ENV PATH=/opt/conda/bin:$PATH
SHELL ["bash", "-lc"]

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

# --------------------------------------------------------
# Python environment
# The root environment.yml is the single source of truth.
# --------------------------------------------------------
COPY environment.yml /tmp/environment.yml

RUN conda env create -f /tmp/environment.yml \
    && conda clean -afy \
    && conda info --envs

# Make the project environment the default Python environment in the image.
ENV PATH=/opt/conda/envs/tfg_vpr/bin:/opt/conda/bin:$PATH

# --------------------------------------------------------
# VGGT
# --------------------------------------------------------
RUN git clone https://github.com/facebookresearch/vggt.git /opt/vggt

# --------------------------------------------------------
# Mapillary SLS
# --------------------------------------------------------
# This commit matches the source layout used by the compatibility patch.
# If the exact commit used in the thesis experiments is known, replace this
# value with that commit before the final v1.0.0 release.
ARG MAPILLARY_SLS_COMMIT=8f950f59515bfbc69727151dcccb19e0e8a01824

RUN git clone https://github.com/mapillary/mapillary_sls.git /opt/mapillary_sls \
    && git -C /opt/mapillary_sls checkout "${MAPILLARY_SLS_COMMIT}"

COPY patches/mapillary_sls_numpy_object.patch /tmp/mapillary_sls_numpy_object.patch

RUN git -C /opt/mapillary_sls apply --check /tmp/mapillary_sls_numpy_object.patch \
    && git -C /opt/mapillary_sls apply /tmp/mapillary_sls_numpy_object.patch \
    && rm /tmp/mapillary_sls_numpy_object.patch

ENV PYTHONPATH=/opt/vggt:/opt/mapillary_sls:$PYTHONPATH

# --------------------------------------------------------
# Runtime
# --------------------------------------------------------
RUN mkdir -p /workspace/vpr-overlap-training
WORKDIR /workspace/vpr-overlap-training

CMD ["/bin/bash"]
