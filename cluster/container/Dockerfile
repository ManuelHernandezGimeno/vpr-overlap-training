FROM nvcr.io/nvidia/cuda:12.1.0-devel-ubuntu22.04

# Avoid interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# Make Python logs appear immediately
ENV PYTHONUNBUFFERED=1

# -------------------------
# Basic utilities
# -------------------------
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
# Install Miniconda
# --------------------------------------------------------
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

ENV PATH=/opt/conda/bin:$PATH

# Use bash for subsequent commands
SHELL ["bash", "-lc"]
# Accept Anaconda TOS to avoid interactive prompt during package installs
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

# --------------------------------------------------------
# Create the conda environment
# --------------------------------------------------------
COPY environment.yml /tmp/environment.yml

RUN bash -lc "conda env create -f /tmp/environment.yml && conda clean -afy && conda info --envs"

# -------------------------
# Install Python dependencies inside tfg_vpr
# -------------------------
RUN bash -lc "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate tfg_vpr && \
    python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121 && \
    python -m pip install \
        numpy==1.26.4 \
        pandas==2.1.3 \
        scipy \
        scikit-learn \
        Pillow \
        tqdm \
        matplotlib \
        h5py \
        pyyaml \
        pyparsing \
        joblib \
        opencv-python-headless==4.9.0.80 \
        termcolor \
        pyproj \
        utm \
        shapely \
        tensorboard \
        einops \
        huggingface-hub \
        safetensors"

# -------------------------
# Install VGGT source code
# -------------------------
RUN git clone https://github.com/facebookresearch/vggt.git /opt/vggt

# -------------------------
# Install Mapillary SLS source code
# -------------------------
RUN git clone https://github.com/mapillary/mapillary_sls.git /opt/mapillary_sls

ENV PYTHONPATH=/opt/vggt:/opt/mapillary_sls:$PYTHONPATH

# --------------------------------------------------------
# Runtime settings
# --------------------------------------------------------
RUN mkdir -p /workspace/vpr-overlap-training
WORKDIR /workspace/vpr-overlap-training

CMD ["/bin/bash"]
