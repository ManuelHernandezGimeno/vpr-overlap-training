## Installation

### 1. Create the Python environment

The repository uses a Conda environment named `tfg_vpr`:

```bash
conda env create -f environment.yml
conda activate tfg_vpr
```

### 2. Clone the external repositories

Clone Mapillary SLS and VGGT outside this repository:

```bash
git clone https://github.com/mapillary/mapillary_sls.git /path/to/mapillary_sls
git clone https://github.com/facebookresearch/vggt.git /path/to/vggt
```

### 3. Apply the Mapillary SLS compatibility patch

The project scripts do not modify third-party source code at runtime. Apply the
required compatibility patch explicitly once after cloning Mapillary SLS:

```bash
export MAPILLARY_SLS_ROOT=/path/to/mapillary_sls
bash scripts/apply_msls_patch.sh
```

See [`patches/README.md`](patches/README.md) for the exact modification.

### 4. Configure paths

Create a machine-specific configuration from the provided template:

```bash
cp config/env.example config/env.sh
```

Edit `config/env.sh`, then load it:

```bash
source config/env.sh
```

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the complete execution workflow.
