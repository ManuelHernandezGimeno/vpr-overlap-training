# Reproducibility

## 1. Clone this repository

```bash
git clone https://github.com/ManuelHernandezGimeno/vpr-overlap-training.git
cd vpr-overlap-training
```

## 2. Create the Conda environment

The repository uses a single root environment definition:

```bash
conda env create -f environment.yml
conda activate tfg_vpr
```

The environment contains the dependencies used by the training, overlap, analysis,
and qualitative-evaluation scripts, including PyTorch 2.3.1, torchvision 0.18.1,
NumPy 1.26.4, scikit-learn, and Shapely.

## 3. Clone the external repositories

Clone Mapillary SLS and VGGT outside this repository:

```bash
git clone https://github.com/mapillary/mapillary_sls.git /path/to/mapillary_sls
git clone https://github.com/facebookresearch/vggt.git /path/to/vggt
```

For the final release, the exact external commits used for the thesis experiments
should be recorded whenever they are known.

## 4. Apply the Mapillary SLS compatibility patch

The training and evaluation scripts do not modify Mapillary SLS automatically.

Set the external repository path and explicitly apply the compatibility patch once:

```bash
export MAPILLARY_SLS_ROOT=/path/to/mapillary_sls
bash scripts/apply_msls_patch.sh
```

The patch changes only the conversion of variable-length `pIdx` and `nonNegIdx`
collections so that NumPy stores them as object arrays.

The patch script is idempotent and reports when the patch has already been applied.

## 5. Configure local paths

Create a local configuration file:

```bash
cp config/env.example config/env.sh
```

Edit `config/env.sh` with the paths on your machine and load it:

```bash
source config/env.sh
```

`config/env.sh` should remain untracked because it contains machine-specific paths.

## 6. Recommended execution order

### Generate MSLS positive pairs

```bash
python src/data/generate_msls_positives.py
```

### Compute VGGT geometric overlap

```bash
python src/overlap/compute_vggt_overlap.py
```

### Compute the 2D field-of-view overlap

```bash
python src/overlap/compute_2d_overlap.py
```

### Build overlap-analysis caches

```bash
python src/analysis/read_overlaps.py
python src/analysis/global_summary.py
```

### Train the four VPR configurations

```bash
python src/training/train_baseline.py
python src/training/train_overlap_continuous.py
python src/training/train_overlap_binary.py
python src/training/train_overlap_2d.py
```

### Generate qualitative retrieval examples

```bash
python src/evaluation/generate_qualitative_retrievals.py
```

## 7. Training resumption

The training scripts save metric histories, the best model, and recent epoch
checkpoints. When a run is restarted, the latest epoch checkpoint is loaded and
training continues from the next epoch.

The best model according to validation Recall@1 is stored separately.

## 8. City-wise validation

Validation is performed independently for each MSLS validation city. Database
indices are therefore always interpreted within the city to which they belong.

The reported metrics are:

```text
Recall@1
Recall@5
Recall@10
```

## 9. Docker

The Docker image uses the root `environment.yml` as the Python-environment
definition.

Build it from any directory with:

```bash
bash cluster/container/build.sh
```

The build script uses the repository root as Docker build context. The root
`.dockerignore` prevents datasets, overlap files, outputs, caches, and other large
generated artifacts from being sent to Docker.

The image explicitly applies the same Mapillary SLS compatibility patch during
the image build. The project scripts themselves never edit the external
Mapillary SLS source tree.

The public container paths are:

```text
Project:      /workspace/vpr-overlap-training
MSLS:         /data/msls
VGGT:         /opt/vggt
Mapillary:    /opt/mapillary_sls
Conda env:    tfg_vpr
```

## 10. SLURM

The public launchers mount the repository and the MSLS dataset into the container.

Example:

```bash
cd /path/to/vpr-overlap-training
export MSLS_HOST_ROOT=/path/to/msls

sbatch cluster/submit_train.sh \
    src/training/train_overlap_continuous.py
```

## 11. Reproducibility limitations

GPU training contains stochastic components. If no fixed global random seed was
used for the final thesis experiments, this should be documented rather than
retroactively assigning a seed to the reported results.

The qualitative-retrieval script uses a fixed NumPy generator seed for selecting
examples.
