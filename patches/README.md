# Mapillary SLS compatibility patch

The public training and evaluation scripts do **not** modify third-party source code at runtime.

With NumPy 1.26, the original Mapillary SLS implementation can fail when converting
`pIdx` and `nonNegIdx` to NumPy arrays because these lists contain variable-length
arrays. The compatibility patch in this directory changes only these two conversions:

```python
self.pIdx = np.asarray(self.pIdx, dtype=object)
self.nonNegIdx = np.asarray(self.nonNegIdx, dtype=object)
```

The patch is based on the current Mapillary SLS `main` source layout and is applied
explicitly during installation or container creation.

For a local installation:

```bash
export MAPILLARY_SLS_ROOT=/path/to/mapillary_sls
bash scripts/apply_msls_patch.sh
```

The script is idempotent: if the patch is already applied, it exits without modifying
the repository again.

The Docker image applies the same patch explicitly during the image build.
