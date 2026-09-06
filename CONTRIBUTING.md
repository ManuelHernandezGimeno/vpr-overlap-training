# Contributing

This repository was created as research code for a Bachelor's Thesis, but bug reports and improvements are welcome.

## Before opening an issue

Please include:

- the script being executed,
- Python and PyTorch versions,
- GPU/CUDA information when relevant,
- the dataset split/city involved,
- the full error message,
- enough configuration information to reproduce the problem.

Do not upload MSLS images, private cluster paths, credentials or restricted model files.

## Pull requests

Pull requests should:

- keep code and comments in English,
- avoid adding large generated files,
- preserve the documented training definitions,
- keep baseline, continuous, binary and 2D experiments comparable,
- update documentation when behavior changes.

## Code style

Prefer:

- descriptive English variable names,
- `pathlib.Path` for filesystem paths,
- environment variables for machine-specific locations,
- explicit random seeds where reproducibility matters,
- small reusable functions instead of duplicated training/evaluation logic.
