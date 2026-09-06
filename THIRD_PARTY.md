# Third-party software, models and data

This repository contains original project code but relies on external research software, pretrained models and datasets.

## VGGT

**Visual Geometry Grounded Transformer (VGGT)** is developed by Meta AI and the Visual Geometry Group, University of Oxford.

Official repository:

https://github.com/facebookresearch/vggt

This repository does **not** redistribute the VGGT source code or model checkpoints.

Users must review and comply with the VGGT license that applies to the code and checkpoint they use. In particular, different VGGT checkpoints may have different usage conditions.

If VGGT is used, please cite the original VGGT publication.

## Mapillary Street-Level Sequences (MSLS)

Official code repository:

https://github.com/mapillary/mapillary_sls

The MSLS dataset is not redistributed here.

Users must obtain the dataset from the official source and comply with the applicable dataset terms.

## PyTorch / torchvision

The VPR model uses PyTorch and torchvision, including a ResNet50 backbone.

Official project:

https://pytorch.org/

## Research methods referenced by this project

The repository implementation is also informed by research on:

- Visual Place Recognition evaluation and retrieval;
- ResNet;
- Generalized Mean (GeM) pooling;
- Triplet Loss / FaceNet-style metric learning;
- 2D visual-overlap-based VPR training.

These papers should be cited in the associated thesis or publication when their methods are discussed.

## Redistribution policy

Third-party code should not be copied into this repository unless its license explicitly permits redistribution and all required notices are preserved.

Where possible, external projects should remain external dependencies and be installed from their official repositories.
