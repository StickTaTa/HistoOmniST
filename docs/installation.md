# Installation

## Supported environment

The publication inference workflow was exercised with Python 3.10, PyTorch
2.7.0, CUDA 11.8, NumPy 1.26 and pandas 2.3. `environment.yml` creates the
Python environment and installs all public workflow dependencies.

```bash
conda env create -f environment.yml
conda activate histoomnist
```

For a CUDA installation, install the PyTorch build recommended for the local
driver from <https://pytorch.org/get-started/locally/> before installing the
editable package.

## Pretrained HistoOmniST checkpoints

```bash
python scripts/download_release_models.py
```

The downloader reads `models/release_manifest.json`, writes the two files to
their expected checkpoint paths and verifies file size and SHA-256.

## HIPT ViT-256 dependency

HistoOmniST uses the ViT-256 encoder from the original
[`mahmoodlab/HIPT`](https://github.com/mahmoodlab/HIPT) project. It does not
obtain HIPT through iStar. Download the two required assets directly from the
pinned HIPT revision:

```bash
python scripts/download_hipt_assets.py
```

The downloader creates this local layout:

```text
third_party/HIPT/
  1-Hierarchical-Pretraining/vision_transformer.py
  HIPT_4K/Checkpoints/vit256_small_dino.pth
```

Pinned HIPT revision:

```text
780fafaed2e5b112bc1ed6e78852af1fe6714342
```

Expected SHA-256 values:

```text
vision_transformer.py  3958a2b72d5bb7019e27b2ef6422429d484dd4225bcbaf86d803cc144ed9fd52
vit256_small_dino.pth   6960cd5a8657dc8bb214671aa0c6dbd3f5b698e84386884955836487ddc89e24
```

Both URLs, byte counts and checksums are recorded in
`models/release_manifest.json`. The third-party source and weights are excluded
from Git and remain governed by the HIPT project's terms.

## Public inference example

The executable quickstart uses the official 10x Genomics Xenium FFPE Human
Breast Cancer Rep1 H&E image:

```bash
python scripts/download_example_data.py
jupyter lab notebooks/00_quickstart_inference.ipynb
```

The image is 1.43 GB. Its source URL and SHA-256 are recorded in
`examples/breast_xenium/example_manifest.json`; the downloaded file is excluded
from Git.

## Verification

```bash
python -m pytest -q
python scripts/histoomnist_predict_uploaded_wsi.py --help
```
