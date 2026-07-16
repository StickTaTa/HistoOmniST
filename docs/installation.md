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

HistoOmniST uses the HIPT ViT-256 implementation distributed with iStar. Clone
the pinned upstream source and download the weight from its upstream mirror:

```bash
git clone https://github.com/daviddaiweizhang/istar.git third_party/benchmarks/iStar
hf download JWonderLand/HIPT_unofficial vit256_small_dino.pth \
  --local-dir third_party/benchmarks/iStar/checkpoints
```

Expected SHA-256:

```text
6960cd5a8657dc8bb214671aa0c6dbd3f5b698e84386884955836487ddc89e24
```

The third-party source and weights are intentionally excluded from Git.

## Verification

```bash
python -m pytest -q
python scripts/histoomnist_predict_uploaded_wsi.py --help
```
