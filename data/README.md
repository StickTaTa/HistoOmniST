# Data boundary

This repository contains public HEST metadata, fixed split manifests and gene
lists needed to define the manuscript tasks. It does not redistribute raw or
processed spatial transcriptomics data, whole-slide images, TCGA files,
external validation datasets or model predictions.

Use `scripts/hest_download_assets.py` to obtain HEST assets from their original
source. TCGA and external resources must be obtained from the repositories and
under the access terms described in the manuscript Data availability section.

Generated data belong in ignored `data/*/raw`, `data/*/processed`, `results`
or `outputs` directories.
