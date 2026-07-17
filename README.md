# HistoOmniST

[![CI](https://github.com/StickTaTa/HistoOmniST/actions/workflows/ci.yml/badge.svg)](https://github.com/StickTaTa/HistoOmniST/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/StickTaTa/HistoOmniST)](https://github.com/StickTaTa/HistoOmniST/releases/latest)
[![Python](https://img.shields.io/badge/python-3.10-blue)](environment.yml)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-green)](LICENSE)

**HistoOmniST predicts spatial gene expression from H&E histology and
reconstructs count-scale virtual spatial transcriptomes.** It predicts
expression rate and a mean-one, slide-normalized size factor (SF) separately,
then recombines them at each spot or tissue tile.

```text
predicted count[i, g] = predicted rate[i, g] x predicted SF[i]
mean(predicted SF within each slide) = 1
```

![Overview of the HistoOmniST workflow](docs/assets/histoomnist_workflow.png)

## Quick start

**Start with the fully executed
[`00_quickstart_inference.ipynb`](notebooks/00_quickstart_inference.ipynb).**
It is the single public tutorial and shows the input image, tissue mask, tile
grid, intermediate dimensions, predicted SF, reconstructed genes and molecular
programs from one real breast cancer H&E section.

### 1. Install HistoOmniST

```bash
git clone https://github.com/StickTaTa/HistoOmniST.git
cd HistoOmniST
conda env create -f environment.yml
conda activate histoomnist
```

The publication environment uses Python 3.10, PyTorch 2.7.0, CUDA 11.8,
NumPy 1.26 and pandas 2.3. CPU inference is supported but is slow for large
whole-slide images.

### 2. Run the public breast cancer example

```bash
jupyter lab notebooks/00_quickstart_inference.ipynb
```

The notebook downloads and verifies all required public assets: the two frozen
HistoOmniST checkpoints, the original HIPT ViT-256 source and weight, and the
1.43 GB 10x Genomics Xenium FFPE Human Breast Cancer Rep1 H&E image. A CUDA GPU
is strongly recommended for the approximately 12,000-tile example.

The committed execution provides a reference run:

| Item | Reference output |
| --- | ---: |
| Full-resolution image | 30,786 x 24,241 pixels |
| Tissue tiles | 11,831 |
| Effective stride | 192 pixels |
| HIPT feature matrix | 11,831 x 384 |
| Context feature matrix | 11,831 x 1,161 |
| Predicted genes | 28 |
| Mean predicted SF | 1.00000000 |

Downloaded data and generated predictions remain under ignored `data/` and
`outputs/` directories and are not committed.

### 3. Predict another H&E image

The download commands are idempotent and verify file size and SHA-256:

```bash
python scripts/download_release_models.py
python scripts/download_hipt_assets.py
```

Run inference on an OpenSlide-compatible whole-slide image, tiled TIFF or
ordinary RGB image:

```bash
python scripts/histoomnist_predict_uploaded_wsi.py \
  --input /path/to/slide.svs \
  --out-dir outputs/example \
  --write-report \
  --write-zip
```

For cohorts, repeat `--input` or use `--input-list`. The input-list CSV must
contain one of `path`, `input_path`, `wsi_path` or `local_path`. See
[Inference](docs/inference.md) for the complete schema and tiling controls.

## Understanding the outputs

Each inference run writes per-slide tile predictions, a combined
`tile_predictions.csv`, `slide_features.csv`, thumbnails, optional spatial maps
and `run_summary.json`.

| Column | Meaning |
| --- | --- |
| `rate_log1p_GENE` | Frozen expression branch output, `log1p(rate)` |
| `pred_log_sf` | Log of predicted SF after mean-one slide normalization |
| `pred_sf` | Predicted SF normalized to mean one within the slide |
| `count_GENE` | `max(expm1(rate_log1p_GENE), 0) x pred_sf` |
| `count_log1p_GENE` | Canonical count-scale gene value used downstream |
| `gene_GENE` | Website-compatible alias of `count_log1p_GENE` |
| `program_NAME` | Mean of available count-log1p genes in a prespecified program |

The default inference panel contains 28 prespecified genes and eight programs:
epithelial, T cell, myeloid, stromal, proliferation, hypoxia, EMT/TGF-beta and
Wnt/CRC. Their exact definitions are in
[`configs/manuscript_release_28_gene_panel.json`](configs/manuscript_release_28_gene_panel.json).
Other genes can be requested with `--selected-genes` when present in the
16,942-gene expression checkpoint.

## At a glance

| Component | Public release |
| --- | --- |
| Input | H&E whole-slide image, tiled TIFF or ordinary RGB image |
| Image representation | HIPT ViT-256 features with spatial context |
| Model outputs | `log1p(rate)` and `log(SF)` |
| Reconstruction | `log1p(max(expm1(rate_log1p), 0) x mean-one SF)` |
| Default inference panel | 28 genes and eight molecular programs |
| Expression checkpoint | 16,942 coverage-95 canonical genes |
| Training resource | HEST-1k human Visium sections |
| Evaluation boundary | Fixed slide-level partitions |
| Deployment | Single images or cohorts supplied through an input list |

HistoOmniST outputs are model-derived virtual spatial signals. They are not
measured spatial transcriptomes, diagnostic annotations or a substitute for
experimental spatial profiling when tissue and resources are available.

## Method reproducibility

The repository keeps the model architecture, data preparation, training,
evaluation, benchmark adapters, fixed configurations and split manifests needed
to inspect and reproduce the method. Large datasets, processed arrays and
third-party prediction bundles are intentionally not redistributed.

| Workflow | Public material | Automated validation | Additional requirement |
| --- | --- | --- | --- |
| Quick Start inference | Executed notebook, model downloader and WSI predictor | Real example executed; core outputs and CLI checked in CI | Public assets downloaded automatically |
| Rate and SF training | Source, fixed configs and training entry points | CLI import/help and core SF/count-scale tests | HEST raw/processed arrays and a GPU |
| Combined evaluation | Rate-only, predicted-SF and oracle-SF evaluator | CLI import/help and count-scale unit tests | Held-out HEST arrays and both checkpoints |
| Ten-method benchmark | Fixed splits, source-faithful adapters and common evaluator | Public CLI entry points checked in CI | Upstream repositories, checkpoints and standardized prediction bundles |

"Checked in CI" does not mean that manuscript-scale training or all third-party
methods run inside GitHub Actions. Those workflows are too data- and
compute-intensive for CI and require assets that cannot be redistributed.

### Rate and SF training

The public workflow has six preparation stages: metadata audit, HEST asset
download, conversion to aligned spot arrays, HIPT feature extraction, processed
manifest construction and fixed slide-level split generation. It then trains
the two branches with these committed configurations:

| Branch | Configuration | Target |
| --- | --- | --- |
| Expression rate | `configs/hest1k_human_visium_expression_highconf_symbol95.yaml` | `log1p(rate)` for 16,942 genes |
| Size factor | `configs/hest1k_human_visium_sf_context_distribution_light.yaml` | Mean-one slide-normalized `log(SF)` |

Use the detailed, ordered commands and data contract in
[HEST-1k preparation and training](docs/training.md). The public entry points
can be inspected without starting a job:

```bash
python scripts/train_expression.py --help
python scripts/train_sf.py --help
python scripts/evaluate_combined.py --help
```

### Ten-method benchmark

The manuscript benchmark covers HistoOmniST, HiST, Hist2ST, HisToGene, iStar,
mclSTExp, Path2Space, sCellST, STimage, ST-Net and THItoGene. This repository
contains project-owned adapters, provenance records, fixed HEST splits and the
common evaluator. It does not redistribute complete third-party repositories,
their checkpoints or per-spot prediction bundles.

```bash
python scripts/hest_evaluate_benchmark_predictions.py --help
python scripts/hest_eval_histoomnist_benchmark.py --help
```

See [Benchmark workflow](docs/benchmark.md) and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Smoke and reduced-gene runs
must not be reported as formal benchmark results.

## Documentation

| Topic | Guide |
| --- | --- |
| Environment and HIPT setup | [Installation](docs/installation.md) |
| Whole-slide and cohort inference | [Inference](docs/inference.md) |
| HEST-1k preparation and model training | [Training](docs/training.md) |
| Ten-method evaluation | [Benchmark workflow](docs/benchmark.md) |
| Frozen model files | [Model release v0.1.0](https://github.com/StickTaTa/HistoOmniST/releases/tag/v0.1.0) |

## Repository layout

```text
configs/              fixed model, split, gene-panel and plotting definitions
data/HEST-1k/         public metadata, gene lists and slide-level split manifests
docs/                 installation, inference, training and benchmark guides
models/               model-asset manifest and download documentation
notebooks/            one fully executed inference tutorial
examples/             manifests and instructions for public example data
scripts/              preparation, training, evaluation and WSI entry points
src/histoomnist/      installable HistoOmniST package
tests/                release-contract and core workflow tests
```

## Data and release boundaries

The software package is version `0.1.1`. The unchanged frozen checkpoints remain
attached to model release `v0.1.0`; their byte counts and SHA-256 values are
recorded in [`models/release_manifest.json`](models/release_manifest.json).

The Git repository contains code, configurations, fixed metadata/splits,
documentation and the executed inference tutorial. It intentionally excludes
raw or processed HEST data, TCGA and external validation data, whole-slide
images, full prediction bundles, third-party repositories, website uploads and
manuscript Source Data. Obtain these resources from their original providers
and follow their access and licensing terms.

Generated files belong in ignored `results/`, `outputs/` or local data
directories.

## Citation

If you use HistoOmniST, cite the software and accompanying manuscript described
in [`CITATION.cff`](CITATION.cff):

> Huang J, Zheng L. HistoOmniST maps count-scale virtual spatial transcriptomes
> from histology to identify breast cancer immune-proliferation ecology. 2026.

## License

HistoOmniST project-owned code is released under the
[BSD 3-Clause License](LICENSE). Third-party methods, datasets and model assets
remain governed by their original licenses.
