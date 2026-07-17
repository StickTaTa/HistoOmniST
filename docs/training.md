# HEST-1k data preparation, training and evaluation

This guide documents the manuscript-scale rate and size-factor workflows. The
CLI entry points and core SF/count-scale functions are checked in CI. Full HEST
training is not run in GitHub Actions because it requires large processed
arrays and GPU compute that are not distributed with the repository.

## Prerequisites

1. Create the environment from `environment.yml`.
2. Copy `configs/local_paths.example.yaml` to the ignored
   `configs/local_paths.yaml` and update machine-specific paths only there.
3. Obtain HEST-1k assets under the terms of the original provider.
4. Install the original HIPT ViT-256 source and weight:

```bash
python scripts/download_hipt_assets.py
```

The public CLI contract can be checked without starting a data or GPU job:

```bash
python scripts/hest_audit_metadata.py --help
python scripts/hest_download_assets.py --help
python scripts/hest_convert_raw_to_processed_arrays.py --help
python scripts/hest_extract_patch_features.py --help
python scripts/hest_build_manifest.py --help
python scripts/hest_make_splits.py --help
python scripts/train_expression.py --help
python scripts/train_sf.py --help
python scripts/evaluate_combined.py --help
```

## Size-factor and rate definitions

For valid spots within one slide:

```text
total[i] = sum_g raw_count[i, g]
SF[i] = total[i] / mean(total[valid spots in the slide])
target[i] = log(SF[i])
rate[i, g] = count[i, g] / SF[i]
```

This is a mean-one, slide-normalized definition. Do not replace it with median
normalization. Count-scale reconstruction reverses the decomposition after
prediction:

```text
predicted count[i, g] = predicted rate[i, g] x predicted SF[i]
```

## Processed slide contract

The preparation pipeline writes one directory per slide:

```text
data/HEST-1k/
  HEST_v1_3_0.csv
  raw/
  processed/<slide_id>/
    features.npy
    counts.npz
    coords.npy
    size_factor.npy
    spots.txt
    genes.txt
  manifests/
  splits/
```

`features.npy`, count rows, coordinates, spot identifiers and genes must remain
aligned. Preparation rejects incompatible arrays before they enter training.

## Ordered preparation workflow

Run these stages from the repository root.

### 1. Audit metadata

```bash
python scripts/hest_audit_metadata.py
```

This checks the configured human Visium reporting universe before any large
download begins.

### 2. Download HEST assets

```bash
python scripts/hest_download_assets.py
```

Use `--sample-id`, `--sample-list` and `--asset` for controlled partial
downloads. Raw assets remain outside Git.

### 3. Convert raw assays to aligned arrays

```bash
python scripts/hest_convert_raw_to_processed_arrays.py
```

The conversion writes counts, coordinates, spot identifiers, gene identifiers
and mean-one `size_factor.npy` for each prepared slide.

### 4. Extract HIPT patch features

```bash
python scripts/hest_extract_patch_features.py
```

This stage uses the pinned original HIPT ViT-256 implementation and weight.
Machine-specific HIPT paths belong in `configs/local_paths.yaml`.

### 5. Build the processed-data manifest

```bash
python scripts/hest_build_manifest.py \
  --config configs/hest1k_human_visium_sf_context_distribution_light.yaml
```

An empty manifest means that the required processed arrays are not present; it
is not a valid input for training.

### 6. Generate fixed slide-level splits

```bash
python scripts/hest_make_splits.py --write-split-manifest
```

Headline results use fixed slide-level train, validation and test partitions.
Do not substitute random spot-level splits.

## Train the two branches

### Expression-rate branch

```bash
python scripts/train_expression.py \
  --config configs/hest1k_human_visium_expression_highconf_symbol95.yaml \
  --device cuda
```

The target is `log1p(rate)` for 16,942 coverage-95 canonical genes.

### Size-factor branch

```bash
python scripts/train_sf.py \
  --config configs/hest1k_human_visium_sf_context_distribution_light.yaml \
  --device cuda
```

The target is slide-normalized `log(SF)` from the 1,161-feature context
representation. Both commands accept `--epochs` and `--output-dir` for
controlled runs; do not overwrite frozen release checkpoints during testing.

## Evaluate count-scale reconstruction

```bash
python scripts/evaluate_combined.py \
  --sf-config configs/hest1k_human_visium_sf_context_distribution_light.yaml \
  --expression-config configs/hest1k_human_visium_expression_highconf_symbol95.yaml \
  --sf-checkpoint checkpoints/hest1k_human_visium_sf/context_distribution_light_hipt256_leave_slide_out/best.pt \
  --expression-checkpoint checkpoints/hest1k_human_visium_expression/highconf_symbol95_rate/best.pt \
  --splits test
```

The evaluator keeps the rate prediction fixed and reports three quantities
separately: rate-only reconstruction, predicted-SF count reconstruction and
oracle measured-SF reconstruction.

## Validation boundary

GitHub Actions verifies that every documented CLI imports and supports
`--help`, that the mean-one SF and `rate x SF` transformations pass unit tests,
and that the public configuration and checkpoint contracts remain stable. A
successful CI run does not claim that HEST assets were downloaded or that a
manuscript-scale training job was rerun inside CI.
