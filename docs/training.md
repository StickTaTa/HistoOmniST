# HEST-1k data preparation and training

## Size-factor definition

For valid spots in one slide:

```text
total[i] = sum_g raw_count[i, g]
SF[i] = total[i] / mean(total[valid spots in the slide])
target[i] = log(SF[i])
rate[i, g] = count[i, g] / SF[i]
```

This is a mean-one definition. Do not replace it with median normalization.

## Local data layout

Copy `configs/local_paths.example.yaml` to `configs/local_paths.yaml`. Absolute
paths belong only in the ignored local file.

Download the HIPT source and ViT-256 weight directly from the original HIPT
project before extracting patch features:

```bash
python scripts/download_hipt_assets.py
```

The example local-path configuration already points to the resulting
`third_party/HIPT/` layout.

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

## Preparation

```bash
python scripts/hest_audit_metadata.py
python scripts/hest_download_assets.py
python scripts/hest_convert_raw_to_processed_arrays.py
python scripts/hest_extract_patch_features.py
python scripts/hest_build_manifest.py --config configs/hest1k_human_visium_sf_context_distribution_light.yaml
python scripts/hest_make_splits.py --write-split-manifest
```

Barcode alignment is checked before arrays are accepted. Headline results use
fixed slide-level train, validation and test partitions.

## Training

```bash
python scripts/train_expression.py \
  --config configs/hest1k_human_visium_expression_highconf_symbol95.yaml \
  --device cuda

python scripts/train_sf.py \
  --config configs/hest1k_human_visium_sf_context_distribution_light.yaml \
  --device cuda
```

The rate checkpoint predicts `log1p(rate)` for 16,942 coverage-95 canonical
gene symbols. The SF checkpoint predicts `log(SF)` from the same 1,161-feature
context representation.

## Combined evaluation

```bash
python scripts/evaluate_combined.py \
  --sf-config configs/hest1k_human_visium_sf_context_distribution_light.yaml \
  --expression-config configs/hest1k_human_visium_expression_highconf_symbol95.yaml \
  --sf-checkpoint checkpoints/hest1k_human_visium_sf/context_distribution_light_hipt256_leave_slide_out/best.pt \
  --expression-checkpoint checkpoints/hest1k_human_visium_expression/highconf_symbol95_rate/best.pt \
  --splits test
```

The evaluator reports rate-only reconstruction, predicted-SF count
reconstruction and oracle-SF reconstruction separately.
