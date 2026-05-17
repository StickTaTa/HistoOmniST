# HistoOmniST

HistoOmniST is now organized around one primary task:

```text
HEST-1k human Visium H&E / HIPT features -> mean-one slide-normalized size factor
```

The immediate goal is to train and evaluate a morphology-derived size-factor prior that can later be used in count-scale H&E-to-ST prediction:

```text
count_i,g = rate_i,g * sf_i
```

## Size-Factor Definition

The committed target is mean-one size factor:

```text
total_i = sum_g raw_count_i,g
sf_i = total_i / mean(total_valid_spots_in_slide)
target_i = log(sf_i)
```

This is intentional. The SF module is not an isolated statistic; it participates in the count/rate decomposition. Mean-one normalization keeps the average slice-level count scale fixed and makes `rate = count / sf` consistent across slides. Do not switch this project to median-one SF unless the project owner explicitly changes the definition.

## Data Scope

The first training stage uses only HEST-1k human Visium samples:

```text
species == Homo sapiens
st_technology == Visium
spots_under_tissue >= 200
```

Metadata is stored at:

```text
data/HEST-1k/HEST_v1_3_0.csv
```

Raw HEST assets should be placed under:

```text
data/HEST-1k/raw
```

Prepared slide arrays should be placed under:

```text
data/HEST-1k/processed/<slide_id>/
  features.npy
  counts.npz
  coords.npy
  size_factor.npy      optional; computed from counts if missing
  spots.txt            optional
  genes.txt            optional
```

## Main Commands

Metadata-only smoke pipeline:

```powershell
conda run -n histogene_bench python scripts\run_hest1k_sf_pipeline.py --mode metadata
```

Build processed-data manifest:

```powershell
conda run -n histogene_bench python scripts\hest_build_manifest.py --config configs\hest1k_human_visium_sf.yaml
```

Create slide-level splits:

```powershell
conda run -n histogene_bench python scripts\hest_make_splits.py --write-split-manifest
```

Train SF model after processed arrays exist:

```powershell
conda run -n histogene_bench python scripts\train_sf.py --config configs\hest1k_human_visium_sf.yaml --device cuda
```

Run available baselines:

```powershell
conda run -n histogene_bench python scripts\run_sf_baselines.py --config configs\hest1k_human_visium_sf.yaml
```

## Current Structure

```text
configs/         HEST-1k training, split, baseline, and path configs
data/HEST-1k/    HEST metadata, raw assets, processed arrays, manifests, splits
scripts/         HEST-1k SF pipeline entry points
src/histoomnist/ Core data, model, training, evaluation, and HEST helper code
results/         Generated reports and metrics
checkpoints/     Model checkpoints
```
