# HistoOmniST

HistoOmniST predicts a virtual spatial transcriptome from an H&E slide by
predicting expression rate and a mean-one, slide-normalized size factor (SF),
then reconstructing count-scale expression.

```text
H&E tiles -> expression-rate model ----\
                                        x -> virtual count-scale expression
H&E tiles -> size-factor model --------/

count[i, g] = rate[i, g] * SF[i]
mean(SF within each slide) = 1
```

The frozen expression checkpoint contains 16,942 genes. The public TCGA atlas
and default inference tutorial use a prespecified subset of 28 genes and eight
programs. Outputs are model-derived virtual predictions, not measured spatial
transcriptomes or diagnostic annotations.

## Use a pretrained model

Create the environment and install the package:

```bash
conda env create -f environment.yml
conda activate histoomnist
```

Download and verify the frozen expression-rate and SF checkpoints:

```bash
python scripts/download_release_models.py
```

The HIPT ViT-256 feature extractor is an upstream dependency and is not
redistributed in this repository. Install it as described in
[`docs/installation.md`](docs/installation.md), then run:

```bash
python scripts/histoomnist_predict_uploaded_wsi.py \
  --input /path/to/slide.svs \
  --out-dir outputs/example \
  --hipt-source third_party/benchmarks/iStar \
  --hipt-weights third_party/benchmarks/iStar/checkpoints/vit256_small_dino.pth \
  --write-report --write-zip
```

The command accepts common whole-slide formats through OpenSlide, tiled TIFF
through tifffile/zarr, and ordinary RGB images through Pillow. Repeat
`--input`, or use `--input-list`, for cohort inference.

## Output scale

For each selected gene, inference writes:

- `rate_log1p_GENE`: frozen expression-rate model output.
- `pred_sf`: predicted SF normalized to mean one within the slide.
- `count_GENE`: `max(expm1(rate_log1p_GENE), 0) * pred_sf`.
- `count_log1p_GENE`: `log1p(count_GENE)`.
- `gene_GENE`: website-compatible alias of `count_log1p_GENE`.
- `program_NAME`: mean of available count-log1p member genes.

See [`docs/inference.md`](docs/inference.md) for the complete schema and tiling
options.

## Train on HEST-1k

HEST-1k raw data are downloaded from their original public source and are not
stored in this repository. The training sequence is:

```bash
python scripts/hest_audit_metadata.py
python scripts/hest_download_assets.py
python scripts/hest_convert_raw_to_processed_arrays.py
python scripts/hest_extract_patch_features.py
python scripts/hest_build_manifest.py --config configs/hest1k_human_visium_sf_context_distribution_light.yaml
python scripts/hest_make_splits.py --write-split-manifest
python scripts/train_expression.py --config configs/hest1k_human_visium_expression_highconf_symbol95.yaml --device cuda
python scripts/train_sf.py --config configs/hest1k_human_visium_sf_context_distribution_light.yaml --device cuda
python scripts/evaluate_combined.py \
  --sf-config configs/hest1k_human_visium_sf_context_distribution_light.yaml \
  --expression-config configs/hest1k_human_visium_expression_highconf_symbol95.yaml \
  --sf-checkpoint checkpoints/hest1k_human_visium_sf/context_distribution_light_hipt256_leave_slide_out/best.pt \
  --expression-checkpoint checkpoints/hest1k_human_visium_expression/highconf_symbol95_rate/best.pt
```

All headline evaluations use slide-level splits. See
[`docs/training.md`](docs/training.md) for input layouts, exact definitions and
full commands.

## Tutorials

- [`00_quickstart_inference.ipynb`](notebooks/00_quickstart_inference.ipynb)
- [`01_hest_data_preparation.ipynb`](notebooks/01_hest_data_preparation.ipynb)
- [`02_train_rate_and_sf.ipynb`](notebooks/02_train_rate_and_sf.ipynb)
- [`03_count_scale_evaluation.ipynb`](notebooks/03_count_scale_evaluation.ipynb)
- [`04_benchmark_workflow.ipynb`](notebooks/04_benchmark_workflow.ipynb)

The notebooks are lightweight guides. Full cohort training and ten-method
benchmarking should be run through the command-line workflows.

## Repository scope

Git contains code, configurations, fixed splits, metadata, documentation and
tutorials. The two frozen HistoOmniST checkpoints are GitHub Release assets.
Raw HEST/TCGA/external data, full prediction bundles, third-party repositories,
website user uploads and manuscript Source Data are not stored here.

## Citation and license

See [`CITATION.cff`](CITATION.cff) for citation metadata. HistoOmniST code is
released under the BSD 3-Clause License. Third-party methods and model weights
remain governed by their original licenses; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
