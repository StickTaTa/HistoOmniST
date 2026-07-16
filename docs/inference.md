# Frozen WSI inference

## Inputs

`scripts/histoomnist_predict_uploaded_wsi.py` accepts one or more local H&E
images with repeated `--input`, or a text/CSV file supplied through
`--input-list`. A CSV may contain `path`, `input_path`, `wsi_path` or
`local_path`.

The default panel is `configs/manuscript_release_28_gene_panel.json`. Use
`--selected-genes GENE1 GENE2` to request another subset available in the
16,942-gene expression checkpoint.

Before inference, download the HistoOmniST checkpoints and the original HIPT
assets:

```bash
python scripts/download_release_models.py
python scripts/download_hipt_assets.py
```

The default HIPT paths are
`third_party/HIPT/1-Hierarchical-Pretraining/` and
`third_party/HIPT/HIPT_4K/Checkpoints/vit256_small_dino.pth`. Override them
with `--hipt-source` and `--hipt-weights` only when the same official assets are
stored elsewhere.

## Cohort input lists

For batch inference, pass a newline-delimited text file or a CSV containing one
of `path`, `input_path`, `wsi_path` or `local_path`:

```csv
path
/data/slides/case_001.svs
/data/slides/case_002.svs
```

```bash
python scripts/histoomnist_predict_uploaded_wsi.py \
  --input-list cohort_slides.csv \
  --out-dir outputs/cohort \
  --skip-figures
```

The command writes per-slide tile tables, a combined `tile_predictions.csv`
and `slide_features.csv`. It retains `input_name` and a stable `slide_id`, which
can be joined to cohort metadata outside the prediction workflow. The default
release panel contains the 28 prespecified atlas genes and eight programs; it
does not imply that the public atlas contains predictions for all 16,942 genes
available in the expression-rate checkpoint.

## Processing

1. Read the slide with OpenSlide, tifffile/zarr or Pillow.
2. Identify tissue tiles from a thumbnail mask.
3. Extract HIPT ViT-256 features from 224 x 224 pixel tiles.
4. Predict `log1p(rate)` and raw `log(SF)`.
5. Normalize predicted SF to mean one within each slide.
6. Reconstruct count as `max(expm1(log1p_rate), 0) * SF`.
7. Apply `log1p` and calculate programs from count-log1p member genes.

## Important options

```text
--stride                      fixed grid stride
--target-tiles-per-slide      adapt stride toward a target tile count
--min-stride                  lower bound for adaptive stride
--tissue-threshold            minimum thumbnail tissue fraction
--max-tiles-per-slide         per-slide safety cap; 0 disables the cap
--skip-figures                write tables without rendered maps
--write-report                create an HTML summary
--write-zip                   package the run outputs
```

## Outputs

Each slide receives a tile-level CSV, thumbnail and optional spatial map. The
run also writes combined predictions, slide summaries and `run_summary.json`.
`gene_*` columns are aliases of `count_log1p_*`; they are not rate-only values.

Predictions should not be interpreted as measured spatial transcriptomes or
used for clinical diagnosis.

For a visible, step-by-step walkthrough of these processing stages, use
`notebooks/00_quickstart_inference.ipynb`. It runs the public 10x breast cancer
H&E example without invoking the CLI as a black box.
