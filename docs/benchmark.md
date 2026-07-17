# Ten-method benchmark workflow

The manuscript benchmark compares HistoOmniST with HiST, Hist2ST, HisToGene,
iStar, mclSTExp, Path2Space, sCellST, STimage, ST-Net and THItoGene on fixed
slide-level HEST partitions.

This repository contains project-owned source-faithful adapters, fixed split
manifests, provenance records and a common evaluator. It does not redistribute
complete third-party repositories, third-party checkpoints or standardized
per-spot prediction bundles. Those assets must be obtained from the original
authors under their respective licenses.

## Validation boundary

CI verifies that both public benchmark evaluators import and expose their CLI
contracts:

```bash
python scripts/hest_evaluate_benchmark_predictions.py --help
python scripts/hest_eval_histoomnist_benchmark.py --help
```

CI does not train all external methods or reproduce the manuscript benchmark
table. A formal rerun requires the complete HEST processed arrays and upstream
method environments.

## Required common inputs

All methods must use the committed coverage-95 expression configuration and
fixed slide-level splits. Each evaluated slide requires aligned measured
counts, size factor, spot identifiers and gene identifiers in the processed
HEST manifest described in [Training](training.md).

External predictions are supplied as one two-dimensional NumPy array per
slide. Rows must follow the processed slide spot order. Columns must follow the
target gene order unless `--prediction-genes-path` supplies a newline-delimited
gene list.

The common evaluator accepts `count`, `log1p_count`, `rate` and `log1p_rate`
prediction kinds. By default it searches these layouts:

```text
<prediction_root>/predictions/<sample_id>_<kind>.npy
<prediction_root>/predictions/<sample_id>.npy
<prediction_root>/<sample_id>/<kind>.npy
<prediction_root>/<sample_id>/pred.npy
<prediction_root>/<sample_id>_<kind>.npy
<prediction_root>/<sample_id>.npy
```

Use `--prediction-pattern` when an upstream method uses another deterministic
layout.

## Reproduction stages

### 1. Prepare common HEST targets

Complete the ordered preparation workflow in [Training](training.md), including
the processed manifest and fixed slide-level split assignment.

### 2. Run each upstream method

Obtain the original implementation and checkpoint for each comparator. The
`scripts/hest_train_*_sourcefaithful.py` entry points document the project-owned
adaptation layer. Keep each method in its own compatible environment when
dependencies conflict.

### 3. Export standardized prediction arrays

Export one aligned prediction matrix per slide and record whether its values
represent counts, log1p counts, rates or log1p rates. Do not infer a scale from
the values after export.

### 4. Evaluate an external prediction bundle

The required arguments are the method name and output directory. For example:

```bash
python scripts/hest_evaluate_benchmark_predictions.py \
  --method-name Path2Space \
  --prediction-root /path/to/path2space_predictions \
  --prediction-kind count \
  --out-dir results/benchmark/path2space \
  --splits test
```

The evaluator writes gene-wise, overall, organ-level and slide-level metrics,
the evaluated slide table and `run_summary.json`.

### 5. Evaluate HistoOmniST in the same format

```bash
python scripts/hest_eval_histoomnist_benchmark.py \
  --expression-config configs/hest1k_human_visium_expression_highconf_symbol95.yaml \
  --sf-config configs/hest1k_human_visium_sf_current.yaml \
  --splits test \
  --out-dir results/benchmark/histoomnist
```

The HistoOmniST evaluator resolves the frozen checkpoints from the committed
configs unless explicit checkpoint paths are provided.

## Reporting rules

- Use slide-level partitions; never report random spot-level splits as headline
  benchmark results.
- Preserve each method's source scale and declare `--prediction-kind` exactly.
- Report only genes shared by the target and supplied prediction bundle.
- Treat `--oracle-smoke-test`, `--max-slides`, `--max-slide-spots` and reduced
  gene runs as diagnostics, not formal benchmark rows.
- Record upstream revisions, checkpoints and environment details for every
  comparator. See [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).
