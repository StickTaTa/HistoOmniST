# Ten-method benchmark workflow

The manuscript benchmark compares HistoOmniST with HiST, Hist2ST, HisToGene,
iStar, mclSTExp, Path2Space, sCellST, STimage, ST-Net and THItoGene on fixed
slide-level partitions.

The repository contains HEST adapters, pinned provenance and a common evaluator.
It does not redistribute complete third-party repositories, their checkpoints
or standardized per-spot prediction bundles. Obtain every method from its
upstream source, run the corresponding `scripts/hest_train_*_sourcefaithful.py`
entry point, and export predictions in the common bundle layout expected by:

```bash
python scripts/hest_evaluate_benchmark_predictions.py --help
python scripts/hest_eval_histoomnist_benchmark.py --help
```

The fixed split manifests and the final gene-wise metric tables are sufficient
to regenerate manuscript benchmark plots. Smoke and reduced-gene runs must not
be reported as formal benchmark rows.
