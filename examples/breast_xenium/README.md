# Breast Xenium inference example

The quickstart notebook uses the public 10x Genomics H&E image from Xenium
FFPE Human Breast Cancer Rep1, the same public tissue section used for the
manuscript's Xenium discussion example. The 1.43 GB image is not stored in this
repository.

Run `python scripts/download_example_data.py` or execute the download cell in
`notebooks/00_quickstart_inference.ipynb`. The verified image is written to:

```text
data/examples/breast_xenium/Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image.ome.tif
```

The source URL, expected byte count and SHA-256 are fixed in
`example_manifest.json`. Prediction tables and plots are written under
`outputs/example_breast_xenium/`, which is also excluded from Git.
