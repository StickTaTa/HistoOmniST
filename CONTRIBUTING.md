# Contributing

Open an issue before changing the model or output contract. Pull requests must:

1. Preserve the mean-one slide-normalized SF definition.
2. Keep headline evaluation splits at slide level.
3. Add a focused test for behavior changes.
4. Pass `python -m pytest -q` and `python scripts/audit_public_release.py`.
5. Avoid committing raw data, checkpoints, predictions or third-party source.

Use `configs/local_paths.yaml` for machine-specific paths. Do not add local
absolute paths to source, notebooks or public configuration files.
