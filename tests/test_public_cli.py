import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_HELP_SCRIPTS = [
    "histoomnist_predict_uploaded_wsi.py",
    "hest_audit_metadata.py",
    "hest_download_assets.py",
    "hest_convert_raw_to_processed_arrays.py",
    "hest_extract_patch_features.py",
    "hest_build_manifest.py",
    "hest_make_splits.py",
    "train_expression.py",
    "train_sf.py",
    "evaluate_combined.py",
    "hest_evaluate_benchmark_predictions.py",
    "hest_eval_histoomnist_benchmark.py",
]


@pytest.mark.parametrize("script_name", PUBLIC_HELP_SCRIPTS)
def test_public_cli_supports_help(script_name: str) -> None:
    environment = os.environ.copy()
    python_path = str(ROOT / "src")
    if environment.get("PYTHONPATH"):
        python_path += os.pathsep + environment["PYTHONPATH"]
    environment["PYTHONPATH"] = python_path

    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script_name), "--help"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout.lower()
