import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def asset(asset_id: str, source: Path, target: str) -> dict:
    payload = source.read_bytes()
    return {
        "id": asset_id,
        "url": source.as_uri(),
        "target": target,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def test_download_hipt_assets_uses_the_manifest_targets(tmp_path: Path) -> None:
    module = load_script("download_hipt_assets")
    source_file = tmp_path / "source.py"
    weight_file = tmp_path / "weight.pth"
    source_file.write_bytes(b"official-hipt-source")
    weight_file.write_bytes(b"official-hipt-weight")
    manifest = {
        "external_dependencies": [
            {
                "id": "hipt_vit256",
                "source_file": asset("hipt_source", source_file, "third_party/HIPT/source.py"),
                "weights": asset("hipt_weight", weight_file, "third_party/HIPT/weight.pth"),
            }
        ]
    }
    manifest_path = tmp_path / "release_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    destination_root = tmp_path / "checkout"

    paths = module.download_hipt_assets(
        manifest_path=manifest_path,
        root=destination_root,
    )

    assert paths["hipt_source"].read_bytes() == b"official-hipt-source"
    assert paths["hipt_weight"].read_bytes() == b"official-hipt-weight"


def test_download_example_uses_the_manifest_target(tmp_path: Path) -> None:
    module = load_script("download_example_data")
    source = tmp_path / "example.tif"
    source.write_bytes(b"public-breast-example")
    manifest_path = tmp_path / "example_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "asset": asset(
                    "breast_example",
                    source,
                    "data/examples/breast_xenium/example.tif",
                )
            }
        ),
        encoding="utf-8",
    )
    destination_root = tmp_path / "checkout"

    path = module.download_example(
        manifest_path=manifest_path,
        root=destination_root,
    )

    assert path == destination_root / "data/examples/breast_xenium/example.tif"
    assert path.read_bytes() == b"public-breast-example"


@pytest.mark.parametrize(
    "script_name",
    [
        "download_release_models.py",
        "download_hipt_assets.py",
        "download_example_data.py",
    ],
)
def test_download_commands_work_from_an_uninstalled_checkout(script_name: str) -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)

    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script_name), "--help"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
