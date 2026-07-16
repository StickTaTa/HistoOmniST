import hashlib
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "download_release_models.py"


def load_module():
    spec = importlib.util.spec_from_file_location("download_release_models", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_verify_file_accepts_the_expected_size_and_hash(tmp_path: Path) -> None:
    module = load_module()
    path = tmp_path / "model.pt"
    payload = b"frozen-model"
    path.write_bytes(payload)

    module.verify_file(
        path,
        expected_bytes=len(payload),
        expected_sha256=hashlib.sha256(payload).hexdigest(),
    )


def test_verify_file_rejects_a_hash_mismatch(tmp_path: Path) -> None:
    module = load_module()
    path = tmp_path / "model.pt"
    path.write_bytes(b"wrong-model")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        module.verify_file(path, expected_bytes=11, expected_sha256="0" * 64)
