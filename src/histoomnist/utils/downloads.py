from __future__ import annotations

import hashlib
import shutil
import urllib.request
from pathlib import Path


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(
    path: str | Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
) -> None:
    asset_path = Path(path)
    actual_bytes = asset_path.stat().st_size
    if actual_bytes != int(expected_bytes):
        raise ValueError(
            f"File-size mismatch for {asset_path}: expected {expected_bytes}, got {actual_bytes}"
        )
    actual_sha256 = sha256_file(asset_path)
    if actual_sha256.lower() != expected_sha256.lower():
        raise ValueError(
            f"SHA-256 mismatch for {asset_path}: expected {expected_sha256}, got {actual_sha256}"
        )


def download_file(
    *,
    url: str,
    destination: str | Path,
    expected_bytes: int,
    expected_sha256: str,
    force: bool = False,
) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        verify_file(
            path,
            expected_bytes=expected_bytes,
            expected_sha256=expected_sha256,
        )
        return path

    temporary = path.with_suffix(path.suffix + ".download")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "HistoOmniST/0.1"})
        with urllib.request.urlopen(request) as response, temporary.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        verify_file(
            temporary,
            expected_bytes=expected_bytes,
            expected_sha256=expected_sha256,
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path
