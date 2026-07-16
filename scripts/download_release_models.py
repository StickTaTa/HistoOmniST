from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "models" / "release_manifest.json"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, *, expected_bytes: int, expected_sha256: str) -> None:
    actual_bytes = path.stat().st_size
    if actual_bytes != int(expected_bytes):
        raise ValueError(
            f"File-size mismatch for {path}: expected {expected_bytes}, got {actual_bytes}"
        )
    actual_sha256 = sha256_file(path)
    if actual_sha256.lower() != expected_sha256.lower():
        raise ValueError(
            f"SHA-256 mismatch for {path}: expected {expected_sha256}, got {actual_sha256}"
        )


def download_asset(asset: dict, *, root: Path, force: bool = False) -> Path:
    destination = root / asset["target"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        verify_file(
            destination,
            expected_bytes=asset["bytes"],
            expected_sha256=asset["sha256"],
        )
        return destination

    temporary = destination.with_suffix(destination.suffix + ".download")
    try:
        with urllib.request.urlopen(asset["url"]) as response, temporary.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        verify_file(
            temporary,
            expected_bytes=asset["bytes"],
            expected_sha256=asset["sha256"],
        )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and verify the frozen HistoOmniST release checkpoints."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    root = args.root.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for asset in manifest["assets"]:
        path = download_asset(asset, root=root, force=args.force)
        print(f"verified {asset['id']}: {path}")


if __name__ == "__main__":
    main()
