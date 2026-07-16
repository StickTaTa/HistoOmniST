from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from histoomnist.utils.downloads import download_file  # noqa: E402


DEFAULT_MANIFEST = ROOT / "models" / "release_manifest.json"


def hipt_assets(manifest: dict) -> list[dict]:
    dependency = next(
        item for item in manifest["external_dependencies"] if item["id"] == "hipt_vit256"
    )
    return [dependency["source_file"], dependency["weights"]]


def download_hipt_assets(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    root: Path = ROOT,
    force: bool = False,
) -> dict[str, Path]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    downloaded: dict[str, Path] = {}
    for asset in hipt_assets(manifest):
        downloaded[asset["id"]] = download_file(
            url=asset["url"],
            destination=root / asset["target"],
            expected_bytes=asset["bytes"],
            expected_sha256=asset["sha256"],
            force=force,
        )
    return downloaded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the required source file and ViT-256 weight from mahmoodlab/HIPT."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    for asset_id, path in download_hipt_assets(
        manifest_path=manifest_path,
        root=args.root.resolve(),
        force=args.force,
    ).items():
        print(f"verified {asset_id}: {path}")


if __name__ == "__main__":
    main()
