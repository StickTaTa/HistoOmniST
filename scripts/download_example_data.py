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


DEFAULT_MANIFEST = ROOT / "examples" / "breast_xenium" / "example_manifest.json"


def download_example(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    root: Path = ROOT,
    force: bool = False,
) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    asset = manifest["asset"]
    return download_file(
        url=asset["url"],
        destination=root / asset["target"],
        expected_bytes=asset["bytes"],
        expected_sha256=asset["sha256"],
        force=force,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the public 10x Xenium breast cancer H&E example."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    path = download_example(
        manifest_path=manifest_path,
        root=args.root.resolve(),
        force=args.force,
    )
    print(f"verified breast_xenium_rep1_he: {path}")


if __name__ == "__main__":
    main()
