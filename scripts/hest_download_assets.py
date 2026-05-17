from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.hf_api import RepoFile

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from histoomnist.hest.metadata import filter_hest_metadata, load_hest_metadata
from histoomnist.utils.config import load_config
from histoomnist.utils.project_paths import resolve_project_path


ASSET_PATTERNS = {
    "metadata": ("metadata/{sample_id}.json",),
    "st": ("st/{sample_id}.h5ad",),
    "patches": ("patches/{sample_id}.h5",),
    "thumbnails": ("thumbnails/{sample_id}_downscaled_fullres.jpeg",),
    "wsis": ("wsis/{sample_id}.tif",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download selected HEST-1k assets from the gated MahmoodLab/hest "
            "Hugging Face dataset. Defaults to a dry-run plan."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("configs/hest1k_human_visium_sf.yaml"))
    parser.add_argument("--repo-id", default="MahmoodLab/hest")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument(
        "--asset",
        action="append",
        choices=sorted(ASSET_PATTERNS),
        default=None,
        help="Asset group to download. Repeat to request multiple groups.",
    )
    parser.add_argument("--sample-id", action="append", default=None)
    parser.add_argument("--max-slides", type=int, default=None)
    parser.add_argument("--out-csv", type=Path, default=Path("data/HEST-1k/manifests/hest_download_plan.csv"))
    parser.add_argument("--download", action="store_true", help="Actually download files. Without this flag, only plans.")
    parser.add_argument("--token", default=None, help="Hugging Face token. Prefer HF_TOKEN env var or hf auth login.")
    parser.add_argument("--token-file", type=Path, default=None, help="File containing a Hugging Face token.")
    return parser.parse_args()


def resolve_token(args: argparse.Namespace) -> str | bool:
    if args.token:
        return args.token.strip()
    if args.token_file:
        token_path = resolve_project_path(args.token_file)
        return token_path.read_text(encoding="utf-8").strip()
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        token = os.environ.get(key)
        if token:
            return token.strip()
    return True


def selected_metadata(config: dict, sample_ids: list[str] | None, max_slides: int | None) -> pd.DataFrame:
    metadata_csv = resolve_project_path(config["paths"]["metadata_csv"])
    filters = config["filters"]
    df = load_hest_metadata(metadata_csv)
    filtered = filter_hest_metadata(
        df,
        species=str(filters.get("species", "Homo sapiens")),
        st_technology=str(filters.get("st_technology", "Visium")),
        min_spots_under_tissue=int(filters.get("min_spots_under_tissue", 200)),
    )
    if sample_ids:
        wanted = set(sample_ids)
        filtered = filtered[filtered["id"].astype(str).isin(wanted)].copy()
        missing = sorted(wanted - set(filtered["id"].astype(str)))
        if missing:
            raise ValueError(f"sample_id values do not match selected HEST filters: {missing}")
    if max_slides is not None:
        filtered = filtered.head(max_slides).copy()
    return filtered.reset_index(drop=True)


def planned_paths(sample_ids: list[str], assets: list[str]) -> list[str]:
    paths: list[str] = []
    for sample_id in sample_ids:
        for asset in assets:
            paths.extend(pattern.format(sample_id=sample_id) for pattern in ASSET_PATTERNS[asset])
    return paths


def repo_file_sizes(repo_id: str, revision: str) -> dict[str, int]:
    sizes: dict[str, int] = {}
    api = HfApi()
    for item in api.list_repo_tree(repo_id=repo_id, repo_type="dataset", revision=revision, recursive=True):
        if isinstance(item, RepoFile):
            sizes[item.path] = int(item.size or 0)
    return sizes


def format_gib(size_bytes: int) -> str:
    return f"{size_bytes / 1024**3:.2f} GiB"


def main() -> None:
    args = parse_args()
    cfg = load_config(resolve_project_path(args.config))
    raw_root = resolve_project_path(args.raw_root or cfg["paths"]["raw_root"])
    assets = args.asset or ["metadata", "st", "patches"]
    selected = selected_metadata(cfg, sample_ids=args.sample_id, max_slides=args.max_slides)
    sample_ids = selected["id"].astype(str).tolist()
    paths = planned_paths(sample_ids, assets)
    sizes = repo_file_sizes(args.repo_id, args.revision)

    rows = []
    missing = []
    for path in paths:
        exists = path in sizes
        if not exists:
            missing.append(path)
        rows.append(
            {
                "repo_id": args.repo_id,
                "revision": args.revision,
                "path": path,
                "asset_group": path.split("/", 1)[0],
                "sample_id": Path(path).stem.split("_downscaled_fullres")[0],
                "size_bytes": sizes.get(path, 0),
                "exists_in_repo": exists,
                "local_path": str(raw_root / path),
                "already_downloaded": (raw_root / path).exists(),
            }
        )
    plan = pd.DataFrame(rows)
    out_csv = resolve_project_path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(out_csv, index=False)

    present = plan[plan["exists_in_repo"]]
    total_size = int(present["size_bytes"].sum()) if not present.empty else 0
    already = int(present.loc[present["already_downloaded"], "size_bytes"].sum()) if not present.empty else 0
    remaining = total_size - already
    print(f"selected_slides={len(sample_ids)}")
    print(f"asset_groups={','.join(assets)}")
    print(f"planned_files={len(plan)}")
    print(f"missing_files={len(missing)}")
    print(f"total_size={format_gib(total_size)}")
    print(f"already_downloaded={format_gib(already)}")
    print(f"remaining_download={format_gib(remaining)}")
    print(f"wrote {out_csv}")

    if missing:
        preview = "\n".join(missing[:20])
        raise FileNotFoundError(f"Some planned files are missing from {args.repo_id}:\n{preview}")
    if not args.download:
        print("dry_run=true; pass --download to download files.")
        return

    token = resolve_token(args)
    raw_root.mkdir(parents=True, exist_ok=True)
    for row in plan.itertuples(index=False):
        if row.already_downloaded:
            continue
        print(f"downloading {row.path}")
        hf_hub_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            revision=args.revision,
            filename=row.path,
            local_dir=raw_root,
            token=token,
        )
    print("download_complete=true")


if __name__ == "__main__":
    main()
