from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from histoomnist.utils.project_paths import resolve_project_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume HEST WSI downloads file-by-file and verify local file sizes."
    )
    parser.add_argument("--plan-csv", type=Path, default=Path("data/HEST-1k/manifests/hest_wsi_download_plan.csv"))
    parser.add_argument("--repo-id", default="MahmoodLab/hest")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--raw-root", type=Path, default=Path("data/HEST-1k/raw"))
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--sleep-seconds", type=float, default=20.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("results/hest1k_human_visium_sf/wsi_download_integrity.csv"),
    )
    return parser.parse_args()


def resolve_token() -> str | bool:
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        token = os.environ.get(key)
        if token:
            return token.strip()
    return True


def local_status(plan: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in plan.itertuples(index=False):
        local = Path(row.local_path)
        expected = int(row.size_bytes)
        actual = local.stat().st_size if local.exists() else 0
        if local.exists() and actual == expected:
            status = "complete"
        elif local.exists():
            status = "size_mismatch"
        else:
            status = "missing"
        rows.append(
            {
                "sample_id": row.sample_id,
                "path": row.path,
                "local_path": str(local),
                "expected_bytes": expected,
                "actual_bytes": actual,
                "remaining_bytes": max(expected - actual, 0),
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def write_report(report: pd.DataFrame, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_csv, index=False)
    print(report["status"].value_counts().to_string())
    print(f"expected_gib={report['expected_bytes'].sum() / 1024**3:.2f}")
    print(f"actual_final_gib={report['actual_bytes'].sum() / 1024**3:.2f}")
    remaining = report.loc[report["status"] != "complete", "expected_bytes"].sum()
    print(f"remaining_expected_gib={remaining / 1024**3:.2f}")
    print(f"wrote {out_csv}")


def main() -> None:
    args = parse_args()
    plan_path = resolve_project_path(args.plan_csv)
    raw_root = resolve_project_path(args.raw_root)
    out_csv = resolve_project_path(args.out_csv)
    plan = pd.read_csv(plan_path)
    report = local_status(plan)
    pending = report[report["status"] != "complete"].copy()
    if args.limit is not None:
        pending = pending.head(args.limit).copy()

    print("initial_status")
    write_report(report, out_csv)
    print(f"pending_this_run={len(pending)}")
    token = resolve_token()
    raw_root.mkdir(parents=True, exist_ok=True)

    failures: list[dict[str, object]] = []
    for idx, row in enumerate(pending.itertuples(index=False), start=1):
        print(f"[{idx}/{len(pending)}] download {row.path}", flush=True)
        ok = False
        for attempt in range(1, args.max_attempts + 1):
            try:
                hf_hub_download(
                    repo_id=args.repo_id,
                    repo_type="dataset",
                    revision=args.revision,
                    filename=str(row.path),
                    local_dir=raw_root,
                    token=token,
                )
                local = Path(row.local_path)
                actual = local.stat().st_size if local.exists() else 0
                if actual == int(row.expected_bytes):
                    ok = True
                    break
                raise RuntimeError(
                    f"size mismatch after download: expected={row.expected_bytes}, actual={actual}"
                )
            except Exception as exc:
                print(f"attempt={attempt} failed for {row.path}: {type(exc).__name__}: {exc}", flush=True)
                if attempt < args.max_attempts:
                    time.sleep(args.sleep_seconds * attempt)
        if not ok:
            failures.append({"path": row.path, "sample_id": row.sample_id})

        if idx % 10 == 0 or not ok:
            report = local_status(plan)
            write_report(report, out_csv)

    report = local_status(plan)
    write_report(report, out_csv)
    if failures:
        fail_path = out_csv.with_name("wsi_download_failures.csv")
        pd.DataFrame(failures).to_csv(fail_path, index=False)
        print(f"failed_files={len(failures)}")
        print(f"wrote {fail_path}")
        raise SystemExit(1)
    print("download_complete=true")


if __name__ == "__main__":
    main()

