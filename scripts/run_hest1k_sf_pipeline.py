from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_command(cmd: list[str], *, dry_run: bool) -> int:
    print(" ".join(cmd))
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=ROOT, check=False).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the HEST-1k SF pipeline.")
    parser.add_argument("--config", type=Path, default=Path("configs/hest1k_human_visium_sf.yaml"))
    parser.add_argument("--split-config", type=Path, default=Path("configs/hest1k_splits.yaml"))
    parser.add_argument("--mode", choices=["metadata", "full"], default="metadata")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    py = sys.executable
    commands = [
        [py, "scripts/hest_audit_metadata.py", "--config", str(args.config)],
        [py, "scripts/hest_build_manifest.py", "--config", str(args.config)],
        [py, "scripts/hest_make_splits.py", "--config", str(args.split_config), "--source", "auto", "--write-split-manifest"],
    ]
    if args.mode == "full":
        commands.extend(
            [
                [py, "scripts/hest_prepare_slide_arrays.py"],
                [py, "scripts/train_sf.py", "--config", str(args.config), "--device", args.device],
                [py, "scripts/run_sf_baselines.py", "--config", str(args.config)],
            ]
        )

    log_dir = ROOT / "results" / "hest1k_human_visium_sf" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    rows = []
    for cmd in commands:
        rc = run_command(cmd, dry_run=args.dry_run)
        rows.append(f"returncode={rc} command={' '.join(cmd)}")
        if rc != 0:
            break
    log_path.write_text("\n".join(rows), encoding="utf-8")
    print(f"pipeline_log={log_path}")
    failures = [row for row in rows if row.startswith("returncode=") and not row.startswith("returncode=0")]
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
