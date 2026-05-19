from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from histoomnist.eval.evaluate_combined import evaluate as evaluate_combined
from histoomnist.eval.evaluate_expression import evaluate as evaluate_expression
from histoomnist.eval.evaluate_sf import evaluate as evaluate_sf
from histoomnist.train.train_expression import train as train_expression
from histoomnist.train.train_sf import train as train_sf
from histoomnist.utils.config import load_config
from histoomnist.utils.project_paths import project_root, resolve_project_path


DEFAULT_TASK_TABLE = "results/hest1k_human_visium_expression/generalization_readiness/generated_task_files.csv"
DEFAULT_OUT_DIR = "results/hest1k_human_visium_expression/generalization_runs"
DEFAULT_CHECKPOINT_ROOT = "checkpoints/hest1k_generalization_runs"
MAX_TASK_DIR_NAME_LENGTH = 64


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def rel_project_path(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(project_root())).replace("\\", "/")
    except ValueError:
        return str(p).replace("\\", "/")


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    if isinstance(value, Path):
        return rel_project_path(value)
    return value


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=False), encoding="utf-8")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(data), indent=2), encoding="utf-8")


def run_name_default() -> str:
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


def compact_task_dir_name(task_slug: str, max_length: int = MAX_TASK_DIR_NAME_LENGTH) -> str:
    slug = str(task_slug)
    if len(slug) <= max_length:
        return slug
    digest = hashlib.sha1(slug.encode("utf-8")).hexdigest()[:8]
    prefix = slug[: max_length - len(digest) - 1].rstrip("_-")
    return f"{prefix}_{digest}"


def load_task_table(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    required = {
        "split_type",
        "heldout",
        "task_slug",
        "task_manifest",
        "expression_config",
        "sf_config",
    }
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Task table missing columns: {missing}")
    return table


def select_tasks(
    table: pd.DataFrame,
    *,
    split_types: list[str] | None,
    task_slugs: list[str] | None,
    heldouts: list[str] | None,
    max_tasks: int | None,
) -> pd.DataFrame:
    out = table.copy()
    if split_types:
        out = out[out["split_type"].astype(str).isin(split_types)].copy()
    if task_slugs:
        out = out[out["task_slug"].astype(str).isin(task_slugs)].copy()
    if heldouts:
        out = out[out["heldout"].astype(str).isin(heldouts)].copy()
    if max_tasks is not None:
        out = out.head(int(max_tasks)).copy()
    if out.empty:
        raise ValueError("No generalization tasks selected.")
    return out.reset_index(drop=True)


def update_runtime_config(
    cfg: dict[str, Any],
    *,
    device: str | None,
    epochs: int | None,
    output_dir: Path,
) -> dict[str, Any]:
    runtime_cfg = copy.deepcopy(cfg)
    if device is not None:
        runtime_cfg["device"] = device
    if epochs is not None:
        runtime_cfg.setdefault("training", {})["epochs"] = int(epochs)
    runtime_cfg.setdefault("output", {})["dir"] = rel_project_path(output_dir)
    return runtime_cfg


def log_context(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", encoding="utf-8")
    return handle, redirect_stdout(Tee(sys.stdout, handle)), redirect_stderr(Tee(sys.stderr, handle))


def flush_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def base_result(row: pd.Series, stage: str, task_dir: Path) -> dict[str, Any]:
    return {
        "split_type": str(row["split_type"]),
        "heldout": str(row["heldout"]),
        "task_slug": str(row["task_slug"]),
        "stage": stage,
        "task_dir": rel_project_path(task_dir),
    }


def run_sf_stage(
    row: pd.Series,
    *,
    task_dir: Path,
    checkpoint_dir: Path,
    device: str | None,
    epochs: int | None,
    splits: list[str],
    skip_existing: bool,
) -> dict[str, Any]:
    result = base_result(row, "sf", task_dir)
    sf_checkpoint = checkpoint_dir / "sf" / "best.pt"
    metrics_path = task_dir / "sf_test_metrics.json"
    runtime_cfg_path = task_dir / "sf_config.runtime.yaml"
    log_path = task_dir / "sf.log"
    cfg = load_config(resolve_project_path(row["sf_config"]))
    runtime_cfg = update_runtime_config(cfg, device=device, epochs=epochs, output_dir=sf_checkpoint.parent)
    write_yaml(runtime_cfg_path, runtime_cfg)
    start = time.time()
    train_status = "skipped_existing"
    if not (skip_existing and sf_checkpoint.exists()):
        handle, stdout_redirect, stderr_redirect = log_context(log_path)
        try:
            with handle, stdout_redirect, stderr_redirect:
                train_sf(runtime_cfg)
        finally:
            flush_cuda_cache()
        train_status = "ran"
    metrics = evaluate_sf(runtime_cfg, checkpoint=sf_checkpoint, split_names=splits, out_json=metrics_path)
    result.update(
        {
            "status": "ok",
            "train_status": train_status,
            "checkpoint": rel_project_path(sf_checkpoint),
            "runtime_config": rel_project_path(runtime_cfg_path),
            "log": rel_project_path(log_path),
            "metrics_json": rel_project_path(metrics_path),
            "elapsed_seconds": time.time() - start,
            **{f"metric_{key}": value for key, value in metrics.items()},
        }
    )
    return result


def run_expression_stage(
    row: pd.Series,
    *,
    task_dir: Path,
    checkpoint_dir: Path,
    device: str | None,
    epochs: int | None,
    splits: list[str],
    skip_existing: bool,
) -> dict[str, Any]:
    result = base_result(row, "expression", task_dir)
    expression_checkpoint = checkpoint_dir / "expression" / "best.pt"
    metrics_path = task_dir / "expression_test_metrics.json"
    runtime_cfg_path = task_dir / "expression_config.runtime.yaml"
    log_path = task_dir / "expression.log"
    cfg = load_config(resolve_project_path(row["expression_config"]))
    runtime_cfg = update_runtime_config(cfg, device=device, epochs=epochs, output_dir=expression_checkpoint.parent)
    write_yaml(runtime_cfg_path, runtime_cfg)
    start = time.time()
    train_status = "skipped_existing"
    if not (skip_existing and expression_checkpoint.exists()):
        handle, stdout_redirect, stderr_redirect = log_context(log_path)
        try:
            with handle, stdout_redirect, stderr_redirect:
                train_expression(runtime_cfg)
        finally:
            flush_cuda_cache()
        train_status = "ran"
    metrics = evaluate_expression(runtime_cfg, checkpoint=expression_checkpoint, split_names=splits, out_json=metrics_path)
    result.update(
        {
            "status": "ok",
            "train_status": train_status,
            "checkpoint": rel_project_path(expression_checkpoint),
            "runtime_config": rel_project_path(runtime_cfg_path),
            "log": rel_project_path(log_path),
            "metrics_json": rel_project_path(metrics_path),
            "elapsed_seconds": time.time() - start,
            **{f"metric_{key}": value for key, value in metrics.items()},
        }
    )
    return result


def run_combined_stage(
    row: pd.Series,
    *,
    task_dir: Path,
    checkpoint_dir: Path,
    device: str | None,
    splits: list[str],
) -> dict[str, Any]:
    result = base_result(row, "combined", task_dir)
    sf_checkpoint = checkpoint_dir / "sf" / "best.pt"
    expression_checkpoint = checkpoint_dir / "expression" / "best.pt"
    metrics_path = task_dir / "combined_test_metrics.json"
    if not sf_checkpoint.exists() or not expression_checkpoint.exists():
        result.update(
            {
                "status": "skipped_missing_checkpoint",
                "sf_checkpoint": rel_project_path(sf_checkpoint),
                "expression_checkpoint": rel_project_path(expression_checkpoint),
            }
        )
        return result
    sf_cfg = load_config(resolve_project_path(row["sf_config"]))
    expression_cfg = load_config(resolve_project_path(row["expression_config"]))
    if device is not None:
        sf_cfg["device"] = device
        expression_cfg["device"] = device
    start = time.time()
    metrics = evaluate_combined(
        sf_config=sf_cfg,
        expression_config=expression_cfg,
        sf_checkpoint=sf_checkpoint,
        expression_checkpoint=expression_checkpoint,
        split_names=splits,
        out_json=metrics_path,
    )
    result.update(
        {
            "status": "ok",
            "sf_checkpoint": rel_project_path(sf_checkpoint),
            "expression_checkpoint": rel_project_path(expression_checkpoint),
            "metrics_json": rel_project_path(metrics_path),
            "elapsed_seconds": time.time() - start,
            **{f"metric_{key}": value for key, value in metrics.items()},
        }
    )
    return result


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def run_generalization_tasks(
    *,
    task_table: Path,
    out_dir: Path,
    checkpoint_root: Path,
    run_name: str,
    stages: list[str],
    split_types: list[str] | None,
    task_slugs: list[str] | None,
    heldouts: list[str] | None,
    max_tasks: int | None,
    device: str | None,
    sf_epochs: int | None,
    expression_epochs: int | None,
    splits: list[str],
    skip_existing: bool,
) -> Path:
    root = project_root()
    os.chdir(root)
    table = load_task_table(task_table)
    selected = select_tasks(
        table,
        split_types=split_types,
        task_slugs=task_slugs,
        heldouts=heldouts,
        max_tasks=max_tasks,
    )
    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(run_dir / "selected_tasks.csv", index=False)
    run_meta = {
        "run_name": run_name,
        "task_table": rel_project_path(task_table),
        "stages": stages,
        "split_types": split_types,
        "task_slugs": task_slugs,
        "heldouts": heldouts,
        "max_tasks": max_tasks,
        "device": device,
        "sf_epochs": sf_epochs,
        "expression_epochs": expression_epochs,
        "splits": splits,
        "skip_existing": skip_existing,
    }
    write_json(run_dir / "run_manifest.json", run_meta)

    summary_rows: list[dict[str, Any]] = []
    summary_path = run_dir / "summary.csv"
    for row in selected.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        task_dir_name = compact_task_dir_name(str(row_series["task_slug"]))
        task_dir = run_dir / str(row_series["split_type"]) / task_dir_name
        checkpoint_dir = checkpoint_root / run_name / str(row_series["split_type"]) / task_dir_name
        for stage in stages:
            try:
                if stage == "sf":
                    stage_result = run_sf_stage(
                        row_series,
                        task_dir=task_dir,
                        checkpoint_dir=checkpoint_dir,
                        device=device,
                        epochs=sf_epochs,
                        splits=splits,
                        skip_existing=skip_existing,
                    )
                elif stage == "expression":
                    stage_result = run_expression_stage(
                        row_series,
                        task_dir=task_dir,
                        checkpoint_dir=checkpoint_dir,
                        device=device,
                        epochs=expression_epochs,
                        splits=splits,
                        skip_existing=skip_existing,
                    )
                elif stage == "combined":
                    stage_result = run_combined_stage(
                        row_series,
                        task_dir=task_dir,
                        checkpoint_dir=checkpoint_dir,
                        device=device,
                        splits=splits,
                    )
                else:
                    raise ValueError(f"Unsupported stage: {stage}")
            except Exception as exc:
                stage_result = base_result(row_series, stage, task_dir)
                stage_result.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
                summary_rows.append(stage_result)
                write_summary(summary_path, summary_rows)
                raise
            summary_rows.append(stage_result)
            write_summary(summary_path, summary_rows)
    print(f"wrote {rel_project_path(summary_path)}")
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and summarize HEST coverage95 generalization tasks.")
    parser.add_argument("--task-table", default=DEFAULT_TASK_TABLE)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--checkpoint-root", default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--stages", nargs="+", choices=["sf", "expression", "combined"], default=["sf"])
    parser.add_argument("--split-types", nargs="*", default=None)
    parser.add_argument("--task-slugs", nargs="*", default=None)
    parser.add_argument("--heldouts", nargs="*", default=None)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--sf-epochs", type=int, default=None)
    parser.add_argument("--expression-epochs", type=int, default=None)
    parser.add_argument("--splits", nargs="*", default=["test"])
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_table = resolve_project_path(args.task_table)
    out_dir = resolve_project_path(args.out_dir)
    checkpoint_root = resolve_project_path(args.checkpoint_root)
    if task_table is None or out_dir is None or checkpoint_root is None:
        raise ValueError("task-table, out-dir, and checkpoint-root must resolve to paths.")
    run_generalization_tasks(
        task_table=task_table,
        out_dir=out_dir,
        checkpoint_root=checkpoint_root,
        run_name=args.run_name or run_name_default(),
        stages=[str(x) for x in args.stages],
        split_types=None if args.split_types is None else [str(x) for x in args.split_types],
        task_slugs=None if args.task_slugs is None else [str(x) for x in args.task_slugs],
        heldouts=None if args.heldouts is None else [str(x) for x in args.heldouts],
        max_tasks=args.max_tasks,
        device=args.device,
        sf_epochs=args.sf_epochs,
        expression_epochs=args.expression_epochs,
        splits=[str(x) for x in args.splits],
        skip_existing=bool(args.skip_existing),
    )


if __name__ == "__main__":
    main()
