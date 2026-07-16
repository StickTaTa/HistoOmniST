from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 25 * 1024 * 1024
BLOCKED_PREFIXES = (
    "analysis/",
    "checkpoints/",
    "cleanup_manifests/",
    "data/tcga/",
    "data/external",
    "figures/",
    "logs/",
    "output/",
    "outputs/",
    "reports/",
    "results/",
    "runs/",
    "third_party/",
    "tmp/",
)
BLOCKED_NAMES = {"agents.md", "codex_histoomnist_conversations.md", "local_paths.yaml"}
TEXT_SUFFIXES = {
    ".cfg",
    ".cff",
    ".csv",
    ".ini",
    ".ipynb",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
ABSOLUTE_LOCAL_PATH = re.compile(
    r"(?i)(?<![a-z0-9])(?:[a-z]:[\\/]|/(?:home|mnt|users)/)"
)
SECRET_PATTERN = re.compile(
    r"(?i)(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)\s*[:=]\s*['\"][^'\"]+"
)


def audit_paths(root: Path, paths: Iterable[Path]) -> list[str]:
    issues: list[str] = []
    for relative in paths:
        normalized = relative.as_posix().lstrip("./")
        lowered = normalized.lower()
        path = root / relative
        if not path.is_file():
            continue
        if lowered in BLOCKED_NAMES or any(lowered.startswith(prefix) for prefix in BLOCKED_PREFIXES):
            issues.append(f"blocked path: {normalized}")
        if path.stat().st_size > MAX_FILE_BYTES:
            issues.append(f"file exceeds 25 MiB: {normalized}")
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {"LICENSE", ".gitignore", ".gitattributes"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if ABSOLUTE_LOCAL_PATH.search(text):
            issues.append(f"absolute local path: {normalized}")
        if SECRET_PATTERN.search(text):
            issues.append(f"credential-like value: {normalized}")
    return issues


def candidate_paths(root: Path) -> list[Path]:
    completed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return [Path(line) for line in completed.stdout.splitlines() if line.strip()]


def main() -> None:
    issues = audit_paths(ROOT, candidate_paths(ROOT))
    if issues:
        raise SystemExit("Public-release audit failed:\n- " + "\n- ".join(sorted(set(issues))))
    print("Public-release audit passed.")


if __name__ == "__main__":
    main()
