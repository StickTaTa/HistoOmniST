from __future__ import annotations

from pathlib import Path


def write_markdown_report(path: str | Path, title: str, sections: list[tuple[str, str]]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    parts = [f"# {title}", ""]
    for heading, body in sections:
        parts.extend([f"## {heading}", "", body.strip(), ""])
    out.write_text("\n".join(parts), encoding="utf-8")
    return out
