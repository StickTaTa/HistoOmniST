from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from histoomnist.utils.project_paths import load_yaml, resolve_project_path


DEFAULT_STYLE = {
    "dpi": 600,
    "formats": ["png", "pdf", "svg"],
    "font_family": "Arial",
    "font_fallback": ["Arial", "Helvetica", "DejaVu Sans"],
    "title_fontsize": 9,
    "axis_label_fontsize": 8,
    "tick_fontsize": 7,
    "legend_fontsize": 7,
    "axes_linewidth": 0.8,
    "tick_width": 0.7,
    "text_color": "#202020",
}


def load_plot_style(config: str | Path | None = None) -> dict:
    if config is None:
        path = resolve_project_path("configs/plot_style.yaml")
    else:
        path = resolve_project_path(config)
    if path is not None and path.exists():
        style = DEFAULT_STYLE | load_yaml(path)
    else:
        style = dict(DEFAULT_STYLE)
    return style


def apply_plot_style(config: str | Path | dict | None = None) -> dict:
    style = load_plot_style(config) if not isinstance(config, dict) else (DEFAULT_STYLE | config)
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": style.get("font_fallback", ["Arial", "Helvetica", "DejaVu Sans"]),
            "font.size": int(style.get("axis_label_fontsize", 8)),
            "axes.titlesize": int(style.get("title_fontsize", 9)),
            "axes.labelsize": int(style.get("axis_label_fontsize", 8)),
            "xtick.labelsize": int(style.get("tick_fontsize", 7)),
            "ytick.labelsize": int(style.get("tick_fontsize", 7)),
            "legend.fontsize": int(style.get("legend_fontsize", 7)),
            "axes.linewidth": float(style.get("axes_linewidth", 0.8)),
            "xtick.major.width": float(style.get("tick_width", 0.7)),
            "ytick.major.width": float(style.get("tick_width", 0.7)),
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": int(style.get("dpi", 600)),
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "text.color": style.get("text_color", "#202020"),
            "axes.edgecolor": style.get("text_color", "#202020"),
            "xtick.color": style.get("text_color", "#202020"),
            "ytick.color": style.get("text_color", "#202020"),
        }
    )
    return style


def save_figure(
    fig: plt.Figure,
    out_dir: str | Path,
    name: str,
    style: dict | None = None,
    source_data: pd.DataFrame | None = None,
    source_data_dir: str | Path | None = None,
) -> list[Path]:
    style = DEFAULT_STYLE | (style or {})
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for ext in style.get("formats", ["png", "pdf", "svg"]):
        path = out / f"{name}.{ext}"
        kwargs = {"bbox_inches": "tight"}
        if ext.lower() == "png":
            kwargs["dpi"] = int(style.get("dpi", 600))
        fig.savefig(path, **kwargs)
        saved.append(path)
    if source_data is not None and style.get("save_source_data", True):
        data_dir = Path(source_data_dir) if source_data_dir else out / "source_data"
        data_dir.mkdir(parents=True, exist_ok=True)
        source_path = data_dir / f"{name}_source_data.csv"
        source_data.to_csv(source_path, index=False)
        saved.append(source_path)
    return saved

