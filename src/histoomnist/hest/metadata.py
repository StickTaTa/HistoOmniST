from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_METADATA_COLUMNS = {
    "dataset_title",
    "id",
    "image_filename",
    "organ",
    "species",
    "patient",
    "st_technology",
    "spots_under_tissue",
    "nb_genes",
}


def load_hest_metadata(path: str | Path) -> pd.DataFrame:
    """Load and validate the HEST metadata CSV."""

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"HEST metadata CSV not found: {p}")
    df = pd.read_csv(p)
    missing = sorted(REQUIRED_METADATA_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"HEST metadata missing required columns: {missing}")
    df["id"] = df["id"].astype(str)
    df["spots_under_tissue"] = pd.to_numeric(df["spots_under_tissue"], errors="coerce")
    df["nb_genes"] = pd.to_numeric(df["nb_genes"], errors="coerce")
    return df


def filter_hest_metadata(
    df: pd.DataFrame,
    *,
    species: str = "Homo sapiens",
    st_technology: str = "Visium",
    min_spots_under_tissue: int = 200,
) -> pd.DataFrame:
    """Return the HEST subset used for the first SF training stage."""

    keep = (
        df["species"].eq(species)
        & df["st_technology"].eq(st_technology)
        & (df["spots_under_tissue"] >= int(min_spots_under_tissue))
    )
    out = df.loc[keep].copy()
    out = out.sort_values(["organ", "dataset_title", "id"], kind="stable").reset_index(drop=True)
    return out


def summarize_hest_metadata(df: pd.DataFrame, filtered: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build summary tables for reports."""

    return {
        "species_counts": df["species"].value_counts(dropna=False).rename_axis("species").reset_index(name="n_slides"),
        "human_platform_counts": df.loc[df["species"].eq("Homo sapiens"), "st_technology"]
        .value_counts(dropna=False)
        .rename_axis("st_technology")
        .reset_index(name="n_slides"),
        "filtered_organ_counts": filtered["organ"]
        .value_counts(dropna=False)
        .rename_axis("organ")
        .reset_index(name="n_slides"),
        "filtered_dataset_counts": filtered["dataset_title"]
        .value_counts(dropna=False)
        .rename_axis("dataset_title")
        .reset_index(name="n_slides"),
        "filtered_spot_stats_by_organ": filtered.groupby("organ", dropna=False)["spots_under_tissue"]
        .agg(["count", "sum", "median", "min", "max"])
        .reset_index(),
    }


def write_metadata_report(
    *,
    report_path: str | Path,
    metadata_path: str | Path,
    df: pd.DataFrame,
    filtered: pd.DataFrame,
    summaries: dict[str, pd.DataFrame],
    species: str,
    st_technology: str,
    min_spots_under_tissue: int,
) -> None:
    """Write a compact Markdown audit report."""

    def table(df: pd.DataFrame, max_rows: int = 20) -> str:
        if df.empty:
            return "_No rows._"
        view = df.head(max_rows).fillna("")
        headers = [str(col) for col in view.columns]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for _, row in view.iterrows():
            lines.append("| " + " | ".join(str(row[col]) for col in view.columns) + " |")
        if len(df) > max_rows:
            lines.append(f"\n_Only first {max_rows} of {len(df)} rows shown._")
        return "\n".join(lines)

    out = Path(report_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# HEST-1k Metadata Audit",
        "",
        f"- Metadata CSV: `{Path(metadata_path)}`",
        f"- Total slides: {len(df)}",
        f"- Selected slides: {len(filtered)}",
        f"- Filter: species=`{species}`, st_technology=`{st_technology}`, min_spots_under_tissue=`{min_spots_under_tissue}`",
        "",
        "## Species",
        "",
        table(summaries["species_counts"]),
        "",
        "## Human Platforms",
        "",
        table(summaries["human_platform_counts"]),
        "",
        "## Selected Organs",
        "",
        table(summaries["filtered_organ_counts"]),
        "",
        "## Notes",
        "",
        "- This audit only checks metadata. Raw HEST assets and processed HIPT/count files are checked by `hest_build_manifest.py`.",
        "- The committed SF target remains mean-one: `sf_i = total_i / mean(total_valid_spots_in_slide)`.",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
