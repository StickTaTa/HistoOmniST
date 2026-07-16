import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def notebook_source(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_public_notebooks_use_python_apis_instead_of_command_wrappers() -> None:
    for path in sorted((ROOT / "notebooks").glob("0*.ipynb")):
        source = notebook_source(path)
        assert "subprocess.run(" not in source, path.name


def test_quickstart_is_a_real_breast_xenium_count_scale_workflow() -> None:
    source = notebook_source(ROOT / "notebooks" / "00_quickstart_inference.ipynb")

    assert "Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image.ome.tif" in source
    assert "rgb_tissue_mask" in source
    assert "hipt256_features" in source
    assert "features_for_checkpoint" in source
    assert "predict_rate_selected" in source
    assert "predict_sf" in source
    assert "add_count_columns" in source
    assert "count_log1p_" in source


def test_public_guides_do_not_couple_hipt_to_istar() -> None:
    paths = [
        ROOT / "README.md",
        ROOT / "THIRD_PARTY_NOTICES.md",
        ROOT / "docs/installation.md",
        ROOT / "docs/inference.md",
        ROOT / "docs/training.md",
        *(ROOT / "notebooks").glob("0*.ipynb"),
    ]
    blocked = [
        "third_party/benchmarks/iStar",
        "JWonderLand/HIPT_unofficial",
        "HIPT ViT-256 implementation distributed with iStar",
        "iStar/HIPT upstream ecosystem",
    ]

    for path in paths:
        text = path.read_text(encoding="utf-8")
        for phrase in blocked:
            assert phrase not in text, f"{phrase!r} remains in {path.name}"
