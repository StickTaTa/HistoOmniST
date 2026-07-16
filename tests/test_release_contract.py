import json
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_GENES = [
    "EPCAM",
    "KRT8",
    "CD3D",
    "CD3E",
    "CD8A",
    "CD4",
    "LST1",
    "AIF1",
    "C1QA",
    "C1QB",
    "COL1A1",
    "COL1A2",
    "DCN",
    "LUM",
    "ACTA2",
    "MKI67",
    "TOP2A",
    "PCNA",
    "CA9",
    "SLC2A1",
    "VIM",
    "TGFB1",
    "ZEB1",
    "SNAI2",
    "AXIN2",
    "LGR5",
    "MYC",
    "ASCL2",
]


def test_release_panel_matches_the_public_atlas_contract() -> None:
    path = ROOT / "configs" / "manuscript_release_28_gene_panel.json"
    config = json.loads(path.read_text(encoding="utf-8"))

    assert config["selected_genes"] == EXPECTED_GENES
    assert len(config["selected_genes"]) == len(set(config["selected_genes"])) == 28
    assert list(config["programs"]) == [
        "epithelial",
        "t_cell",
        "myeloid",
        "stromal",
        "proliferation",
        "hypoxia",
        "emt_tgfb",
        "wnt_crc",
    ]
    gene_set = set(config["selected_genes"])
    assert all(set(members) <= gene_set for members in config["programs"].values())


def test_release_manifest_records_the_two_frozen_checkpoints() -> None:
    path = ROOT / "models" / "release_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assets = {asset["id"]: asset for asset in manifest["assets"]}

    assert set(assets) == {"expression_rate", "size_factor"}
    assert assets["expression_rate"]["bytes"] == 22_851_145
    assert assets["expression_rate"]["sha256"] == (
        "baf13faa87bcfe6f8c57d08ba24b12f2e11961d39be5c7cff45f6a4275a85362"
    )
    assert assets["size_factor"]["bytes"] == 50_879_405
    assert assets["size_factor"]["sha256"] == (
        "104bfb28ea2b3e93c70000418ee4aafda8923339440f5eee70a29c81c61f4d1e"
    )


def test_public_wsi_cli_defaults_to_the_release_panel() -> None:
    script = ROOT / "scripts" / "histoomnist_predict_uploaded_wsi.py"
    spec = importlib.util.spec_from_file_location("histoomnist_predict_uploaded_wsi", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.DEFAULT_TARGET_CONFIG == (
        ROOT / "configs" / "manuscript_release_28_gene_panel.json"
    )
