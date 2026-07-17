import json
import importlib.util
from pathlib import Path

import yaml


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


def test_software_patch_uses_the_unchanged_v0_1_0_model_release() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    manifest = json.loads(
        (ROOT / "models" / "release_manifest.json").read_text(encoding="utf-8")
    )

    assert 'version = "0.1.1"' in pyproject
    assert citation["version"] == "0.1.1"
    assert str(citation["date-released"]) == "2026-07-17"
    assert manifest["release"] == "v0.1.0"

    for asset in manifest["assets"]:
        assert asset["filename"].endswith("_v0.1.0.pt")
        assert "/releases/download/v0.1.0/" in asset["url"]
        assert asset["url"].endswith(asset["filename"])


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


def test_release_manifest_uses_the_original_hipt_project() -> None:
    path = ROOT / "models" / "release_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    dependencies = {item["id"]: item for item in manifest["external_dependencies"]}
    hipt = dependencies["hipt_vit256"]

    assert hipt["source"] == "https://github.com/mahmoodlab/HIPT"
    assert hipt["source_revision"] == "780fafaed2e5b112bc1ed6e78852af1fe6714342"
    assert hipt["source_file"]["target"] == (
        "third_party/HIPT/1-Hierarchical-Pretraining/vision_transformer.py"
    )
    assert hipt["weights"]["target"] == (
        "third_party/HIPT/HIPT_4K/Checkpoints/vit256_small_dino.pth"
    )
    assert hipt["weights"]["bytes"] == 704_238_867
    assert hipt["weights"]["sha256"] == (
        "6960cd5a8657dc8bb214671aa0c6dbd3f5b698e84386884955836487ddc89e24"
    )


def test_breast_xenium_example_records_the_public_10x_asset() -> None:
    path = ROOT / "examples" / "breast_xenium" / "example_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    asset = manifest["asset"]

    assert manifest["provider"] == "10x Genomics"
    assert asset["filename"] == "Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image.ome.tif"
    assert asset["target"] == (
        "data/examples/breast_xenium/"
        "Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image.ome.tif"
    )
    assert asset["url"].startswith("https://cf.10xgenomics.com/samples/xenium/1.0.1/")
    assert asset["bytes"] == 1_427_110_955
    assert asset["sha256"] == (
        "3c2bd89588f97e886dc1288e68d4eaae1b21231cd59774bc01d2d7e3876430f0"
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


def test_public_wsi_cli_defaults_to_the_original_hipt_layout() -> None:
    script = ROOT / "scripts" / "histoomnist_predict_uploaded_wsi.py"
    spec = importlib.util.spec_from_file_location("histoomnist_predict_uploaded_wsi", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.DEFAULT_HIPT_SOURCE == (
        ROOT / "third_party" / "HIPT" / "1-Hierarchical-Pretraining"
    )
    assert module.DEFAULT_HIPT_WEIGHTS == (
        ROOT / "third_party" / "HIPT" / "HIPT_4K" / "Checkpoints" / "vit256_small_dino.pth"
    )
