import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_public_release.py"


def load_module():
    spec = importlib.util.spec_from_file_location("audit_public_release", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_audit_rejects_blocked_directories_and_absolute_paths(tmp_path: Path) -> None:
    module = load_module()
    blocked = tmp_path / "results" / "prediction.csv"
    blocked.parent.mkdir()
    blocked.write_text("value\n1\n", encoding="utf-8")
    source = tmp_path / "scripts" / "run.py"
    source.parent.mkdir()
    local_path = "G:" + "\\\\private\\\\data"
    source.write_text(f"ROOT = r'{local_path}'\n", encoding="utf-8")

    issues = module.audit_paths(tmp_path, [Path("results/prediction.csv"), Path("scripts/run.py")])

    assert any("blocked path" in issue for issue in issues)
    assert any("absolute local path" in issue for issue in issues)


def test_audit_accepts_small_portable_source_files(tmp_path: Path) -> None:
    module = load_module()
    source = tmp_path / "src" / "package.py"
    source.parent.mkdir()
    source.write_text("VALUE = 1\n", encoding="utf-8")

    assert module.audit_paths(tmp_path, [Path("src/package.py")]) == []


def test_audit_does_not_treat_https_urls_as_local_drive_paths(tmp_path: Path) -> None:
    module = load_module()
    citation = tmp_path / "CITATION.cff"
    citation.write_text(
        'repository-code: "https://github.com/StickTaTa/HistoOmniST"\n',
        encoding="utf-8",
    )

    assert module.audit_paths(tmp_path, [Path("CITATION.cff")]) == []
