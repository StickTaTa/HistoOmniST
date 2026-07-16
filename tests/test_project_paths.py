from pathlib import Path

import histoomnist.utils.project_paths as project_paths


def test_missing_local_config_uses_portable_project_relative_defaults(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(project_paths, "project_root", lambda: tmp_path)

    paths = project_paths.load_local_paths(tmp_path / "missing.yaml")

    assert paths.old_project_root == tmp_path
    assert paths.new_project_root == tmp_path
    assert paths.manuscript_root == tmp_path
    assert paths.hest1k_root == tmp_path / "data" / "HEST-1k"
