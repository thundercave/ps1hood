from __future__ import annotations

from pathlib import Path

from ps1_hood.config import ProjectSpec
from ps1_hood.geo import BBox
from ps1_hood.project import create_project, open_project


def test_create_and_reload(tmp_path: Path) -> None:
    spec = ProjectSpec(
        name="unit",
        bbox=BBox(52.0, 5.0, 52.01, 5.01),
        source="google_static",
    )
    project = create_project(spec, runs_root=tmp_path)
    assert (project.root / "project.yaml").is_file()
    loaded = open_project("unit", runs_root=tmp_path).load_spec()
    assert loaded.name == "unit"
    assert loaded.bbox.south == 52.0
    assert loaded.source == "google_static"
