from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from ps1_hood.capture.discover import cap_panos, subsample_evenly
from ps1_hood.cli import main
from ps1_hood.config import ProjectSpec
from ps1_hood.geo import BBox
from ps1_hood.project import open_project


def test_subsample_evenly_keeps_ends_and_count() -> None:
    items = list(range(10))
    out = subsample_evenly(items, 4)
    assert out[0] == 0
    assert out[-1] == 9
    assert len(out) == 4
    assert out == [0, 3, 6, 9]


def test_subsample_evenly_noop_when_under_cap() -> None:
    items = [1, 2, 3]
    assert subsample_evenly(items, 8) == [1, 2, 3]
    assert subsample_evenly(items, 0) == [1, 2, 3]


def test_cap_panos_logs_contract() -> None:
    panos = [{"pano_id": f"p{i}"} for i in range(20)]
    capped, max_n = cap_panos(panos, 8)
    assert max_n == 8
    assert len(capped) == 8
    assert capped[0]["pano_id"] == "p0"
    assert capped[-1]["pano_id"] == "p19"
    untouched, none_max = cap_panos(panos, None)
    assert none_max is None
    assert len(untouched) == 20


def test_project_spec_roundtrips_max_panos_and_pitches_alias(tmp_path: Path) -> None:
    spec = ProjectSpec(
        name="budget",
        bbox=BBox(52.0, 5.0, 52.01, 5.01),
        max_panos=8,
        extra_pitches=[0],
    )
    data = spec.to_dict()
    assert data["max_panos"] == 8
    assert data["extra_pitches"] == [0]
    loaded = ProjectSpec.from_dict(data)
    assert loaded.max_panos == 8
    assert loaded.extra_pitches == [0]

    via_alias = ProjectSpec.from_dict(
        {
            "name": "alias",
            "bbox": {"south": 52.0, "west": 5.0, "north": 52.01, "east": 5.01},
            "pitches": [0],
            "max_panos": 5,
        }
    )
    assert via_alias.extra_pitches == [0]
    assert via_alias.max_panos == 5


def test_init_smoke_preset_defaults(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "init",
            "smoke-demo",
            "--south",
            "52.0895",
            "--west",
            "5.1192",
            "--north",
            "52.0900",
            "--east",
            "5.1200",
            "--preset",
            "smoke",
        ],
    )
    assert result.exit_code == 0, result.output
    spec = open_project("smoke-demo", runs_root=tmp_path / "runs").load_spec()
    assert spec.spacing_m == 25.0
    assert spec.heading_step == 90
    assert spec.interp_steps == 3
    assert spec.max_panos == 8
    assert spec.extra_pitches == [0]


def test_init_smoke_keeps_spacing_override(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "init",
            "smoke-ov",
            "--south",
            "52.0895",
            "--west",
            "5.1192",
            "--north",
            "52.0900",
            "--east",
            "5.1200",
            "--preset",
            "smoke",
            "--spacing",
            "40",
            "--max-panos",
            "4",
        ],
    )
    assert result.exit_code == 0, result.output
    spec = open_project("smoke-ov", runs_root=tmp_path / "runs").load_spec()
    assert spec.spacing_m == 40.0
    assert spec.max_panos == 4
    assert spec.heading_step == 90
    assert spec.extra_pitches == [0]
