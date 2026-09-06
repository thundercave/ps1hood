from __future__ import annotations

import json
from pathlib import Path

from ps1_hood.project import Project
from ps1_hood.reconstruct.keyframes import load_keyframes


def _write_jpeg(path: Path) -> None:
    # Minimal valid-ish JPEG bytes are unnecessary; empty file is enough for
    # Path.is_file(), and load_keyframes only checks existence.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake")


def _cam(
    tmp: Path,
    *,
    name: str,
    pitch: float,
    e: float = 0.0,
    n: float = 0.0,
    missing: bool = False,
) -> dict:
    shot = tmp / "cropped" / f"{name}.jpg"
    if not missing:
        _write_jpeg(shot)
    return {
        "pano_id": f"pano-{name}",
        "shot_path": str(shot),
        "mask": str(tmp / "cropped" / f"{name}.mask.png"),
        "e": e,
        "n": n,
        "u": 2.5,
        "heading": 90.0,
        "pitch": pitch,
        "fov": 90.0,
    }


def test_load_keyframes_prefers_near_horizon(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    (root / "project.yaml").write_text("name: t\nbbox: {south: 0, west: 0, north: 1, east: 1}\n")
    align = root / "align"
    align.mkdir()
    cams = [
        _cam(root, name="down", pitch=-30.0, e=0.0),
        _cam(root, name="a", pitch=0.0, e=1.0),
        _cam(root, name="b", pitch=2.0, e=2.0),
        _cam(root, name="up", pitch=18.0, e=3.0),
        _cam(root, name="gone", pitch=0.0, e=4.0, missing=True),
    ]
    (align / "cameras.json").write_text(json.dumps(cams), encoding="utf-8")

    frames = load_keyframes(Project(root))
    assert len(frames) == 2
    assert {f["path"] for f in frames} == {
        str(root / "cropped" / "a.jpg"),
        str(root / "cropped" / "b.jpg"),
    }
    assert all(abs(f["pitch"]) <= 5 for f in frames)
    assert frames[0]["mask"] is not None
    assert frames[0]["pano_id"].startswith("pano-")
    assert set(frames[0]) >= {"path", "e", "n", "u", "heading", "pitch", "fov", "mask", "pano_id"}


def test_load_keyframes_keeps_all_if_too_few_near_horizon(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    (root / "project.yaml").write_text("name: t\nbbox: {south: 0, west: 0, north: 1, east: 1}\n")
    align = root / "align"
    align.mkdir()
    cams = [
        _cam(root, name="down", pitch=-30.0, e=0.0),
        _cam(root, name="flat", pitch=0.0, e=1.0),
        _cam(root, name="up", pitch=18.0, e=2.0),
    ]
    (align / "cameras.json").write_text(json.dumps(cams), encoding="utf-8")

    frames = load_keyframes(Project(root))
    # only one near-horizon → keep all existing
    assert len(frames) == 3


def test_load_keyframes_max_views_subsample(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    (root / "project.yaml").write_text("name: t\nbbox: {south: 0, west: 0, north: 1, east: 1}\n")
    align = root / "align"
    align.mkdir()
    cams = [_cam(root, name=f"v{i}", pitch=0.0, e=float(i)) for i in range(6)]
    (align / "cameras.json").write_text(json.dumps(cams), encoding="utf-8")

    frames = load_keyframes(Project(root), max_views=3)
    assert len(frames) == 3
    # evenly spaced includes first + last
    assert frames[0]["e"] == 0.0
    assert frames[-1]["e"] == 5.0
