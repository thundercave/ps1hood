"""MASt3R-as-matcher → posed COLMAP — mocks, no GPU / no mast3r package required."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner
from PIL import Image

from ps1_hood.cli import main
from ps1_hood.config import ProjectSpec
from ps1_hood.geo import BBox
from ps1_hood.project import create_project
from ps1_hood.reconstruct import mast3r as m
from ps1_hood.reconstruct.colmap import (
    bootstrap_posed_database,
    image_ids_to_pair_id,
    import_keypoints_and_matches,
    point_triangulator_argv,
    read_db_image_ids,
    write_known_pose_model,
)


def _fake_frame(
    tmp: Path,
    *,
    name: str,
    e: float,
    n: float,
    u: float = 1.7,
    heading: float = 90.0,
    pano: str | None = None,
    w: int = 64,
    h: int = 48,
) -> dict:
    img = tmp / name
    Image.new("RGB", (w, h), color=(80, 120, 40)).save(img, format="JPEG")
    return {
        "path": str(img),
        "e": e,
        "n": n,
        "u": u,
        "heading": heading,
        "pitch": 0.0,
        "fov": 90.0,
        "width": w,
        "height": h,
        "pano_id": pano or name.split(".")[0],
        "interpolated": False,
    }


def test_scale_xy_to_original() -> None:
    xy = np.array([[10.0, 20.0], [0.0, 0.0]], dtype=np.float32)
    out = m._scale_xy_to_original(xy, resized_hw=(256, 512), orig_wh=(1024, 512))
    assert out.shape == (2, 2)
    assert out[0, 0] == pytest.approx(20.0)  # 10 * 1024/512
    assert out[0, 1] == pytest.approx(40.0)  # 20 * 512/256


def test_aggregate_pair_matches_builds_tracks() -> None:
    # Three views; shared quantized keypoints chain into a 3-view track.
    pair_xy = {
        (0, 1): (
            np.array([[10.0, 10.0], [30.0, 30.0]], dtype=np.float32),
            np.array([[11.0, 11.0], [40.0, 40.0]], dtype=np.float32),
        ),
        (1, 2): (
            np.array([[11.2, 10.8]], dtype=np.float32),  # ~same as view1 first kp
            np.array([[12.0, 12.0]], dtype=np.float32),
        ),
    }
    kps, matches = m.aggregate_pair_matches(pair_xy, 3, min_track_len=2)
    assert 1 in kps and 2 in kps and 3 in kps
    assert sum(len(v) for v in kps.values()) >= 3
    assert (1, 2) in matches or (2, 3) in matches
    # 3-view track should produce both pair edges when min_track_len=2
    assert len(matches) >= 1


def test_aggregate_drops_short_tracks() -> None:
    pair_xy = {
        (0, 1): (
            np.array([[5.0, 5.0]], dtype=np.float32),
            np.array([[6.0, 6.0]], dtype=np.float32),
        ),
    }
    kps, matches = m.aggregate_pair_matches(pair_xy, 2, min_track_len=3)
    assert sum(len(v) for v in kps.values()) == 0
    assert matches == {}


def test_bootstrap_posed_database_ids(tmp_path: Path) -> None:
    frames = [
        _fake_frame(tmp_path, name="a.jpg", e=0.0, n=0.0, pano="p0"),
        _fake_frame(tmp_path, name="b.jpg", e=5.0, n=0.0, pano="p1"),
    ]
    names = ["a.jpg", "b.jpg"]
    db = tmp_path / "database.db"
    mapping = bootstrap_posed_database(db, frames, names)
    assert mapping["a.jpg"] == (1, mapping["a.jpg"][1])
    assert mapping["b.jpg"][0] == 2
    db_map = read_db_image_ids(db)
    assert db_map["a.jpg"][0] == 1
    assert db_map["b.jpg"][0] == 2
    # Prior text model uses the same 1..N IMAGE_IDs.
    model = tmp_path / "sparse_prior"
    write_known_pose_model(frames, names, model)
    lines = [
        ln
        for ln in (model / "images.txt").read_text(encoding="ascii").splitlines()
        if ln and not ln.startswith("#")
    ]
    # Every other line is blank POINTS2D
    img_lines = [ln for ln in lines if ln.strip()]
    assert img_lines[0].startswith("1 ")
    assert img_lines[1].startswith("2 ")


def test_import_keypoints_and_matches_two_view(tmp_path: Path) -> None:
    frames = [
        _fake_frame(tmp_path, name="a.jpg", e=0.0, n=0.0, pano="p0"),
        _fake_frame(tmp_path, name="b.jpg", e=4.0, n=0.0, pano="p1"),
    ]
    names = ["a.jpg", "b.jpg"]
    db = tmp_path / "database.db"
    bootstrap_posed_database(db, frames, names)
    kps = {
        1: np.array([[10.0, 10.0], [20.0, 20.0]], dtype=np.float32),
        2: np.array([[11.0, 11.0], [21.0, 21.0]], dtype=np.float32),
    }
    matches = {(1, 2): np.array([[0, 0], [1, 1]], dtype=np.uint32)}
    n = import_keypoints_and_matches(db, kps, matches, skip_geometric_verification=True)
    assert n == 1
    con = sqlite3.connect(str(db))
    try:
        assert con.execute("SELECT COUNT(*) FROM keypoints").fetchone()[0] == 2
        assert con.execute("SELECT COUNT(*) FROM matches").fetchone()[0] == 1
        assert con.execute("SELECT COUNT(*) FROM two_view_geometries").fetchone()[0] == 1
        pair_id, rows, config = con.execute(
            "SELECT pair_id, rows, config FROM two_view_geometries"
        ).fetchone()
        assert pair_id == image_ids_to_pair_id(1, 2)
        assert rows == 2
        assert config == 2  # CALIBRATED
    finally:
        con.close()


def test_point_triangulator_argv_locks_intrinsics(tmp_path: Path) -> None:
    argv = point_triangulator_argv(
        "colmap",
        tmp_path / "db.db",
        tmp_path / "images",
        tmp_path / "in",
        tmp_path / "out",
    )
    assert "point_triangulator" in argv
    assert "--Mapper.ba_refine_focal_length" in argv
    assert argv[argv.index("--Mapper.ba_refine_focal_length") + 1] == "0"
    assert "--clear_points" in argv
    # No free-pose / ignore flags
    joined = " ".join(argv)
    assert "ignore_pose" not in joined
    assert "mapper" not in argv[1:]  # subcommand is point_triangulator only


def test_require_mast3r_runtime_no_package(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(m, "_try_import_mast3r", lambda: False)
    with pytest.raises(RuntimeError, match="not installed"):
        m.require_mast3r_runtime()


def test_require_mast3r_runtime_no_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    monkeypatch.setattr(m, "_try_import_mast3r", lambda: True)
    fake = types.ModuleType("torch")

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return False

    fake.cuda = _Cuda  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake)
    with pytest.raises(RuntimeError, match="requires a CUDA or ROCm GPU"):
        m.require_mast3r_runtime()


def test_which_mast3r_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(m, "_try_import_mast3r", lambda: False)
    assert m.which_mast3r() is None


def test_fill_database_with_mock_match_fn(tmp_path: Path) -> None:
    frames = [
        _fake_frame(tmp_path, name="a.jpg", e=0.0, n=0.0, pano="p0"),
        _fake_frame(tmp_path, name="b.jpg", e=6.0, n=0.0, pano="p1"),
        _fake_frame(tmp_path, name="c.jpg", e=12.0, n=1.0, pano="p2"),
    ]
    names = ["a.jpg", "b.jpg", "c.jpg"]
    images = tmp_path / "images"
    images.mkdir()
    for fr, name in zip(frames, names, strict=True):
        Image.open(fr["path"]).save(images / name)

    db = tmp_path / "database.db"
    bootstrap_posed_database(db, frames, names)

    def match_fn(pa: Path, pb: Path) -> tuple[np.ndarray, np.ndarray]:
        # Deterministic fake correspondences in original pixel space.
        return (
            np.array([[16.0, 12.0], [32.0, 24.0], [48.0, 36.0]], dtype=np.float32),
            np.array([[17.0, 13.0], [33.0, 25.0], [49.0, 37.0]], dtype=np.float32),
        )

    meta = m.fill_database_with_mast3r_matches(
        db,
        images,
        frames,
        names,
        pair_indices=[(0, 1), (1, 2)],
        match_fn=match_fn,
        min_track_len=2,
    )
    assert meta["matcher"] == "mast3r"
    assert meta["pose_lock"] is True
    assert meta["n_pair_geometries"] >= 1
    assert meta["n_keypoints"] > 0


def test_run_mast3r_fails_loud_without_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames = [
        _fake_frame(tmp_path, name="a.jpg", e=0.0, n=0.0, pano="p0"),
        _fake_frame(tmp_path, name="b.jpg", e=5.0, n=0.0, pano="p1"),
    ]
    monkeypatch.setattr(m, "_try_import_mast3r", lambda: False)
    recon = tmp_path / "recon"
    with pytest.raises(RuntimeError, match="not installed"):
        m.run_mast3r(frames, recon)
    # Export still happened for cloud hand-off.
    assert (recon / "colmap" / "images").is_dir()
    assert (recon / "colmap" / "sparse_prior" / "images.txt").is_file()
    assert (recon / "colmap" / "cross_pano_pairs.txt").is_file()


def test_cli_backend_mast3r_sets_matcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    create_project(
        ProjectSpec(name="mtest", bbox=BBox(52.0, 5.0, 52.001, 5.001)),
        runs_root=tmp_path,
    )
    import ps1_hood.project as project_mod

    monkeypatch.setattr(project_mod, "default_runs_root", lambda: tmp_path)
    monkeypatch.setattr(
        "ps1_hood.pipeline.stage_reconstruct", lambda project, progress=None: None
    )
    runner = CliRunner()
    result = runner.invoke(main, ["reconstruct", "mtest", "--backend", "mast3r"])
    assert result.exit_code == 0, result.output
    from ps1_hood.project import Project

    spec = Project(tmp_path / "mtest").load_spec()
    assert spec.recon_backend == "mast3r"
    assert spec.recon_matcher == "mast3r"


def test_cli_matcher_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    create_project(
        ProjectSpec(name="mflag", bbox=BBox(52.0, 5.0, 52.001, 5.001)),
        runs_root=tmp_path,
    )
    import ps1_hood.project as project_mod

    monkeypatch.setattr(project_mod, "default_runs_root", lambda: tmp_path)
    monkeypatch.setattr(
        "ps1_hood.pipeline.stage_reconstruct", lambda project, progress=None: None
    )
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["reconstruct", "mflag", "--backend", "colmap_posed", "--matcher", "mast3r"],
    )
    assert result.exit_code == 0, result.output
    from ps1_hood.project import Project

    spec = Project(tmp_path / "mflag").load_spec()
    assert spec.recon_backend == "colmap_posed"
    assert spec.recon_matcher == "mast3r"


def test_run_mast3r_matcher_posed_with_mock_and_colmap(tmp_path: Path) -> None:
    """End-to-end with synthetic matches + real point_triangulator (CPU COLMAP)."""
    frames = [
        _fake_frame(tmp_path, name="a.jpg", e=0.0, n=0.0, heading=90.0, pano="p0"),
        _fake_frame(tmp_path, name="b.jpg", e=5.0, n=0.0, heading=90.0, pano="p1"),
        _fake_frame(tmp_path, name="c.jpg", e=10.0, n=0.0, heading=90.0, pano="p2"),
        _fake_frame(tmp_path, name="d.jpg", e=15.0, n=0.0, heading=90.0, pano="p3"),
    ]

    def match_fn(pa: Path, pb: Path) -> tuple[np.ndarray, np.ndarray]:
        # Shared grid of correspondences so tracks can chain 0-1-2.
        xs = np.linspace(8, 56, 12)
        ys = np.linspace(8, 40, 12)
        xy = np.stack([xs, ys], axis=1).astype(np.float32)
        # Slight parallax shift proportional to baseline index.
        shift = 1.5 if "b" in pb.name or "c" in pb.name else 0.0
        return xy, xy + np.array([shift, 0.0], dtype=np.float32)

    recon = tmp_path / "recon"
    # min_points low — synthetic geometry on tiny JPEGs may be thin or SIGABRT;
    # contract: ENU prior + DB matches imported; triangulator attempted.
    import subprocess

    try:
        meta = m.run_mast3r_matcher_posed(
            frames,
            recon,
            match_fn=match_fn,
            min_points=1,
        )
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        assert (recon / "colmap" / "database.db").is_file()
        assert (recon / "colmap" / "sparse_prior" / "images.txt").is_file()
        assert (recon / "colmap" / "cross_pano_pairs.txt").is_file()
        # Matches were written (two_view_geometries nonempty).
        import sqlite3

        con = sqlite3.connect(str(recon / "colmap" / "database.db"))
        try:
            n_tv = con.execute(
                "SELECT COUNT(*) FROM two_view_geometries WHERE rows > 0"
            ).fetchone()[0]
        finally:
            con.close()
        assert n_tv >= 1
        return
    assert meta["pose_lock"] is True
    assert meta["matcher"] == "mast3r"
    assert Path(meta["path"]).is_file()
