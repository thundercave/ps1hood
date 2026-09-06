"""Known-pose COLMAP text model + cross-pano pair filtering."""

from __future__ import annotations

from pathlib import Path

import pytest

import numpy as np

from ps1_hood.reconstruct.colmap import (
    _focal_px,
    _rotmat_to_qvec,
    choose_colmap_cpu_gpu_flags,
    colmap_cpu_gpu_flags,
    cross_pano_pair_indices,
    feature_extractor_argv,
    filter_frames_registered_in_matches,
    frame_sizes_uniform,
    frames_have_known_poses,
    matches_importer_argv,
    mean_track_length,
    read_db_cameras,
    read_db_image_ids,
    remap_known_pose_model_to_db,
    write_cross_pano_match_list,
    write_known_pose_model,
)
from ps1_hood.geo import camera_rotation_cv



def _fr(e, n, heading, pano="a", path="/tmp/x.jpg", pitch=0.0):
    return {
        "e": e,
        "n": n,
        "u": 2.5,
        "heading": heading,
        "pitch": pitch,
        "fov": 90.0,
        "pano_id": pano,
        "path": path,
        "width": 640,
        "height": 480,
    }


def _insert_pinhole_camera(con, camera_id: int, width: int, height: int, *, fx=None):
    """Insert a COLMAP PINHOLE cameras row (model id 1, float64 params blob)."""
    import numpy as np

    if fx is None:
        fx = width / 2.0
    params = np.asarray([fx, fx, width / 2.0, height / 2.0], dtype=np.float64)
    con.execute(
        "CREATE TABLE IF NOT EXISTS cameras ("
        "camera_id INTEGER PRIMARY KEY, model INTEGER, width INTEGER, height INTEGER, "
        "params BLOB, prior_focal_length INTEGER)"
    )
    con.execute(
        "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
        (camera_id, 1, width, height, params.tobytes(), 0),
    )

def test_focal_from_hfov():
    # 90° → fx = W/2
    assert abs(_focal_px(640, 90.0) - 320.0) < 1e-6


def test_t_is_neg_R_C_not_center():
    fr = _fr(10.0, 20.0, 0.0)
    R_cw = np.array(camera_rotation_cv(fr["heading"], fr["pitch"]), dtype=np.float64).T
    C = np.array([fr["e"], fr["n"], fr["u"]])
    t = -R_cw @ C
    # Must not equal camera centre
    assert not np.allclose(t, C)
    # Round-trip: C = -R^T t
    C2 = -R_cw.T @ t
    assert np.allclose(C, C2, atol=1e-9)


def test_qvec_unit():
    R = np.array(camera_rotation_cv(45.0, 10.0), dtype=np.float64).T
    q = _rotmat_to_qvec(R)
    assert abs(np.linalg.norm(q) - 1.0) < 1e-9
    assert q[0] >= 0


def test_cross_pano_skips_orbit_mates():
    frames = [
        _fr(0, 0, 0, "p1"),
        _fr(0, 0, 90, "p1"),  # same center — pure rotation
        _fr(8, 0, 5, "p2"),
        _fr(8, 0, 95, "p2"),
    ]
    pairs = cross_pano_pair_indices(frames)
    for i, j in pairs:
        assert frames[i]["pano_id"] != frames[j]["pano_id"]
    assert (0, 2) in pairs or (0, 2) in [(a, b) for a, b in pairs]


def test_write_known_pose_model_pinhole(tmp_path: Path):
    frames = [_fr(0, 0, 0, "p1"), _fr(8, 1, 10, "p2")]
    names = ["p1_h000.jpg", "p2_h010.jpg"]
    model = write_known_pose_model(frames, names, tmp_path / "sparse")
    cams = (model / "cameras.txt").read_text()
    assert "PINHOLE" in cams
    assert "SIMPLE_PINHOLE" not in cams
    imgs = (model / "images.txt").read_text().splitlines()
    data = [ln for ln in imgs if ln and not ln.startswith("#")]
    # every other line empty → one pose line per image
    pose_lines = [ln for ln in data if ln.strip()]
    assert len(pose_lines) == 2
    assert (model / "points3D.txt").is_file()
    assert frames_have_known_poses(frames)


def test_remap_image_ids(tmp_path: Path):
    import sqlite3

    frames = [_fr(0, 0, 0, "p1"), _fr(8, 0, 0, "p2")]
    names = ["aaa.jpg", "bbb.jpg"]
    model = write_known_pose_model(frames, names, tmp_path / "sparse")
    db = tmp_path / "database.db"
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT, camera_id INTEGER)"
    )
    # Deliberately non-sequential / offset ids
    con.execute("INSERT INTO images VALUES (7, 'aaa.jpg', 1)")
    con.execute("INSERT INTO images VALUES (9, 'bbb.jpg', 1)")
    _insert_pinhole_camera(con, 1, 640, 480)
    con.commit()
    con.close()
    path, kept_frames, kept_names = remap_known_pose_model_to_db(model, frames, names, db)
    assert path == model
    assert kept_names == names
    assert len(kept_frames) == 2
    text = (model / "images.txt").read_text()
    assert text.splitlines()[3].startswith("7 ") or any(
        ln.startswith("7 ") for ln in text.splitlines()
    )
    assert any(ln.startswith("9 ") for ln in text.splitlines())
    mp = read_db_image_ids(db)
    assert mp["aaa.jpg"] == (7, 1)


def test_match_list_cross_pano(tmp_path: Path):
    frames = [
        _fr(0, 0, 0, "p1"),
        _fr(0, 0, 90, "p1"),
        _fr(8, 0, 5, "p2"),
    ]
    names = ["a.jpg", "b.jpg", "c.jpg"]
    n = write_cross_pano_match_list(frames, names, tmp_path / "pairs.txt")
    lines = (tmp_path / "pairs.txt").read_text().strip().splitlines()
    assert n == len(lines)
    for ln in lines:
        assert "a.jpg b.jpg" not in ln  # same pano excluded

def test_remap_filters_missing_db_images(tmp_path: Path):
    import sqlite3

    frames = [
        _fr(0, 0, 0, "p1"),
        _fr(8, 0, 0, "p2"),
        _fr(16, 0, 0, "p3"),
        _fr(24, 0, 5, "p4"),
        _fr(32, 0, 5, "p5"),
    ]
    names = ["a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg"]
    model = write_known_pose_model(frames, names, tmp_path / "sparse")
    db = tmp_path / "database.db"
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT, camera_id INTEGER)"
    )
    # Omit one image (20% would hard-fail; 1/5=20% > 10%). Use 10 frames so 1 missing = 10%.
    con.execute("INSERT INTO images VALUES (1, 'a.jpg', 1)")
    con.execute("INSERT INTO images VALUES (2, 'b.jpg', 1)")
    con.execute("INSERT INTO images VALUES (3, 'c.jpg', 2)")
    con.execute("INSERT INTO images VALUES (4, 'd.jpg', 2)")
    # e.jpg missing
    con.commit()
    con.close()

    # 1/5 = 20% > 10% → hard fail
    try:
        remap_known_pose_model_to_db(model, frames, names, db)
        raise AssertionError("expected RuntimeError for >10% missing")
    except RuntimeError as exc:
        assert "missing" in str(exc).lower()

    # 10 frames, 1 missing → 10% exactly allowed (frac > max, so use 0.10 and 1/11)
    frames10 = [_fr(i * 8, 0, 0 if i % 2 == 0 else 5, f"p{i}") for i in range(10)]
    # ensure cross-pano: alternate pano ids with baseline
    frames10 = []
    names10 = []
    for i in range(10):
        frames10.append(_fr((i // 2) * 8, 0, (i % 2) * 90, f"pano{i // 2}"))
        names10.append(f"img{i}.jpg")
    model2 = write_known_pose_model(frames10, names10, tmp_path / "sparse2")
    db2 = tmp_path / "database2.db"
    con = sqlite3.connect(db2)
    con.execute(
        "CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT, camera_id INTEGER)"
    )
    for i, n in enumerate(names10):
        if n == "img9.jpg":
            continue  # 1/10 = 10% — frac > 0.10 is False for ==? 0.1 > 0.1 is False
        con.execute("INSERT INTO images VALUES (?, ?, 1)", (i + 1, n))
    _insert_pinhole_camera(con, 1, 640, 480)
    con.commit()
    con.close()
    path, kept_f, kept_n = remap_known_pose_model_to_db(model2, frames10, names10, db2)
    assert "img9.jpg" not in kept_n
    assert len(kept_n) == 9
    assert len(kept_f) == 9
    text_imgs = (path / "images.txt").read_text()
    assert "img9.jpg" not in text_imgs
    assert "img0.jpg" in text_imgs


def test_remap_hard_fails_when_too_few_remain(tmp_path: Path):
    import sqlite3

    frames = [_fr(0, 0, 0, "p1"), _fr(8, 0, 0, "p2"), _fr(16, 0, 0, "p3")]
    names = ["a.jpg", "b.jpg", "c.jpg"]
    model = write_known_pose_model(frames, names, tmp_path / "sparse")
    db = tmp_path / "database.db"
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT, camera_id INTEGER)"
    )
    con.execute("INSERT INTO images VALUES (1, 'a.jpg', 1)")
    con.commit()
    con.close()
    try:
        remap_known_pose_model_to_db(model, frames, names, db)
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "missing" in str(exc).lower()


def test_feature_extractor_argv_uniform_uses_single_camera(monkeypatch):
    monkeypatch.setattr(
        "ps1_hood.reconstruct.colmap.colmap_cpu_gpu_flags",
        lambda _c: (("--SiftExtraction.use_gpu", "0"), ("--SiftMatching.use_gpu", "0")),
    )
    frames = [_fr(0, 0, 0, "p1"), _fr(8, 0, 0, "p2")]
    assert frame_sizes_uniform(frames)
    argv = feature_extractor_argv("colmap", Path("db.db"), Path("images"), frames)
    assert "--ImageReader.single_camera" in argv
    i = argv.index("--ImageReader.single_camera")
    assert argv[i + 1] == "1"
    assert "--ImageReader.camera_params" in argv
    assert "PINHOLE" in argv


def test_feature_extractor_argv_mixed_sizes_omits_global_params(monkeypatch):
    monkeypatch.setattr(
        "ps1_hood.reconstruct.colmap.colmap_cpu_gpu_flags",
        lambda _c: (("--FeatureExtraction.use_gpu", "0"), ("--FeatureMatching.use_gpu", "0")),
    )
    a = _fr(0, 0, 0, "p1")
    b = _fr(8, 0, 0, "p2")
    b["width"] = 1728  # differ from 640
    b["height"] = 1004
    assert not frame_sizes_uniform([a, b])
    argv = feature_extractor_argv("colmap", Path("db.db"), Path("images"), [a, b])
    assert "--FeatureExtraction.use_gpu" in argv
    assert argv[argv.index("--FeatureExtraction.use_gpu") + 1] == "0"
    assert "--ImageReader.single_camera" in argv
    i = argv.index("--ImageReader.single_camera")
    assert argv[i + 1] == "0"
    assert "--ImageReader.camera_params" not in argv
    assert argv[argv.index("--ImageReader.camera_model") + 1] == "PINHOLE"


def test_choose_colmap_cpu_gpu_flags_legacy_sift():
    feat = "  --SiftExtraction.use_gpu arg (=1)\n  --SiftExtraction.gpu_index arg (=-1)\n"
    match = "  --SiftMatching.use_gpu arg (=1)\n  --SiftMatching.max_ratio arg (=0.8)\n"
    extract, matching = choose_colmap_cpu_gpu_flags(feat, match)
    assert extract == ["--SiftExtraction.use_gpu", "0"]
    assert matching == ["--SiftMatching.use_gpu", "0"]


def test_choose_colmap_cpu_gpu_flags_v313_feature():
    feat = "  --FeatureExtraction.use_gpu arg (=1)\n  --FeatureExtraction.gpu_index arg (=-1)\n"
    match = "  --FeatureMatching.use_gpu arg (=1)\n  --FeatureMatching.max_ratio arg (=0.8)\n"
    extract, matching = choose_colmap_cpu_gpu_flags(feat, match)
    assert extract == ["--FeatureExtraction.use_gpu", "0"]
    assert matching == ["--FeatureMatching.use_gpu", "0"]


def test_choose_colmap_cpu_gpu_flags_empty_help_defaults_legacy():
    extract, matching = choose_colmap_cpu_gpu_flags("", "")
    assert extract == ["--SiftExtraction.use_gpu", "0"]
    assert matching == ["--SiftMatching.use_gpu", "0"]


def test_colmap_cpu_gpu_flags_probes_help(monkeypatch):
    colmap_cpu_gpu_flags.cache_clear()

    def fake_help(_colmap: str, subcommand: str) -> str:
        if subcommand == "feature_extractor":
            return "--FeatureExtraction.use_gpu arg (=1)\n"
        if subcommand == "exhaustive_matcher":
            return "--FeatureMatching.use_gpu arg (=1)\n"
        return ""

    monkeypatch.setattr(
        "ps1_hood.reconstruct.colmap._colmap_subcommand_help", fake_help
    )
    extract, matching = colmap_cpu_gpu_flags("/fake/colmap")
    assert extract == ("--FeatureExtraction.use_gpu", "0")
    assert matching == ("--FeatureMatching.use_gpu", "0")
    colmap_cpu_gpu_flags.cache_clear()


def test_colmap_cpu_gpu_flags_probes_legacy(monkeypatch):
    colmap_cpu_gpu_flags.cache_clear()

    def fake_help(_colmap: str, subcommand: str) -> str:
        if subcommand == "feature_extractor":
            return "--SiftExtraction.use_gpu arg (=1)\n"
        return "--SiftMatching.use_gpu arg (=1)\n"

    monkeypatch.setattr(
        "ps1_hood.reconstruct.colmap._colmap_subcommand_help", fake_help
    )
    extract, matching = colmap_cpu_gpu_flags("/fake/colmap-legacy")
    assert extract == ("--SiftExtraction.use_gpu", "0")
    assert matching == ("--SiftMatching.use_gpu", "0")
    colmap_cpu_gpu_flags.cache_clear()


def test_filter_frames_drops_images_without_tvg(tmp_path: Path):
    import sqlite3

    frames = [
        _fr(0, 0, 0, "p1"),
        _fr(8, 0, 0, "p2"),
        _fr(16, 0, 5, "p3"),
        _fr(24, 0, 5, "p4"),
        _fr(32, 0, 0, "p5"),
    ]
    names = ["a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg"]
    db = tmp_path / "database.db"
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT, camera_id INTEGER)"
    )
    con.execute(
        "CREATE TABLE two_view_geometries (pair_id INTEGER PRIMARY KEY, rows INTEGER)"
    )
    for i, n in enumerate(names, start=1):
        con.execute("INSERT INTO images VALUES (?, ?, 1)", (i, n))
    _insert_pinhole_camera(con, 1, 640, 480)
    # COLMAP pair_id: id1 * max + id2 with id1 < id2, max=2147483647
    max_images = 2147483647
    # Successful track between 1-2 and 3-4; image 5 orphan
    con.execute(
        "INSERT INTO two_view_geometries VALUES (?, ?)",
        (1 * max_images + 2, 10),
    )
    con.execute(
        "INSERT INTO two_view_geometries VALUES (?, ?)",
        (3 * max_images + 4, 8),
    )
    con.execute(
        "INSERT INTO two_view_geometries VALUES (?, ?)",
        (1 * max_images + 5, 0),  # empty — must not count
    )
    con.commit()
    con.close()
    model = write_known_pose_model(frames, names, tmp_path / "sparse")
    kept_f, kept_n = filter_frames_registered_in_matches(
        frames, names, db, model_dir=model
    )
    assert "e.jpg" not in kept_n
    assert set(kept_n) == {"a.jpg", "b.jpg", "c.jpg", "d.jpg"}
    assert "e.jpg" not in (model / "images.txt").read_text()


def test_remap_cameras_txt_widths_match_db_mixed_sizes(tmp_path: Path):
    """Mixed crop sizes + per-image DB cameras → cameras.txt must match DB, not frames.

    Reproduces the PC SIGABRT: reconstruction.cc Check failed:
    existing_camera.width == camera.width (1901 vs 1920) when cameras.txt was
    rebuilt from a single frame while DB held a different width for that CAMERA_ID.
    """
    import sqlite3

    frames = [
        _fr(0, 0, 0, "p1"),
        _fr(8, 0, 0, "p2"),
        _fr(16, 0, 5, "p3"),
        _fr(24, 0, 5, "p4"),
    ]
    # Frame metadata claims 1901 — but DB camera for that id is 1920 (the bug case).
    frames[0]["width"] = 1901
    frames[0]["height"] = 1080
    frames[1]["width"] = 1920
    frames[1]["height"] = 1080
    frames[2]["width"] = 1728
    frames[2]["height"] = 1004
    frames[3]["width"] = 1920
    frames[3]["height"] = 1080
    names = ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
    model = write_known_pose_model(frames, names, tmp_path / "sparse")
    db = tmp_path / "database.db"
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT, camera_id INTEGER)"
    )
    # Per-image cameras as single_camera=0; DB widths are ground truth for COLMAP.
    db_sizes = {
        1: (1920, 1080),  # deliberate mismatch vs frames[0] 1901
        2: (1920, 1080),
        3: (1728, 1004),
        4: (1920, 1080),
    }
    for i, n in enumerate(names, start=1):
        con.execute("INSERT INTO images VALUES (?, ?, ?)", (i, n, i))
        w, h = db_sizes[i]
        _insert_pinhole_camera(con, i, w, h, fx=float(w))
    con.commit()
    con.close()

    db_cams = read_db_cameras(db)
    assert db_cams[1][1] == 1920  # width from DB, not 1901

    path, kept_f, kept_n = remap_known_pose_model_to_db(model, frames, names, db)
    assert kept_n == names
    cams = (path / "cameras.txt").read_text()
    cam_lines = [ln for ln in cams.splitlines() if ln and not ln.startswith("#")]
    assert len(cam_lines) == 4  # one cameras.txt row per DB camera_id
    # Must match DB widths — never the stale 1901 from frame metadata.
    assert "1901" not in cams
    assert "1 PINHOLE 1920 1080" in cams
    assert "2 PINHOLE 1920 1080" in cams
    assert "3 PINHOLE 1728 1004" in cams
    assert "4 PINHOLE 1920 1080" in cams
    # images.txt CAMERA_IDs must be the DB camera_ids
    imgs = (path / "images.txt").read_text()
    pose_lines = [
        ln for ln in imgs.splitlines() if ln and not ln.startswith("#") and ln.strip()
    ]
    assert len(pose_lines) == 4
    for ln, expect_cid in zip(pose_lines, [1, 2, 3, 4], strict=True):
        parts = ln.split()
        assert int(parts[8]) == expect_cid


def test_remap_shared_camera_still_reads_db_row(tmp_path: Path):
    import sqlite3

    frames = [_fr(0, 0, 0, "p1"), _fr(8, 0, 0, "p2"), _fr(16, 0, 5, "p3"), _fr(24, 0, 5, "p4")]
    names = ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
    model = write_known_pose_model(frames, names, tmp_path / "sparse")
    db = tmp_path / "database.db"
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT, camera_id INTEGER)"
    )
    for i, n in enumerate(names, start=1):
        con.execute("INSERT INTO images VALUES (?, ?, 1)", (i, n))
    _insert_pinhole_camera(con, 1, 640, 480, fx=321.5)
    con.commit()
    con.close()
    path, _, _ = remap_known_pose_model_to_db(model, frames, names, db)
    cams = (path / "cameras.txt").read_text()
    cam_lines = [ln for ln in cams.splitlines() if ln and not ln.startswith("#")]
    assert len(cam_lines) == 1
    assert cam_lines[0].startswith("1 PINHOLE 640 480")
    assert "321.500000" in cam_lines[0]



def test_matches_importer_argv_enables_guided_matching(monkeypatch):
    monkeypatch.setattr(
        "ps1_hood.reconstruct.colmap.colmap_cpu_gpu_flags",
        lambda _c: (("--SiftExtraction.use_gpu", "0"), ("--SiftMatching.use_gpu", "0")),
    )
    monkeypatch.setattr(
        "ps1_hood.reconstruct.colmap._colmap_subcommand_help",
        lambda _c, _s: "--SiftMatching.guided_matching arg (=0)\n",
    )
    argv = matches_importer_argv(
        "colmap", Path("db.db"), Path("pairs.txt"), guided_matching=True
    )
    assert "--match_type" in argv
    assert argv[argv.index("--match_type") + 1] == "pairs"
    assert "--SiftMatching.guided_matching" in argv
    assert argv[argv.index("--SiftMatching.guided_matching") + 1] == "1"
    assert "--SiftMatching.use_gpu" in argv


def test_mean_track_length_from_bin(tmp_path: Path):
    # Minimal points3D.bin: 1 point, track length 3
    import struct

    model = tmp_path / "model"
    model.mkdir()
    # id, xyz, rgb, error, track_len, then 3*(image_id u32, point2D u32)
    blob = struct.pack("<Q", 1)  # count
    blob += struct.pack("<Q", 42)  # id
    blob += struct.pack("<ddd", 1.0, 2.0, 3.0)
    blob += bytes([10, 20, 30])
    blob += struct.pack("<d", 0.5)
    blob += struct.pack("<Q", 3)
    for img in (1, 2, 3):
        blob += struct.pack("<II", img, 0)
    (model / "points3D.bin").write_bytes(blob)
    assert mean_track_length(model) == pytest.approx(3.0)


def test_cross_pano_prefers_more_pairs():
    # Three panos along a line with overlapping headings → more than one pair/frame budget
    frames = [
        _fr(0, 0, 0, "p1"),
        _fr(8, 0, 5, "p2"),
        _fr(16, 0, 0, "p3"),
        _fr(24, 0, 5, "p4"),
    ]
    pairs_default = cross_pano_pair_indices(frames)
    pairs_tight = cross_pano_pair_indices(frames, max_pairs_per_frame=1)
    assert len(pairs_default) >= len(pairs_tight)
    assert len(pairs_default) >= 3
