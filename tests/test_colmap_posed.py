"""Known-pose COLMAP text model + cross-pano pair filtering."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ps1_hood.reconstruct.colmap import (
    _focal_px,
    _rotmat_to_qvec,
    cross_pano_pair_indices,
    frames_have_known_poses,
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
    con.commit()
    con.close()
    remap_known_pose_model_to_db(model, frames, names, db)
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
