from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ps1_hood.interpolate.flow import interpolate_pair, interpolate_track
from ps1_hood.interpolate.sequence import MIN_LERP_BASELINE_M


def test_interpolate_pair_count_and_shape() -> None:
    a = np.zeros((40, 60, 3), dtype=np.uint8)
    b = np.zeros((40, 60, 3), dtype=np.uint8)
    a[:, :30] = (20, 40, 200)
    b[:, 15:45] = (20, 40, 200)
    frames = interpolate_pair(a, b, steps=3)
    assert len(frames) == 5  # start + 3 mids + end
    assert frames[0].shape == a.shape
    assert frames[-1].shape == b.shape


def _write_pose(tmp: Path, name: str, e: float, n: float, heading: float = 90.0) -> dict:
    img = np.zeros((32, 48, 3), dtype=np.uint8)
    img[:] = (10, 20, 30)
    path = tmp / f"{name}.jpg"
    cv2.imwrite(str(path), img)
    return {
        "shot_path": str(path),
        "e": e,
        "n": n,
        "u": 2.5,
        "heading": heading,
        "pitch": 0.0,
        "fov": 90.0,
        "width": 48,
        "height": 32,
        "pano_id": name,
    }


def test_interpolate_track_lerps_enu_and_skips_short_baseline(tmp_path: Path) -> None:
    dest = tmp_path / "frames"
    # A→B far enough for FILM; B→C same-pano heading mate (< 2 m) → skip mids
    poses = [
        _write_pose(tmp_path, "a", 0.0, 0.0, 90),
        _write_pose(tmp_path, "b", 10.0, 0.0, 90),
        _write_pose(tmp_path, "c", 11.2, 0.0, 180),  # ~1.2 m: < MIN_LERP but > order_track 0.5 m
    ]
    assert abs(poses[2]["e"] - poses[1]["e"]) < MIN_LERP_BASELINE_M
    assert abs(poses[2]["e"] - poses[1]["e"]) >= 0.5  # still on the drive track
    meta = interpolate_track(poses, dest, steps=2)
    # Every frame has ENU + PINHOLE size (assert inside interpolate_track)
    for f in meta:
        for k in ("e", "n", "u", "heading", "fov", "width", "height"):
            assert f.get(k) is not None
    # Midframes between A and B must sit on the lerp line
    mids = [f for f in meta if f.get("interpolated")]
    assert mids, "expected FILM midframes on the long leg"
    for m in mids:
        assert 0.0 < m["e"] < 10.0
        assert m["n"] == 0.0
    # Short B→C leg: destination keyframe only (no new mids after last key of long leg)
    # Last frame should be C at e≈10.3, not interpolated
    assert meta[-1]["interpolated"] is False
    assert abs(meta[-1]["e"] - 11.2) < 1e-9
