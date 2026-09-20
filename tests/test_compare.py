"""Pano ↔ mesh compare: projection math + fail-loud empty runs."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from ps1_hood.reconstruct.compare import (
    CompareError,
    load_product_quads,
    project_quad,
    quad_visible,
    run_compare,
    subsample_cameras,
)
from ps1_hood.reconstruct.photo_planes import K_from_frame, P_from_Rt, project_points


def test_project_points_optical_axis_is_image_center() -> None:
    K = K_from_frame(640, 480, 90.0)
    Rcw = np.eye(3, dtype=np.float64)
    t = np.zeros(3, dtype=np.float64)
    P = P_from_Rt(K, Rcw, t)
    X = np.array(
        [[0.0, 0.0, 5.0], [1.0, 0.0, 5.0], [1.0, 1.0, 5.0], [0.0, 1.0, 5.0]],
        dtype=np.float64,
    )
    uv, front = project_points(P, X)
    assert bool(front.all())
    assert uv[0, 0] == pytest.approx(320.0, abs=0.5)
    assert uv[0, 1] == pytest.approx(240.0, abs=0.5)


def test_project_quad_wall_in_front_is_visible() -> None:
    frame = {
        "e": 0.0,
        "n": 0.0,
        "u": 2.0,
        "heading": 0.0,
        "pitch": 0.0,
        "fov": 90.0,
    }
    corners = np.array(
        [
            [-2.0, 8.0, 0.5],
            [2.0, 8.0, 0.5],
            [2.0, 8.0, 6.0],
            [-2.0, 8.0, 6.0],
        ],
        dtype=np.float64,
    )
    uv, front = project_quad(frame, corners, width=640, height=480)
    assert int(front.sum()) >= 3
    assert quad_visible(uv, front, 640, 480)


def test_project_quad_behind_camera_not_visible() -> None:
    frame = {
        "e": 0.0,
        "n": 0.0,
        "u": 2.0,
        "heading": 0.0,
        "pitch": 0.0,
        "fov": 90.0,
    }
    corners = np.array(
        [
            [-1.0, -8.0, 0.5],
            [1.0, -8.0, 0.5],
            [1.0, -8.0, 4.0],
            [-1.0, -8.0, 4.0],
        ],
        dtype=np.float64,
    )
    uv, front = project_quad(frame, corners, width=640, height=480)
    assert not quad_visible(uv, front, 640, 480)


def test_subsample_cameras_dedupes_heading_buckets() -> None:
    frames = [
        {"pano_id": "a", "heading": 0.0},
        {"pano_id": "a", "heading": 5.0},
        {"pano_id": "a", "heading": 90.0},
        {"pano_id": "b", "heading": 0.0},
    ]
    out = subsample_cameras(frames, max_cams=10)
    assert len(out) == 3


def test_empty_run_no_product_fail_loud(tmp_path: Path) -> None:
    (tmp_path / "align").mkdir()
    (tmp_path / "recon").mkdir()
    img = tmp_path / "shot.jpg"
    cv2.imwrite(str(img), np.zeros((64, 64, 3), dtype=np.uint8))
    cam = {
        "pano_id": "x",
        "shot_path": str(img),
        "e": 0.0,
        "n": 0.0,
        "u": 2.0,
        "heading": 0.0,
        "pitch": 0.0,
        "fov": 90.0,
        "width": 64,
        "height": 64,
    }
    (tmp_path / "align" / "cameras.json").write_text(
        json.dumps([cam]), encoding="utf-8"
    )
    with pytest.raises(CompareError, match="(?i)no product quads"):
        run_compare(tmp_path)


def test_empty_cameras_fail_loud(tmp_path: Path) -> None:
    (tmp_path / "align").mkdir()
    (tmp_path / "recon").mkdir()
    (tmp_path / "align" / "cameras.json").write_text("[]", encoding="utf-8")
    # Minimal plane so product load is not the first failure if cameras empty
    # after product — order is product first then cameras.
    plane = {
        "planes": [
            {
                "id": "f0",
                "n": [1.0, 0.0, 0.0],
                "d": -5.0,
                "quad": [
                    [5.0, -1.0, 0.0],
                    [5.0, 1.0, 0.0],
                    [5.0, 1.0, 3.0],
                    [5.0, -1.0, 3.0],
                ],
            }
        ]
    }
    (tmp_path / "recon" / "planes.json").write_text(
        json.dumps(plane), encoding="utf-8"
    )
    with pytest.raises(CompareError, match="(?i)camera"):
        run_compare(tmp_path)


def test_smoke_dense_compare_writes_summary() -> None:
    root = Path("runs/smoke-dense")
    if not (root / "align" / "cameras.json").is_file():
        pytest.skip("smoke-dense run not present")
    if not (root / "recon" / "planes.json").is_file():
        pytest.skip("smoke-dense planes missing")
    out = root / "recon" / "compare_test_out"
    summary = run_compare(root, max_cams=4, out_dir=out)
    assert int(summary["n_cams_scored"]) >= 1
    assert (out / "summary.json").is_file()
    assert summary["worst"]
    assert "zncc_mean" in summary["global"]
    assert summary.get("diagnose_only") is True
    # product untouched
    assert (root / "recon" / "planes.json").is_file()
    load_product_quads(root)  # still loadable
