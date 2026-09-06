"""Photo-consistency façade scoring / homography / filter tests."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from ps1_hood.reconstruct.photo_planes import (
    View,
        hypothesize_vertical_planes,
    plane_dict_for_obj,
    plane_homography,
    score_vertical_plane,
    search_photo_consistent_planes,
    vertical_plane_from_point_heading,
    zncc,
)


def test_zncc_identical_is_one() -> None:
    import pytest

    a = np.arange(64, dtype=np.float64).reshape(8, 8)
    assert zncc(a, a) == pytest.approx(1.0, abs=1e-6)


def test_zncc_anticorrelated() -> None:
    a = np.linspace(0, 1, 64).reshape(8, 8)
    b = 1.0 - a
    assert zncc(a, b) < -0.99


def test_zncc_too_small_is_nan() -> None:
    a = np.array([1.0, 2.0])
    assert math.isnan(zncc(a, a))


def _two_cams_looking_at_plane_z() -> tuple[list[View], np.ndarray, float, np.ndarray]:
    """Synthetic: plane z=+5 (cam forward +Z), textured wall visible in both."""
    K = np.array([[800.0, 0, 320], [0, 800.0, 240], [0, 0, 1]], dtype=np.float64)
    # Paint a distinctive pattern
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    rng = np.random.default_rng(42)
    noise = rng.integers(0, 255, (480, 640), dtype=np.uint8)
    img[:, :, 0] = noise
    img[:, :, 1] = np.roll(noise, 17, axis=1)
    img[:, :, 2] = np.roll(noise, 9, axis=0)
    cv2.rectangle(img, (180, 80), (460, 400), (30, 140, 220), -1)
    cv2.putText(img, "FACADE", (220, 260), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)

    R = np.eye(3, dtype=np.float64)
    # Cameras along +X with identity R → looking +Z; plane z=5
    t0 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    t1 = np.array([-1.5, 0.0, 0.0], dtype=np.float64)  # baseline in cam X
    n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    d = -5.0
    center = np.array([0.0, 0.0, 5.0], dtype=np.float64)
    v0 = View(img, K, R, t0, frame_index=0, pano_id="a")
    # Source view: warp img as if translated — for identical image content at same
    # pixels this is approximate; use same image for high-ZNCC smoke (geometry valid).
    v1 = View(img.copy(), K, R, t1, frame_index=1, pano_id="b")
    return [v0, v1], n, d, center


def test_plane_homography_maps_plane_points() -> None:
    views, n, d, center = _two_cams_looking_at_plane_z()
    v0, v1 = views
    H = plane_homography(v0.K, v0.Rcw, v0.t, v1.K, v1.Rcw, v1.t, n, d)
    # A world point on the plane → project both cams; H should map uv0 → uv1
    X = center.reshape(1, 3)
    from ps1_hood.reconstruct.photo_planes import P_from_Rt, project_points

    uv0, f0 = project_points(P_from_Rt(v0.K, v0.Rcw, v0.t), X)
    uv1, f1 = project_points(P_from_Rt(v1.K, v1.Rcw, v1.t), X)
    assert f0.all() and f1.all()
    p = np.array([uv0[0, 0], uv0[0, 1], 1.0], dtype=np.float64)
    mapped = H @ p
    mapped = mapped[:2] / mapped[2]
    assert np.allclose(mapped, uv1[0], atol=0.5)


def test_score_vertical_plane_accepts_consistent_texture() -> None:
    views, n, d, center = _two_cams_looking_at_plane_z()
    # Warp view1 image into view0 geometry so content matches plane transfer
    v0, v1 = views
    H = plane_homography(v0.K, v0.Rcw, v0.t, v1.K, v1.Rcw, v1.t, n, d)
    # Synthesize src image by warping ref through H^{-1} so ZNCC should be high
    h, w = v0.image_bgr.shape[:2]
    v1.image_bgr = cv2.warpPerspective(v0.image_bgr, H, (w, h), flags=cv2.INTER_LINEAR)
    result = score_vertical_plane(
        [v0, v1], n, d, center, width_m=4.0, height_m=3.0, zncc_accept=0.35, patch=64
    )
    assert result["ok"] is True
    assert result["zncc"] >= 0.35


def test_score_vertical_plane_rejects_wrong_depth() -> None:
    views, n, d, center = _two_cams_looking_at_plane_z()
    v0, v1 = views
    H = plane_homography(v0.K, v0.Rcw, v0.t, v1.K, v1.Rcw, v1.t, n, d)
    h, w = v0.image_bgr.shape[:2]
    v1.image_bgr = cv2.warpPerspective(v0.image_bgr, H, (w, h), flags=cv2.INTER_LINEAR)
    # Wrong plane depth → low consistency
    bad = score_vertical_plane(
        [v0, v1],
        n,
        -12.0,
        np.array([0.0, 0.0, 12.0]),
        width_m=4.0,
        height_m=3.0,
        zncc_accept=0.35,
        patch=64,
    )
    assert bad["ok"] is False


def test_vertical_plane_heading_convention_north() -> None:
    # heading 0 → +N normal
    p0 = np.array([10.0, 20.0, 2.0])
    n, d = vertical_plane_from_point_heading(p0, 0.0)
    assert abs(n[0]) < 1e-9
    assert abs(n[1] - 1.0) < 1e-9
    assert abs(n @ p0 + d) < 1e-9


def test_hypothesize_from_heading_distance() -> None:
    frames = [
        {"e": 0.0, "n": 0.0, "u": 2.5, "heading": 90.0, "pitch": 0.0, "fov": 90.0, "pano_id": "p0"},
        {"e": 8.0, "n": 0.0, "u": 2.5, "heading": 90.0, "pitch": 0.0, "fov": 90.0, "pano_id": "p1"},
    ]
    hyps = hypothesize_vertical_planes(frames, distances_m=(10.0, 14.0))
    assert len(hyps) >= 2
    assert any(h["source"] == "heading_distance" for h in hyps)


def test_search_fail_loud_empty_without_frames() -> None:
    kept = search_photo_consistent_planes([], zncc_accept=0.35)
    assert kept == []


def test_search_fail_loud_random_uncorrelated_images(tmp_path: Path) -> None:
    """Two unrelated images at different panos → no acceptances."""
    rng = np.random.default_rng(1)
    paths = []
    for i in range(2):
        img = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
        p = tmp_path / f"v{i}.jpg"
        cv2.imwrite(str(p), img)
        paths.append(p)
    frames = [
        {
            "path": str(paths[0]),
            "e": 0.0,
            "n": 0.0,
            "u": 2.5,
            "heading": 90.0,
            "pitch": 0.0,
            "fov": 90.0,
            "pano_id": "a",
        },
        {
            "path": str(paths[1]),
            "e": 10.0,
            "n": 0.0,
            "u": 2.5,
            "heading": 90.0,
            "pitch": 0.0,
            "fov": 90.0,
            "pano_id": "b",
        },
    ]
    kept = search_photo_consistent_planes(
        frames, zncc_accept=0.55, refine=False, max_keep=4
    )
    assert kept == []


def test_plane_dict_for_obj_converts_normal_form() -> None:
    n = np.array([1.0, 0.0, 0.0])
    d = -5.0  # plane x=5
    pl = {
        "n": n,
        "d": d,
        "center": np.array([5.0, 0.0, 4.0]),
        "width_m": 8.0,
        "height_m": 9.0,
        "zncc": 0.5,
        "source": "heading_distance",
    }
    out = plane_dict_for_obj(pl, ground_z=0.0)
    assert abs(out["nx"] - 1.0) < 1e-9
    assert abs(out["d"] - 5.0) < 1e-6  # nx*e + ny*n = d_xy
    assert len(out["quad"]) == 4


def test_extract_facades_writes_ground_only_on_fail(tmp_path: Path) -> None:
    from ps1_hood.reconstruct.facades import extract_facades

    rng = np.random.default_rng(2)
    frames = []
    for i in range(2):
        img = rng.integers(0, 255, (180, 240, 3), dtype=np.uint8)
        p = tmp_path / f"f{i}.jpg"
        cv2.imwrite(str(p), img)
        frames.append(
            {
                "path": str(p),
                "e": float(i * 12),
                "n": 0.0,
                "u": 2.5,
                "heading": 0.0,
                "pitch": 0.0,
                "fov": 90.0,
                "pano_id": f"p{i}",
            }
        )
    dest = tmp_path / "facades.obj"
    meta = extract_facades(None, dest, frames=frames, zncc_accept=0.9, n_planes=4)
    assert meta["source"] == "photo_consistency"
    assert meta["planes"] == 0
    assert dest.is_file()
    body = dest.read_text()
    assert "usemtl ground" in body
    assert "facade_00" not in body
