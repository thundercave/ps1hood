from __future__ import annotations

import math

import numpy as np

from ps1_hood.geo import (
    BBox,
    LocalFrame,
    bearing_deg,
    camera_rotation_cv,
    haversine_m,
    heading_diff,
    nearest_point_on_segment,
    sample_polyline_m,
    wrap_heading,
)


def test_haversine_known_distance() -> None:
    # roughly 1 degree of latitude
    d = haversine_m(52.0, 5.0, 53.0, 5.0)
    assert 110_000 < d < 112_000


def test_enu_roundtrip() -> None:
    frame = LocalFrame(52.09, 5.12)
    e, n, u = frame.to_enu(52.091, 5.121, 12.0)
    lat, lon, alt = frame.to_geodetic(e, n, u)
    assert abs(lat - 52.091) < 1e-7
    assert abs(lon - 5.121) < 1e-7
    assert abs(alt - 12.0) < 1e-3


def test_heading_wrap() -> None:
    assert wrap_heading(370) == 10
    assert abs(heading_diff(350, 10) - 20) < 1e-9
    assert abs(heading_diff(10, 350) + 20) < 1e-9


def test_bearing_east() -> None:
    b = bearing_deg(0.0, 0.0, 0.0, 1.0)
    assert abs(b - 90.0) < 0.5


def test_camera_rotation_orthonormal() -> None:
    R = np.array(camera_rotation_cv(37.0, -12.0))
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)
    assert abs(np.linalg.det(R) - 1.0) < 1e-6
    # heading 0, pitch 0 looks north (+Y)
    forward = np.array(camera_rotation_cv(0.0, 0.0))[:, 2]
    assert abs(forward[0]) < 1e-6
    assert abs(forward[1] - 1.0) < 1e-6


def test_bbox_contains() -> None:
    box = BBox(52.0, 5.0, 52.1, 5.1)
    assert box.contains(52.05, 5.05)
    assert not box.contains(52.2, 5.05)
    assert box.width_m() > 0
    assert box.height_m() > 0


def test_bbox_contains_m_slop() -> None:
    box = BBox(53.24587, 6.60282, 53.24639, 6.60377)
    # a point ~15 m east of the east edge should pass 25 m slop, fail 5 m
    lat = 53.24613
    lon = 6.60400
    assert not box.contains(lat, lon)
    assert box.contains_m(lat, lon, slop_m=25.0)
    assert not box.contains_m(lat, lon, slop_m=5.0)
    padded = box.padded(20.0)
    assert padded.contains(lat, lon)
    wider = box.union_point(53.25, 6.61)
    assert wider.north >= 53.25
    assert wider.east >= 6.61


def test_sample_polyline() -> None:
    # ~111 m north
    line = [(5.0, 52.0), (5.0, 52.001)]
    samples = sample_polyline_m(line, 20.0)
    assert len(samples) >= 5
    assert all(abs(h) < 2 or abs(h - 360) < 2 for _, _, h in samples)


def test_nearest_on_segment() -> None:
    qx, qy, t = nearest_point_on_segment(1.0, 1.0, 0.0, 0.0, 4.0, 0.0)
    assert abs(qx - 1.0) < 1e-9
    assert abs(qy) < 1e-9
    assert 0 <= t <= 1
