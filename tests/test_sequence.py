from __future__ import annotations

from ps1_hood.interpolate.sequence import lerp_pose, order_track


def _pose(e: float, n: float, heading: float) -> dict:
    return {
        "e": e,
        "n": n,
        "u": 2.5,
        "heading": heading,
        "pitch": 0.0,
        "fov": 90.0,
        "pano_id": f"{e:.0f}-{n:.0f}",
    }


def test_orders_along_heading() -> None:
    poses = [
        _pose(20, 0, 90),
        _pose(0, 0, 90),
        _pose(10, 0, 90),
    ]
    track = order_track(poses)
    assert [p["e"] for p in track] == [0, 10, 20]


def test_lerp_heading_wraps() -> None:
    a = _pose(0, 0, 350)
    b = _pose(10, 0, 10)
    mid = lerp_pose(a, b, 0.5)
    assert abs(mid["heading"] - 0) < 1e-6 or abs(mid["heading"] - 360) < 1e-6
    assert abs(mid["e"] - 5) < 1e-9
