from __future__ import annotations

import pytest

from ps1_hood.interpolate.sequence import (
    DENSIFY_MIDFRAME_STRIDE,
    MIN_LERP_BASELINE_M,
    assert_interp_frame_poses,
    enu_baseline_m,
    lerp_pose,
    order_track,
    select_densify_frames,
    select_posed_sparse_frames,
)


def _pose(e: float, n: float, heading: float, **extra) -> dict:
    out = {
        "e": e,
        "n": n,
        "u": 2.5,
        "heading": heading,
        "pitch": 0.0,
        "fov": 90.0,
        "width": 640,
        "height": 480,
        "pano_id": f"{e:.0f}-{n:.0f}",
    }
    out.update(extra)
    return out


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


def test_lerp_preserves_fov_width_height_for_pinhole() -> None:
    a = _pose(0, 0, 0, fov=80.0, width=1920, height=1080)
    b = _pose(10, 0, 0, fov=100.0, width=1920, height=1080)
    mid = lerp_pose(a, b, 0.5)
    assert mid["fov"] == pytest.approx(90.0)
    assert mid["width"] == 1920
    assert mid["height"] == 1080
    assert mid["e"] == pytest.approx(5.0)
    assert mid["n"] == pytest.approx(0.0)
    assert mid["u"] == pytest.approx(2.5)


def test_lerp_pose_fails_loud_without_size() -> None:
    a = _pose(0, 0, 0)
    b = _pose(10, 0, 0)
    del a["width"]
    del a["height"]
    del b["width"]
    del b["height"]
    with pytest.raises(RuntimeError, match="width/height"):
        lerp_pose(a, b, 0.5)


def test_lerp_pose_fails_loud_without_fov() -> None:
    a = _pose(0, 0, 0)
    b = _pose(10, 0, 0)
    del a["fov"]
    del b["fov"]
    with pytest.raises(RuntimeError, match="fov"):
        lerp_pose(a, b, 0.5)


def test_enu_baseline_and_min_lerp_threshold() -> None:
    a = _pose(0, 0, 0)
    near = _pose(0.5, 0, 90)  # same-ish pano, heading mate
    far = _pose(8, 0, 0)
    assert enu_baseline_m(a, near) < MIN_LERP_BASELINE_M
    assert enu_baseline_m(a, far) >= MIN_LERP_BASELINE_M


def _frame(i: int, *, interpolated: bool, e: float = 0.0) -> dict:
    return {
        "index": i,
        "path": f"/tmp/{i:05d}.jpg",
        "e": e,
        "n": 0.0,
        "u": 2.5,
        "heading": 90.0,
        "pitch": 0.0,
        "fov": 90.0,
        "width": 640,
        "height": 480,
        "interpolated": interpolated,
    }


def test_assert_interp_frame_poses_ok() -> None:
    frames = [_frame(0, interpolated=False), _frame(1, interpolated=True, e=1.0)]
    assert_interp_frame_poses(frames)  # no raise


def test_assert_interp_frame_poses_missing_enu() -> None:
    bad = _frame(0, interpolated=True)
    del bad["e"]
    with pytest.raises(RuntimeError, match="missing"):
        assert_interp_frame_poses([bad])


def test_assert_interp_frame_poses_missing_pinhole_size() -> None:
    bad = _frame(0, interpolated=True)
    del bad["width"]
    with pytest.raises(RuntimeError, match="width"):
        assert_interp_frame_poses([bad])


def test_assert_interp_frame_poses_empty() -> None:
    with pytest.raises(RuntimeError, match="empty"):
        assert_interp_frame_poses([])


def test_select_densify_frames_keeps_keyframes_and_strided_mids() -> None:
    # key, mid, mid, mid, mid, key  → with stride 4: key, mid0, key
    frames = [
        _frame(0, interpolated=False, e=0),
        _frame(1, interpolated=True, e=1),
        _frame(2, interpolated=True, e=2),
        _frame(3, interpolated=True, e=3),
        _frame(4, interpolated=True, e=4),
        _frame(5, interpolated=False, e=5),
    ]
    chosen = select_densify_frames(frames, midframe_stride=4)
    assert [f["index"] for f in chosen] == [0, 1, 5]
    # Assert ran on full set — poison a mid that is not selected and ensure fail
    frames[3]["e"] = None
    with pytest.raises(RuntimeError, match="missing"):
        select_densify_frames(frames, midframe_stride=4)


def test_select_densify_default_stride() -> None:
    assert DENSIFY_MIDFRAME_STRIDE >= 1


def _keyframe(i: int, e: float, *, pano: str = "p") -> dict:
    return {
        "path": f"/tmp/kf_{i}.jpg",
        "e": e,
        "n": 0.0,
        "u": 2.5,
        "heading": 90.0,
        "pitch": 0.0,
        "fov": 90.0,
        "width": 640,
        "height": 480,
        "pano_id": pano,
        "interpolated": False,
    }


def test_select_posed_sparse_adds_strided_mids_with_baseline() -> None:
    # Keyframes at 0 and 20 m; mids every 1 m along the leg.
    keyframes = [_keyframe(0, 0.0, pano="a"), _keyframe(1, 20.0, pano="b")]
    interp = [
        _frame(0, interpolated=False, e=0.0),
        _frame(1, interpolated=True, e=2.0),
        _frame(2, interpolated=True, e=4.0),
        _frame(3, interpolated=True, e=6.0),
        _frame(4, interpolated=True, e=8.0),
        _frame(5, interpolated=True, e=10.0),
        _frame(6, interpolated=False, e=20.0),
    ]
    chosen = select_posed_sparse_frames(
        keyframes, interp, midframe_stride=2, min_midframe_baseline_m=2.0
    )
    # All keyframes kept
    assert chosen[0]["path"] == keyframes[0]["path"]
    assert chosen[1]["path"] == keyframes[1]["path"]
    mids = [f for f in chosen if f.get("interpolated")]
    # stride 2 → mid indices 0,2,4 among interpolated (e=2,6,10)
    assert [m["e"] for m in mids] == [2.0, 6.0, 10.0]
    # Each mid ≥2 m from every other selected camera
    for i, a in enumerate(chosen):
        for b in chosen[i + 1 :]:
            if a.get("interpolated") or b.get("interpolated"):
                assert enu_baseline_m(a, b) >= 2.0 - 1e-9


def test_select_posed_sparse_skips_mids_too_close_to_keyframes() -> None:
    keyframes = [_keyframe(0, 0.0), _keyframe(1, 10.0)]
    interp = [
        _frame(0, interpolated=False, e=0.0),
        _frame(1, interpolated=True, e=0.5),  # too close to keyframe at 0
        _frame(2, interpolated=True, e=5.0),  # ok
        _frame(3, interpolated=False, e=10.0),
    ]
    chosen = select_posed_sparse_frames(
        keyframes, interp, midframe_stride=1, min_midframe_baseline_m=2.0
    )
    mids = [f for f in chosen if f.get("interpolated")]
    assert [m["e"] for m in mids] == [5.0]


def test_select_posed_sparse_no_interp_returns_keyframes() -> None:
    keyframes = [_keyframe(0, 0.0), _keyframe(1, 8.0)]
    assert select_posed_sparse_frames(keyframes, None) == keyframes
    assert select_posed_sparse_frames(keyframes, []) == keyframes


def test_select_posed_sparse_asserts_full_film_rate() -> None:
    keyframes = [_keyframe(0, 0.0)]
    bad = _frame(1, interpolated=True, e=5.0)
    del bad["fov"]
    with pytest.raises(RuntimeError, match="missing"):
        select_posed_sparse_frames(keyframes, [_frame(0, interpolated=False), bad])


def test_select_densify_subsample_every_nth() -> None:
    frames = [_frame(i, interpolated=(i not in (0, 9)), e=float(i)) for i in range(10)]
    chosen = select_densify_frames(frames, midframe_stride=3)
    # keyframes 0,9 + mids at mid_i 0,3,6 → frames index 1,4,7
    assert [f["index"] for f in chosen] == [0, 1, 4, 7, 9]
