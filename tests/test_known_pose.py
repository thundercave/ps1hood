from __future__ import annotations

from ps1_hood.reconstruct.known_pose import (
    forward_drive_pairs,
    select_stereo_pairs,
    sequential_pairs,
)


def _fr(e: float, n: float, heading: float) -> dict:
    return {"e": e, "n": n, "u": 2.5, "heading": heading, "pitch": 0.0, "fov": 90.0}


def test_select_stereo_pairs_baseline_and_heading() -> None:
    frames = [
        _fr(0.0, 0.0, 0.0),    # 0
        _fr(5.0, 0.0, 5.0),    # 1 — good baseline + heading
        _fr(50.0, 0.0, 0.0),   # 2 — too far
        _fr(1.0, 0.0, 0.0),    # 3 — too close
        _fr(8.0, 1.0, 90.0),   # 4 — heading too different from 0
        _fr(10.0, 0.5, 20.0),  # 5 — good vs 0
    ]
    pairs = select_stereo_pairs(frames, max_pairs_per_frame=4)
    assert (0, 1) in pairs
    assert (0, 5) in pairs
    assert (0, 2) not in pairs  # baseline > 25
    assert (0, 3) not in pairs  # baseline < 2
    assert (0, 4) not in pairs  # heading 90°


def test_select_stereo_pairs_empty_when_no_geometry() -> None:
    frames = [_fr(0.0, 0.0, 0.0), _fr(0.5, 0.0, 0.0)]
    assert select_stereo_pairs(frames) == []


def test_sequential_pairs_fallback() -> None:
    assert sequential_pairs(5, pair_step=2) == [(0, 2), (1, 3), (2, 4)]
    assert sequential_pairs(1) == []


def test_forward_drive_pairs_three_mates() -> None:
    # Evenly spaced along a line — each frame should link to next 3 within baseline.
    frames = [_fr(float(i * 5), 0.0, 0.0) for i in range(6)]
    pairs = forward_drive_pairs(
        frames, n_forward=3, min_baseline_m=2.0, max_baseline_m=25.0, quadratic_overlap=False
    )
    # Frame 0 → 1,2,3
    assert (0, 1) in pairs and (0, 2) in pairs and (0, 3) in pairs
    assert (0, 4) not in pairs  # only 3 forward without quadratic
    # Frame 2 → 3,4,5
    assert (2, 3) in pairs and (2, 4) in pairs and (2, 5) in pairs
    # Creates overlapping edges (3-cycles seed): 0-1, 0-2, 1-2
    assert (1, 2) in pairs


def test_forward_drive_pairs_skips_too_close() -> None:
    frames = [
        _fr(0.0, 0.0, 0.0),
        _fr(0.5, 0.0, 0.0),  # too close — skip, still count toward? No: skip without counting
        _fr(5.0, 0.0, 0.0),
        _fr(10.0, 0.0, 0.0),
        _fr(15.0, 0.0, 0.0),
    ]
    pairs = forward_drive_pairs(frames, n_forward=3, min_baseline_m=2.0)
    assert (0, 1) not in pairs
    assert (0, 2) in pairs and (0, 3) in pairs and (0, 4) in pairs


def test_forward_drive_pairs_quadratic_adds_skip_edges() -> None:
    frames = [_fr(float(i * 3), 0.0, 0.0) for i in range(10)]
    plain = forward_drive_pairs(
        frames, n_forward=1, min_baseline_m=1.0, quadratic_overlap=False
    )
    quad = forward_drive_pairs(
        frames, n_forward=1, min_baseline_m=1.0, quadratic_overlap=True
    )
    assert (0, 1) in plain
    assert (0, 2) not in plain  # n_forward=1 only
    assert (0, 2) in quad  # quadratic i+2
    assert (0, 4) in quad  # quadratic i+4
    assert len(quad) > len(plain)
