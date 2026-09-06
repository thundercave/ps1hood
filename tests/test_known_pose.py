from __future__ import annotations

from ps1_hood.reconstruct.known_pose import select_stereo_pairs, sequential_pairs


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
