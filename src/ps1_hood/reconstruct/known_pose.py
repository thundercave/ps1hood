"""Stereo pair selection for frames with known ENU poses.

Street View orbits are not a sequential drive video — pairing only by list
order misses good baselines. Prefer nearby cameras with overlapping FOV.
"""

from __future__ import annotations

import math
from typing import Any

from ps1_hood.geo import heading_diff


def _baseline_m(a: dict[str, Any], b: dict[str, Any]) -> float:
    return math.hypot(float(a["e"]) - float(b["e"]), float(a["n"]) - float(b["n"]))


def _heading_sep_deg(a: dict[str, Any], b: dict[str, Any]) -> float:
    return abs(heading_diff(float(a["heading"]), float(b["heading"])))


def select_stereo_pairs(
    frames: list[dict[str, Any]],
    *,
    min_baseline_m: float = 2.0,
    max_baseline_m: float = 25.0,
    max_heading_diff_deg: float = 60.0,
    max_pairs_per_frame: int = 5,
    prefer_baseline_m: float = 8.0,
) -> list[tuple[int, int]]:
    """Pick (i, j) index pairs with useful stereo geometry.

    Keeps pairs whose ENU baseline is in ``[min_baseline_m, max_baseline_m]``
    and whose heading difference is ≤ ``max_heading_diff_deg``. Each frame
    keeps up to ``max_pairs_per_frame`` partners closest to ``prefer_baseline_m``.
    """
    n = len(frames)
    if n < 2:
        return []

    # Collect candidate partners per frame, then emit unique unordered pairs.
    best_for: dict[int, list[tuple[float, int]]] = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            baseline = _baseline_m(frames[i], frames[j])
            if baseline < min_baseline_m or baseline > max_baseline_m:
                continue
            if _heading_sep_deg(frames[i], frames[j]) > max_heading_diff_deg:
                continue
            score = abs(baseline - prefer_baseline_m)
            best_for[i].append((score, j))
            best_for[j].append((score, i))

    selected: set[tuple[int, int]] = set()
    for i, cands in best_for.items():
        cands.sort(key=lambda t: t[0])
        for _, j in cands[:max_pairs_per_frame]:
            selected.add((min(i, j), max(i, j)))

    return sorted(selected)


def sequential_pairs(n: int, pair_step: int = 2) -> list[tuple[int, int]]:
    """Fallback: classic sliding-window pairs along list order."""
    if n < 2 or pair_step < 1:
        return []
    return [(i, i + pair_step) for i in range(0, n - pair_step)]
