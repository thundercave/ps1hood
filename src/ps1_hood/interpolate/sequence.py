"""Order refined cameras into drive tracks."""

from __future__ import annotations

from typing import Any

import numpy as np


def order_track(poses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Greedy heading-aware path through the cameras."""
    if not poses:
        return []
    remaining = list(range(len(poses)))
    # start at the south-west-most camera
    start = min(remaining, key=lambda i: (poses[i]["n"], poses[i]["e"]))
    order = [start]
    remaining.remove(start)
    while remaining:
        i = order[-1]
        a = poses[i]
        hx = np.sin(np.deg2rad(a["heading"]))
        hy = np.cos(np.deg2rad(a["heading"]))
        best = None
        best_score = None
        for j in remaining:
            b = poses[j]
            de, dn = b["e"] - a["e"], b["n"] - a["n"]
            dist = float(np.hypot(de, dn))
            if dist < 0.5:
                continue
            ahead = (de * hx + dn * hy) / dist
            # prefer nearby, in-front cameras
            score = dist - 6.0 * max(0.0, ahead)
            if best_score is None or score < best_score:
                best_score = score
                best = j
        if best is None:
            break
        order.append(best)
        remaining.remove(best)
        if len(order) > 1:
            # drop a hop that jumped across the block
            prev = poses[order[-2]]
            cur = poses[order[-1]]
            if np.hypot(cur["e"] - prev["e"], cur["n"] - prev["n"]) > 35.0 and remaining:
                # leave it, but stop extending this track
                break
    return [poses[i] for i in order]


def lerp_pose(a: dict[str, Any], b: dict[str, Any], t: float) -> dict[str, Any]:
    from ps1_hood.geo import heading_diff, wrap_heading

    return {
        "e": a["e"] + t * (b["e"] - a["e"]),
        "n": a["n"] + t * (b["n"] - a["n"]),
        "u": a["u"] + t * (b["u"] - a["u"]),
        "heading": wrap_heading(a["heading"] + t * heading_diff(a["heading"], b["heading"])),
        "pitch": a["pitch"] + t * (b["pitch"] - a["pitch"]),
        "fov": a["fov"] + t * (b["fov"] - a["fov"]),
        "width": a.get("width") or b.get("width"),
        "height": a.get("height") or b.get("height"),
    }
