"""Order refined cameras into drive tracks; lerp ENU poses for midframes."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

# Skip FILM/lerp when the leg is rotation-only / same-pano mates.
MIN_LERP_BASELINE_M = 2.0
# Densify/recon: keep every Nth midframe + all real panos (full FILM ≈ near-dupes).
DENSIFY_MIDFRAME_STRIDE = 4

_POSE_ENU_KEYS = ("e", "n", "u", "heading")
_POSE_PINHOLE_KEYS = ("fov", "width", "height")
_REQUIRED_INTERP_KEYS = _POSE_ENU_KEYS + _POSE_PINHOLE_KEYS


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


def enu_baseline_m(a: dict[str, Any], b: dict[str, Any]) -> float:
    """‖ΔENU‖ in metres between two pose dicts."""
    de = float(b["e"]) - float(a["e"])
    dn = float(b["n"]) - float(a["n"])
    du = float(b["u"]) - float(a["u"])
    return math.sqrt(de * de + dn * dn + du * du)


def lerp_pose(a: dict[str, Any], b: dict[str, Any], t: float) -> dict[str, Any]:
    """Linear ENU pose for a midframe. Never invent extrinsics outside a→b.

    ``fov`` / ``width`` / ``height`` must survive for PINHOLE ``K``; missing size
    yields silent wrong-fx downstream.
    """
    from ps1_hood.geo import heading_diff, wrap_heading

    width = a.get("width") if a.get("width") is not None else b.get("width")
    height = a.get("height") if a.get("height") is not None else b.get("height")
    if a.get("fov") is None and b.get("fov") is None:
        raise RuntimeError("lerp_pose: fov missing on both endpoints (PINHOLE K)")
    if width is None or height is None:
        raise RuntimeError(
            "lerp_pose: width/height missing on both endpoints — "
            "PINHOLE K would get silent wrong-fx"
        )
    fov_a = float(a["fov"] if a.get("fov") is not None else b["fov"])
    fov_b = float(b["fov"] if b.get("fov") is not None else a["fov"])
    pitch_a = float(a.get("pitch") or 0.0)
    pitch_b = float(b.get("pitch") or 0.0)
    return {
        "e": a["e"] + t * (b["e"] - a["e"]),
        "n": a["n"] + t * (b["n"] - a["n"]),
        "u": a["u"] + t * (b["u"] - a["u"]),
        "heading": wrap_heading(a["heading"] + t * heading_diff(a["heading"], b["heading"])),
        "pitch": pitch_a + t * (pitch_b - pitch_a),
        "fov": fov_a + t * (fov_b - fov_a),
        "width": int(width),
        "height": int(height),
    }


def assert_interp_frame_poses(frames: list[dict[str, Any]]) -> None:
    """Fail loud if any interp/FILM frame lacks valid ENU + PINHOLE size.

    Call on the **full** FILM-rate list before densify/recon. Densify may
    subsample midframes, but every midframe must still carry lerp_pose ENU.
    """
    if not frames:
        raise RuntimeError(
            "interp frames empty — densify/recon need posed frames from lerp_pose"
        )
    for i, frame in enumerate(frames):
        missing = [k for k in _REQUIRED_INTERP_KEYS if frame.get(k) is None]
        if missing:
            raise RuntimeError(
                f"interp frame {i} (index={frame.get('index')}) missing {missing}; "
                "midframe poses must come from lerp_pose / align ENU, not invented"
            )
        for key in _POSE_ENU_KEYS + ("fov",):
            try:
                val = float(frame[key])
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"interp frame {i}: {key}={frame.get(key)!r} is not numeric"
                ) from exc
            if not math.isfinite(val):
                raise RuntimeError(f"interp frame {i}: {key}={val} is not finite")
        try:
            w = int(frame["width"])
            h = int(frame["height"])
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"interp frame {i}: width/height must be ints for PINHOLE K "
                f"(got {frame.get('width')!r}×{frame.get('height')!r})"
            ) from exc
        if w <= 0 or h <= 0:
            raise RuntimeError(
                f"interp frame {i}: invalid size {w}×{h} for PINHOLE K"
            )


def select_densify_frames(
    frames: list[dict[str, Any]],
    *,
    midframe_stride: int = DENSIFY_MIDFRAME_STRIDE,
) -> list[dict[str, Any]]:
    """Keyframes (real panos) + every Nth midframe for densify/recon.

    Asserts ENU (+fov/width/height) on **every** input frame first — full FILM
    rate must be posed even when densify only consumes a stride subset.
    """
    assert_interp_frame_poses(frames)
    if midframe_stride < 1:
        raise ValueError(f"midframe_stride must be >= 1, got {midframe_stride}")
    out: list[dict[str, Any]] = []
    mid_i = 0
    for frame in frames:
        if not frame.get("interpolated", False):
            out.append(frame)
            continue
        if mid_i % midframe_stride == 0:
            out.append(frame)
        mid_i += 1
    return out


def _enu_xy_dist_m(a: dict[str, Any], b: dict[str, Any]) -> float:
    de = float(a["e"]) - float(b["e"])
    dn = float(a["n"]) - float(b["n"])
    return math.hypot(de, dn)


def select_posed_sparse_frames(
    keyframes: list[dict[str, Any]],
    interp_frames: list[dict[str, Any]] | None = None,
    *,
    midframe_stride: int = DENSIFY_MIDFRAME_STRIDE,
    min_midframe_baseline_m: float = MIN_LERP_BASELINE_M,
) -> list[dict[str, Any]]:
    """Orbit keyframes + every Nth posed FILM midframe for denser COLMAP tracks.

    Real SV crops / orbit headings stay the backbone. Subsampled midframes that
    already carry ``lerp_pose`` ENU (+ fov/width/height) are appended so landmarks
    see more cameras across baselines — feeding OpenMVS neighbor selection.

    - Asserts ENU + PINHOLE size on the **full** FILM-rate ``interp_frames`` first.
    - Midframes closer than ``min_midframe_baseline_m`` to an already-selected
      camera (keyframe or prior mid) are skipped (near-dupe cams drown MVS).
    - Without interp frames, returns ``keyframes`` unchanged.
    """
    if not keyframes and not interp_frames:
        return []
    if not interp_frames:
        return list(keyframes)

    assert_interp_frame_poses(interp_frames)
    if midframe_stride < 1:
        raise ValueError(f"midframe_stride must be >= 1, got {midframe_stride}")
    if min_midframe_baseline_m < 0:
        raise ValueError(
            f"min_midframe_baseline_m must be >= 0, got {min_midframe_baseline_m}"
        )

    out: list[dict[str, Any]] = list(keyframes)
    mid_i = 0
    for frame in interp_frames:
        if not frame.get("interpolated", False):
            continue
        keep = mid_i % midframe_stride == 0
        mid_i += 1
        if not keep:
            continue
        # Skip near-dupe of any already-selected camera (keyframe or mid).
        if any(
            _enu_xy_dist_m(frame, prev) < min_midframe_baseline_m for prev in out
        ):
            continue
        out.append(frame)
    return out
