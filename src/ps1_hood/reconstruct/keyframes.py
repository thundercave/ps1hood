"""Aligned Street View keyframes for multi-view reconstruction.

Prefer real cropped shots with known ENU poses over DIS-interpolated
midframes — warped guesses poison geometry.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ps1_hood.capture.discover import subsample_evenly
from ps1_hood.project import Project

log = logging.getLogger(__name__)

# Prefer near-horizon shots when enough remain for stereo.
_PITCH_NEAR_HORIZON_DEG = 5.0
_MIN_VIEWS_AFTER_PITCH = 2


def load_keyframes(
    project: Project,
    *,
    max_views: int | None = None,
    pitch_limit_deg: float = _PITCH_NEAR_HORIZON_DEG,
) -> list[dict[str, Any]]:
    """Load aligned cameras as interpolate-shaped frame dicts.

    Reads ``align/cameras.json``, drops missing ``shot_path`` files, prefers
    pitch near 0 when at least two such views remain, then optionally
    evenly subsamples to ``max_views``.
    """
    cameras_path = project.align_dir / "cameras.json"
    if not cameras_path.is_file():
        log.warning("no cameras.json at %s", cameras_path)
        return []

    cameras = project.read_json(cameras_path)
    if not isinstance(cameras, list):
        raise RuntimeError(f"expected list in {cameras_path}")

    existing: list[dict[str, Any]] = []
    for cam in cameras:
        shot = cam.get("shot_path")
        if not shot or not Path(shot).is_file():
            continue
        existing.append(cam)

    if not existing:
        log.warning("cameras.json has no existing shot_path files")
        return []

    near = [
        c
        for c in existing
        if abs(float(c.get("pitch") or 0.0)) <= pitch_limit_deg
    ]
    chosen = near if len(near) >= _MIN_VIEWS_AFTER_PITCH else existing
    if near and chosen is near and len(near) < len(existing):
        log.info(
            "keyframes: prefer |pitch|<=%.0f → %s / %s with files",
            pitch_limit_deg,
            len(near),
            len(existing),
        )
    elif chosen is existing and near and len(near) < _MIN_VIEWS_AFTER_PITCH:
        log.info(
            "keyframes: only %s near-horizon shots; keeping all %s with files",
            len(near),
            len(existing),
        )

    if max_views is not None and max_views > 0 and len(chosen) > max_views:
        before = len(chosen)
        chosen = subsample_evenly(chosen, int(max_views))
        log.info("keyframes: subsampled %s → %s (max_views)", before, len(chosen))

    return [_camera_to_frame(cam) for cam in chosen]


def _camera_to_frame(cam: dict[str, Any]) -> dict[str, Any]:
    """Shape a camera like an interpolate frame dict."""
    out: dict[str, Any] = {
        "path": cam["shot_path"],
        "e": float(cam["e"]),
        "n": float(cam["n"]),
        "u": float(cam["u"]),
        "heading": float(cam["heading"]),
        "pitch": float(cam.get("pitch") or 0.0),
        "fov": float(cam.get("fov") or 90.0),
        "mask": cam.get("mask"),
        "pano_id": cam.get("pano_id"),
        "interpolated": False,
    }
    # PINHOLE K needs size — cameras.json usually has them after crop/align.
    if cam.get("width") is not None:
        out["width"] = int(cam["width"])
    if cam.get("height") is not None:
        out["height"] = int(cam["height"])
    return out
