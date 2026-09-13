"""JSON dumps the viewer and later tools can load."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ps1_hood.capture.satellite import Ortho
from ps1_hood.geo import BBox, LocalFrame


def satellite_with_enu(
    satellite: dict[str, Any] | None, frame: LocalFrame
) -> dict[str, Any] | None:
    """Copy satellite meta and attach Ortho ENU corners for Studio placement."""
    if not satellite:
        return None
    out = dict(satellite)
    raw = out.get("bbox")
    if not isinstance(raw, dict):
        return out
    bbox = BBox.from_dict(raw)
    sw, sh, ee, nn = Ortho.enu_corners(bbox, frame)
    out["enu"] = {
        "sw": sw,
        "sh": sh,
        "ee": ee,
        "nn": nn,
        "width_m": ee - sw,
        "height_m": nn - sh,
        "centre_e": (sw + ee) / 2.0,
        "centre_n": (sh + nn) / 2.0,
    }
    return out


def scene_payload(
    *,
    spec_name: str,
    bbox: BBox,
    frame: LocalFrame,
    poses: list[dict[str, Any]],
    frames: list[dict[str, Any]],
    satellite: dict[str, Any] | None,
    cloud: dict[str, Any] | None,
    buildings: list[dict[str, Any]] | None = None,
    georef: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cameras = []
    for p in poses:
        cameras.append(
            {
                "pano_id": p.get("pano_id"),
                "e": p["e"],
                "n": p["n"],
                "u": p["u"],
                "heading": p["heading"],
                "pitch": p["pitch"],
                "fov": p["fov"],
                "image": p.get("shot_path"),
                "sat_score": p.get("sat_score"),
                "residual_m": p.get("residual_m"),
                "bag_snapped": p.get("bag_snapped"),
            }
        )
    out: dict[str, Any] = {
        "name": spec_name,
        "bbox": bbox.as_dict(),
        "origin": {"lat": frame.lat0, "lon": frame.lon0},
        "cameras": cameras,
        "buildings": buildings or [],
        "frame_count": len(frames),
        "satellite": satellite_with_enu(satellite, frame),
        "cloud": cloud,
    }
    if georef is not None:
        out["georef"] = georef
    return out


def write_scene(path: Path, payload: dict[str, Any]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
