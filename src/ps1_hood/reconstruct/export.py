"""JSON dumps the viewer and later tools can load."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ps1_hood.geo import BBox, LocalFrame


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
    return {
        "name": spec_name,
        "bbox": bbox.as_dict(),
        "origin": {"lat": frame.lat0, "lon": frame.lon0},
        "cameras": cameras,
        "buildings": buildings or [],
        "frame_count": len(frames),
        "satellite": satellite,
        "cloud": cloud,
    }


def write_scene(path: Path, payload: dict[str, Any]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
