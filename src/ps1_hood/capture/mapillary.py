"""Mapillary v4 — open street-level photos with computed poses."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ps1_hood.config import ProjectSpec
from ps1_hood.geo import BBox
from ps1_hood.httputil import get_bytes, get_json

log = logging.getLogger(__name__)

GRAPH = "https://graph.mapillary.com"
FIELDS = (
    "id,computed_geometry,geometry,computed_compass_angle,compass_angle,"
    "captured_at,thumb_2048_url,thumb_1024_url,is_pano,sequence,camera_type"
)


def list_images_in_bbox(bbox: BBox, token: str) -> list[dict[str, Any]]:
    if not token:
        raise RuntimeError(
            "MAPILLARY_TOKEN is required. Create a client token at "
            "https://www.mapillary.com/dashboard/developers"
        )
    if bbox.area_deg2() >= 0.01:
        raise RuntimeError(
            "Mapillary bbox search must be < 0.01 deg². Draw a smaller block "
            "(a couple of streets) or split the area."
        )
    data = get_json(
        f"{GRAPH}/images",
        params={
            "access_token": token,
            "fields": FIELDS,
            "bbox": f"{bbox.west},{bbox.south},{bbox.east},{bbox.north}",
            "limit": 2000,
        },
    )
    panos: list[dict[str, Any]] = []
    for item in data.get("data", []):
        geom = item.get("computed_geometry") or item.get("geometry") or {}
        coords = (geom.get("coordinates") or [None, None])
        lon, lat = coords[0], coords[1]
        if lat is None or lon is None:
            continue
        heading = item.get("computed_compass_angle")
        if heading is None:
            heading = item.get("compass_angle") or 0.0
        panos.append(
            {
                "pano_id": str(item["id"]),
                "lat": float(lat),
                "lon": float(lon),
                "travel_heading": float(heading),
                "date": item.get("captured_at"),
                "is_pano": bool(item.get("is_pano")),
                "thumb": item.get("thumb_2048_url") or item.get("thumb_1024_url"),
                "sequence": item.get("sequence"),
                "provider": "mapillary",
            }
        )
    return panos


def capture_panos(
    panos: list[dict[str, Any]],
    spec: ProjectSpec,
    dest: Path,
    token: str,
) -> list[dict[str, Any]]:
    del spec, token  # Mapillary gives one image (or pano thumb) per capture
    shots: list[dict[str, Any]] = []
    dest.mkdir(parents=True, exist_ok=True)
    for i, pano in enumerate(panos):
        url = pano.get("thumb")
        if not url:
            continue
        pdir = dest / pano["pano_id"]
        pdir.mkdir(parents=True, exist_ok=True)
        path = pdir / "view.jpg"
        if not path.exists():
            log.info("mapillary %s/%s  %s", i + 1, len(panos), pano["pano_id"])
            path.write_bytes(get_bytes(url, timeout=60.0))
        shots.append(
            {
                "pano_id": pano["pano_id"],
                "lat": pano["lat"],
                "lon": pano["lon"],
                "heading": float(pano.get("travel_heading") or 0.0),
                "pitch": 0.0,
                "fov": 90.0,
                "rel_heading": 0,
                "path": str(path),
                "source": "mapillary",
            }
        )
    return shots
