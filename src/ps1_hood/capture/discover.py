"""Find every Street View / Mapillary sample inside a bbox."""

from __future__ import annotations

import logging
from typing import Any

from ps1_hood.capture.google_static import lookup_pano
from ps1_hood.capture.mapillary import list_images_in_bbox
from ps1_hood.config import Settings
from ps1_hood.geo import BBox, sample_polyline_m
from ps1_hood.overpass import fetch_roads, roads_to_lonlat_lines

log = logging.getLogger(__name__)


def discover(bbox: BBox, spec_source: str, settings: Settings, spacing_m: float) -> dict[str, Any]:
    osm = fetch_roads(bbox)
    lines = roads_to_lonlat_lines(osm)
    seeds = []
    for line in lines:
        seeds.extend(sample_polyline_m(line, spacing_m))
    # always include the centre so empty OSM still has a probe
    clat, clon = bbox.center()
    seeds.append((clat, clon, 0.0))

    if spec_source == "mapillary":
        panos = list_images_in_bbox(bbox, settings.mapillary_token)
        return {
            "source": "mapillary",
            "seed_count": len(seeds),
            "panos": panos,
            "osm_way_count": len(lines),
        }

    if spec_source == "google_web":
        # No key: Chromium confirms coverage when it actually opens each seed.
        panos = [
            {
                "pano_id": f"seed-{i:04d}",
                "lat": lat,
                "lon": lon,
                "travel_heading": heading,
                "provider": "google_web",
                "provisional": True,
            }
            for i, (lat, lon, heading) in enumerate(seeds)
        ]
        return {
            "source": "google_web",
            "seed_count": len(seeds),
            "panos": panos,
            "osm_way_count": len(lines),
        }

    if not settings.google_maps_api_key:
        raise RuntimeError(
            "This source needs GOOGLE_MAPS_API_KEY. Use --source google_web "
            "to screengrab public Street View with no key."
        )

    by_id: dict[str, dict[str, Any]] = {}
    for lat, lon, heading in seeds:
        meta = lookup_pano(lat, lon, settings.google_maps_api_key)
        if not meta:
            continue
        if not bbox.contains_m(meta["lat"], meta["lon"], slop_m=25.0):
            continue
        pid = meta["pano_id"]
        if pid not in by_id:
            by_id[pid] = {
                **meta,
                "travel_heading": heading,
                "links": [],
            }
    return {
        "source": spec_source,
        "seed_count": len(seeds),
        "panos": list(by_id.values()),
        "osm_way_count": len(lines),
    }
