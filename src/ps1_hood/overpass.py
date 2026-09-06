"""Pull OSM road centerlines for a bbox via Overpass."""

from __future__ import annotations

from typing import Any

from ps1_hood.geo import BBox, LocalFrame
from ps1_hood.httputil import get_json

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

HIGHWAY_FILTER = (
    '["highway"~"^(motorway|trunk|primary|secondary|tertiary|unclassified|'
    'residential|living_street|service|pedestrian|cycleway)$"]'
)


def fetch_roads(bbox: BBox) -> dict[str, Any]:
    query = (
        "[out:json][timeout:60];"
        f"way{HIGHWAY_FILTER}({bbox.south},{bbox.west},{bbox.north},{bbox.east});"
        "(._;>;);"
        "out body;"
    )
    return get_json(OVERPASS_URL, params={"data": query}, timeout=90.0)


def roads_to_lonlat_lines(osm: dict[str, Any]) -> list[list[tuple[float, float]]]:
    nodes: dict[int, tuple[float, float]] = {}
    for el in osm.get("elements", []):
        if el.get("type") == "node":
            nodes[int(el["id"])] = (float(el["lon"]), float(el["lat"]))
    lines: list[list[tuple[float, float]]] = []
    for el in osm.get("elements", []):
        if el.get("type") != "way":
            continue
        pts = [nodes[nid] for nid in el.get("nodes", []) if nid in nodes]
        if len(pts) >= 2:
            lines.append(pts)
    return lines


def roads_to_enu_lines(osm: dict[str, Any], frame: LocalFrame) -> list[list[tuple[float, float]]]:
    out: list[list[tuple[float, float]]] = []
    for line in roads_to_lonlat_lines(osm):
        enu: list[tuple[float, float]] = []
        for lon, lat in line:
            e, n, _ = frame.to_enu(lat, lon)
            enu.append((e, n))
        out.append(enu)
    return out
