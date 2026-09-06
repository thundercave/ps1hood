"""OSM building footprints via Overpass — optional align prior only.

Not the Studio hero mesh. Product geometry must come from Street View
photos (multi-view stereo / plane facades). These extrusions may seed
camera seating when 3DBAG is missing; they must not replace photo recon.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any

from ps1_hood.capture.bag import _extrude
from ps1_hood.geo import BBox, LocalFrame
from ps1_hood.httputil import get_json
from ps1_hood.overpass import OVERPASS_URL

log = logging.getLogger(__name__)

LEVEL_HEIGHT_M = 3.0
DEFAULT_HEIGHT_M = 9.0
_HEIGHT_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*(m|meter|metres|meters)?\s*$", re.I)


def fetch_osm_buildings(bbox: BBox) -> dict[str, Any]:
    """Overpass: building ways + relations; recurse nodes; out body."""
    query = (
        "[out:json][timeout:90];"
        "("
        f'way["building"]({bbox.south},{bbox.west},{bbox.north},{bbox.east});'
        f'relation["building"]({bbox.south},{bbox.west},{bbox.north},{bbox.east});'
        ");"
        "(._;>;);"
        "out body;"
    )
    return get_json(OVERPASS_URL, params={"data": query}, timeout=120.0)


def _parse_meters(raw: Any) -> float | None:
    if raw is None:
        return None
    s = str(raw).strip()
    m = _HEIGHT_RE.match(s)
    if m:
        return max(2.0, float(m.group(1)))
    try:
        return max(2.0, float(s.split()[0].replace(",", ".")))
    except (TypeError, ValueError, IndexError):
        return None


def parse_height_m(tags: dict[str, Any] | None) -> float:
    """height / building:height → building:levels * 3 → else 9.0."""
    tags = tags or {}
    for key in ("height", "building:height"):
        got = _parse_meters(tags.get(key))
        if got is not None:
            return got
    levels = tags.get("building:levels") or tags.get("levels")
    if levels is not None:
        try:
            n = float(str(levels).split(";")[0].strip())
            if n > 0:
                return max(2.0, n * LEVEL_HEIGHT_M)
        except (TypeError, ValueError):
            pass
    return DEFAULT_HEIGHT_M


def _ring_from_geom(geom: list[dict[str, Any]] | None) -> list[tuple[float, float]]:
    """Overpass geometry nodes → closed lon/lat ring (lon, lat)."""
    if not geom or len(geom) < 3:
        return []
    ring = [(float(p["lon"]), float(p["lat"])) for p in geom if "lon" in p and "lat" in p]
    if len(ring) < 3:
        return []
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring


def _area_lonlat(ring: list[tuple[float, float]]) -> float:
    """Shoelace in lon/lat degrees² (sign = winding). Absolute area for sorting."""
    if len(ring) < 4:
        return 0.0
    pts = ring[:-1] if ring[0] == ring[-1] else ring
    a = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return abs(a) * 0.5


def _nodes_index(osm: dict[str, Any]) -> dict[int, tuple[float, float]]:
    nodes: dict[int, tuple[float, float]] = {}
    for el in osm.get("elements") or []:
        if el.get("type") == "node" and "lon" in el and "lat" in el:
            nodes[int(el["id"])] = (float(el["lon"]), float(el["lat"]))
    return nodes


def _ring_from_nodes(node_ids: list[Any], nodes: dict[int, tuple[float, float]]) -> list[tuple[float, float]]:
    ring: list[tuple[float, float]] = []
    for nid in node_ids or []:
        pt = nodes.get(int(nid))
        if pt is not None:
            ring.append(pt)
    if len(ring) < 3:
        return []
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring


def outer_rings_lonlat(osm: dict[str, Any]) -> list[tuple[Any, list[tuple[float, float]], dict[str, Any]]]:
    """Parse OSM elements → (id, lon/lat outer ring, tags).

    Assembles rings from node refs (out body). Falls back to embedded
    geometry when present (out geom). Ignores multipolygon inners (v1).
    """
    nodes = _nodes_index(osm)
    ways: dict[int, dict[str, Any]] = {}
    for el in osm.get("elements") or []:
        if el.get("type") == "way":
            ways[int(el["id"])] = el

    out: list[tuple[Any, list[tuple[float, float]], dict[str, Any]]] = []
    used_ways: set[int] = set()

    for el in osm.get("elements") or []:
        if el.get("type") != "relation":
            continue
        tags = el.get("tags") or {}
        if "building" not in tags:
            continue
        outers: list[list[tuple[float, float]]] = []
        for mem in el.get("members") or []:
            if mem.get("type") != "way":
                continue
            role = (mem.get("role") or "outer").lower()
            if role not in {"outer", ""}:
                continue  # skip inners v1
            wid = int(mem["ref"])
            way = ways.get(wid)
            ring: list[tuple[float, float]] = []
            if way is not None:
                used_ways.add(wid)
                ring = _ring_from_nodes(way.get("nodes") or [], nodes)
                if len(ring) < 4:
                    ring = _ring_from_geom(way.get("geometry"))
            if not ring:
                ring = _ring_from_geom(mem.get("geometry"))
            if len(ring) >= 4:
                outers.append(ring)
        if not outers:
            continue
        ring = max(outers, key=_area_lonlat)
        out.append((el.get("id"), ring, tags))

    for wid, way in ways.items():
        if wid in used_ways:
            continue
        tags = way.get("tags") or {}
        if "building" not in tags:
            continue
        ring = _ring_from_nodes(way.get("nodes") or [], nodes)
        if len(ring) < 4:
            ring = _ring_from_geom(way.get("geometry"))
        if len(ring) >= 4:
            out.append((way.get("id"), ring, tags))
    return out


def footprint_to_building(
    ident: Any,
    ring_lonlat: list[tuple[float, float]],
    height_m: float,
    frame: LocalFrame,
    *,
    tags: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Extrude one lon/lat footprint into an ENU prism matching bag buildings."""
    ring_xy: list[tuple[float, float]] = []
    for lon, lat in ring_lonlat:
        e, n, _ = frame.to_enu(lat, lon, 0.0)
        ring_xy.append((e, n))
    # Drop duplicate close
    if len(ring_xy) >= 2 and math.hypot(ring_xy[0][0] - ring_xy[-1][0], ring_xy[0][1] - ring_xy[-1][1]) < 1e-6:
        ring_xy = ring_xy[:-1]
    # Deduplicate consecutive
    cleaned: list[tuple[float, float]] = []
    for p in ring_xy:
        if not cleaned or math.hypot(p[0] - cleaned[-1][0], p[1] - cleaned[-1][1]) > 0.15:
            cleaned.append(p)
    if len(cleaned) >= 2 and math.hypot(cleaned[0][0] - cleaned[-1][0], cleaned[0][1] - cleaned[-1][1]) < 0.15:
        cleaned = cleaned[:-1]
    if len(cleaned) < 3:
        return None
    z0, z1 = 0.0, float(height_m)
    verts, faces = _extrude(cleaned, z0, z1)
    if not faces:
        return None
    xs = [p[0] for p in verts]
    ys = [p[1] for p in verts]
    zs = [p[2] for p in verts]
    return {
        "id": f"osm-{ident}" if ident is not None else "osm",
        "dak_type": None,
        "h_maaiveld": z0,
        "h_dak": z1,
        "from_3d": False,
        "source": "osm",
        "height_m": height_m,
        "tags": {k: tags[k] for k in ("building", "height", "building:height", "building:levels", "name") if tags and k in tags},
        "vertices": verts,
        "faces": faces,
        "edges": _edges_from_faces(faces),
        "bbox": [min(xs), min(ys), min(zs), max(xs), max(ys), max(zs)],
    }


def buildings_from_osm(
    osm: dict[str, Any],
    frame: LocalFrame,
    *,
    clip: BBox | None = None,
) -> list[dict[str, Any]]:
    """Convert Overpass payload → list of bag-shaped building dicts in ENU."""
    out: list[dict[str, Any]] = []
    for ident, ring, tags in outer_rings_lonlat(osm):
        height = parse_height_m(tags)
        bld = footprint_to_building(ident, ring, height, frame, tags=tags)
        if bld is None:
            continue
        if clip is not None:
            sw = frame.to_enu(clip.south, clip.west)
            ne = frame.to_enu(clip.north, clip.east)
            pad = 35.0
            e0, e1 = min(sw[0], ne[0]) - pad, max(sw[0], ne[0]) + pad
            n0, n1 = min(sw[1], ne[1]) - pad, max(sw[1], ne[1]) + pad
            be0, bn0, _, be1, bn1, _ = bld["bbox"]
            if be1 < e0 or be0 > e1 or bn1 < n0 or bn0 > n1:
                continue
        out.append(bld)
    return out


def fetch_buildings_enu(bbox: BBox, frame: LocalFrame, *, clip: BBox | None = None) -> list[dict[str, Any]]:
    """Network fetch + parse + extrude for a bbox."""
    raw = fetch_osm_buildings(bbox)
    buildings = buildings_from_osm(raw, frame, clip=clip if clip is not None else bbox)
    log.info("OSM buildings: %s footprints in bbox", len(buildings))
    return buildings
