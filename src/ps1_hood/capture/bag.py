"""3DBAG from the national GeoPackage dump (SOZip) — bbox query, no full unzip."""

from __future__ import annotations

import logging
import os
import zipfile
from pathlib import Path
from typing import Any

from shapely import from_wkb
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon

from ps1_hood.geo import BBox, LocalFrame, rd_to_wgs84, wgs84_to_rd

log = logging.getLogger(__name__)

BAG_ZIP_URL = "https://data.3dbag.nl/v20250903/3dbag_nl.gpkg.zip"
BAG_ZIP_BYTES = 19_646_451_775


def bag_zip_path() -> Path:
    env = os.environ.get("PS1HOOD_BAG_GPKG", "").strip()
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3] / "data" / "3dbag_nl.gpkg.zip"


def zip_status(path: Path | None = None) -> dict[str, Any]:
    path = path or bag_zip_path()
    have = path.stat().st_size if path.is_file() else 0
    return {
        "path": str(path),
        "have": have,
        "expected": BAG_ZIP_BYTES,
        "complete": have == BAG_ZIP_BYTES,
        "percent": round(100.0 * have / BAG_ZIP_BYTES, 2) if BAG_ZIP_BYTES else 0.0,
    }


def _rd_bbox(bbox: BBox, pad_m: float = 40.0) -> tuple[float, float, float, float]:
    # Convert all four WGS corners; a two-corner axis-aligned map drifts
    # at Groningen scale if the RD grid is rotated even a little.
    corners = [
        wgs84_to_rd(bbox.south, bbox.west),
        wgs84_to_rd(bbox.south, bbox.east),
        wgs84_to_rd(bbox.north, bbox.west),
        wgs84_to_rd(bbox.north, bbox.east),
    ]
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    return (
        min(xs) - pad_m,
        min(ys) - pad_m,
        max(xs) + pad_m,
        max(ys) + pad_m,
    )


def _vsi_gpkg(zip_path: Path) -> str:
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".gpkg")]
    if not names:
        raise RuntimeError(f"no .gpkg inside {zip_path}")
    inner = names[0]
    return f"/vsizip/{zip_path}/{inner}"


def _pick_layer(vsi: str) -> str:
    import pyogrio

    layers = pyogrio.list_layers(vsi)
    names = [str(row[0]) for row in layers]
    log.info("3DBAG gpkg layers: %s", ", ".join(names[:20]))
    for needle in ("lod22", "lod2.2", "lod2_2"):
        for name in names:
            if needle in name.lower() and "2d" not in name.lower():
                return name
    for needle in ("lod13", "lod12", "pand"):
        for name in names:
            if needle in name.lower():
                return name
    if not names:
        raise RuntimeError("3DBAG gpkg has no layers")
    return names[0]


def query_gpkg(bbox: BBox, zip_path: Path | None = None) -> list[dict[str, Any]]:
    import pyogrio

    zip_path = zip_path or bag_zip_path()
    st = zip_status(zip_path)
    if not st["complete"]:
        raise RuntimeError(
            f"3DBAG dump still downloading: {st['percent']}% "
            f"({st['have']}/{st['expected']} bytes) → {st['path']}"
        )
    vsi = _vsi_gpkg(zip_path)
    layer = _pick_layer(vsi)
    west, south, east, north = _rd_bbox(bbox)
    meta, table = pyogrio.read_arrow(vsi, layer=layer, bbox=(west, south, east, north))
    geom_col = meta.get("geometry_name") or "wkb_geometry"
    rows: list[dict[str, Any]] = []
    # pyarrow table
    cols = {name: table.column(name) for name in table.column_names}
    n = table.num_rows
    log.info("3DBAG %s bbox hit %s rows", layer, n)
    for i in range(n):
        rec: dict[str, Any] = {}
        for name, col in cols.items():
            val = col[i].as_py()
            rec[name] = val
        wkb = rec.get(geom_col) or rec.get("wkb_geometry") or rec.get("geom")
        if wkb is None:
            continue
        if isinstance(wkb, dict) and "wkb" in wkb:
            wkb = wkb["wkb"]
        try:
            geom = from_wkb(bytes(wkb) if not isinstance(wkb, bytes) else wkb)
        except Exception:
            continue
        rec["_geom"] = geom
        rows.append(rec)
    return rows


def _polygons(geom: Any) -> list[Polygon]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    if isinstance(geom, GeometryCollection):
        out: list[Polygon] = []
        for g in geom.geoms:
            out.extend(_polygons(g))
        return out
    return []


def _coords3(poly: Polygon) -> list[tuple[float, float, float]]:
    coords = []
    for x, y, *rest in poly.exterior.coords:
        z = float(rest[0]) if rest else 0.0
        coords.append((float(x), float(y), z))
    return coords


def _triangulate_pts(pts: list[tuple[float, float, float]]) -> list[tuple[int, int, int]]:
    if len(pts) < 3:
        return []
    n = len(pts)
    if pts[0] == pts[-1]:
        n -= 1
    return [(0, i, i + 1) for i in range(1, n - 1)]


def _extrude(
    ring_xy: list[tuple[float, float]], z0: float, z1: float
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    if ring_xy[0] == ring_xy[-1]:
        ring_xy = ring_xy[:-1]
    n = len(ring_xy)
    if n < 3:
        return [], []
    verts = [(x, y, z0) for x, y in ring_xy] + [(x, y, z1) for x, y in ring_xy]
    faces: list[tuple[int, int, int]] = []
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, n + j))
        faces.append((i, n + j, n + i))
    for i in range(1, n - 1):
        faces.append((n, n + i, n + i + 1))
    return verts, faces


def rows_to_buildings(rows: list[dict[str, Any]], frame: LocalFrame) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rec in rows:
        geom = rec.get("_geom")
        ident = rec.get("identificatie") or rec.get("fid") or rec.get("identificatie_pand")
        dak = rec.get("b3_dak_type")
        rings = [_coords3(poly) for poly in _polygons(geom)]
        zs_all = [c[2] for coords in rings for c in coords]
        z0 = float(rec.get("b3_h_maaiveld") or rec.get("h_maaiveld") or (min(zs_all) if zs_all else 0.0))
        z1 = float(
            rec.get("b3_h_dak_70p")
            or rec.get("b3_h_dak_max")
            or rec.get("h_dak")
            or (max(zs_all) if zs_all and max(zs_all) - min(zs_all) > 2.0 else z0 + 8.0)
        )
        verts_rd: list[tuple[float, float, float]] = []
        faces: list[tuple[int, int, int]] = []
        used_3d = False
        # A ground ring at z ≈ −0.3 still "has Z". Only treat a polygon as a
        # real wall/roof if its own vertices actually span height.
        for coords in rings:
            if len(coords) < 4:
                continue
            zspan = max(c[2] for c in coords) - min(c[2] for c in coords)
            if zspan > 1.0:
                used_3d = True
                base = len(verts_rd)
                body = coords[:-1] if coords[0] == coords[-1] else coords
                verts_rd.extend(body)
                for a, b, c in _triangulate_pts(body):
                    faces.append((base + a, base + b, base + c))
        if not used_3d:
            ring = max(rings, key=len) if rings else []
            if ring:
                ev, ef = _extrude([(c[0], c[1]) for c in ring], z0, z1)
                verts_rd.extend(ev)
                faces.extend(ef)
        if not faces:
            continue
        enu = []
        for x, y, z in verts_rd:
            lat, lon = rd_to_wgs84(x, y)
            e, n, u = frame.to_enu(lat, lon, z)
            enu.append((e, n, u))
        edges: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for a, b, c in faces:
            for u, v in ((a, b), (b, c), (c, a)):
                key = (min(u, v), max(u, v))
                if key not in seen:
                    seen.add(key)
                    edges.append(key)
        xs = [p[0] for p in enu]
        ys = [p[1] for p in enu]
        zs = [p[2] for p in enu]
        out.append(
            {
                "id": ident,
                "dak_type": dak,
                "h_maaiveld": z0,
                "h_dak": z1,
                "from_3d": used_3d,
                "vertices": enu,
                "faces": faces,
                "edges": edges,
                "bbox": [min(xs), min(ys), min(zs), max(xs), max(ys), max(zs)],
            }
        )
    return out


def fetch_bag(bbox: BBox, dest: Path | None = None) -> dict[str, Any]:
    """Query the national dump for this bbox. Does not unpack the 18 GB zip."""
    del dest  # tiles cache unused; dump lives in data/
    st = zip_status()
    if not st["complete"]:
        return {"gpkg_rows": [], "count": 0, "error": f"dump {st['percent']}% downloaded"}
    rows = query_gpkg(bbox)
    return {"gpkg_rows": rows, "count": len(rows), "zip": st["path"]}


def buildings_enu(bag: dict[str, Any], frame: LocalFrame, clip: BBox | None = None) -> list[dict[str, Any]]:
    if bag.get("gpkg_rows") is None:
        return []
    out = rows_to_buildings(bag["gpkg_rows"], frame)
    if clip is None:
        return out
    sw = frame.to_enu(clip.south, clip.west)
    ne = frame.to_enu(clip.north, clip.east)
    # keep façades a street-width outside the drawn rectangle
    pad = 35.0
    e0, e1 = min(sw[0], ne[0]) - pad, max(sw[0], ne[0]) + pad
    n0, n1 = min(sw[1], ne[1]) - pad, max(sw[1], ne[1]) + pad
    kept = []
    for b in out:
        be0, bn0, _, be1, bn1, _ = b["bbox"]
        if be1 < e0 or be0 > e1 or bn1 < n0 or bn0 > n1:
            continue
        kept.append(b)
    return kept


def live_buildings(buildings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    slim = []
    for b in buildings:
        verts = b["vertices"]
        faces = b["faces"]
        slim.append(
            {
                "id": b["id"],
                "dak_type": b.get("dak_type"),
                "vertices": [[round(v[0], 3), round(v[1], 3), round(v[2], 3)] for v in verts],
                "faces": [list(f) for f in faces],
            }
        )
    return slim
