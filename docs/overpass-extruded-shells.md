> **Course-correct:** OSM/BAG extruded shells are **optional align priors only**, NOT Studio hero geometry. Product mesh comes from Street View multi-view stereo.

# Overpass → extruded building shells (PS1 Hood fallback)

**Ask:** 3DBAG zip missing → OSM footprints for NL smoke bbox → LoD1 extruded shells.  
**Fits existing:** `src/ps1_hood/overpass.py` already hits `overpass-api.de` for roads via `get_json`.

## Live smoke bbox check (2026-09-06)

`52.0889,5.1180,52.0904,5.1210` → **236 building ways + 2 building relations**.  
Tag hits: `building:levels` on **8** only; almost no `height`. Plan on **fallback heights**.

## Overpass QL (paste)

```
[out:json][timeout:60];
(
  way["building"]({{bbox}});
  relation["building"]({{bbox}});
);
(._;>;);
out body;
```

bbox order: `south,west,north,east` (same as roads).

## Height fallback

1. `height` or `building:height` (metres, strip `m`)
2. else `building:levels` × **3.0** (+ optional `building:min_level`)
3. else default **9.0** m (3 storeys) — tweak per area
4. Optional: churches/parking get type-based defaults

## Best links (crisp)

1. Overpass building query patterns — https://wiki.openstreetmap.org/wiki/Overpass_API/Overpass_QL  
2. OSM building keys — https://wiki.openstreetmap.org/wiki/Key:building · https://wiki.openstreetmap.org/wiki/Key:building:levels  
3. osmnx footprints (GPL-3) — `ox.features_from_bbox(..., tags={"building":True})` — https://osmnx.readthedocs.io/  
4. Simple LoD1 extrusion mental model — CityJSON LoD1 / OSM2World notes — https://wiki.openstreetmap.org/wiki/OSM-3D · https://wiki.openstreetmap.org/wiki/Simple_3D_buildings  
5. shapely polygonize + trimesh extrude — https://trimsh.org/trimesh.creation.html#trimesh.creation.extrude_polygon  

Prefer **stay in-repo** (extend `overpass.py`) over adding osmnx unless you want GeoPandas.

## Pitfalls

- **Multipolygon relations:** outer/inner roles; extrude outer, subtract inners (or skip holes v1).
- **Unclosed ways / incomplete members:** skip if <3 distinct nodes.
- **CRS:** convert lon/lat → your `LocalFrame` ENU before extrude (same as roads).
- **Rate limits:** reuse `OVERPASS_URL` + existing httputil; small bbox OK; backoff on 429/504.
- **Courtyards:** inner rings ignored → solid block (fine for PS1 cardboard).

## Pasteable sketch (match `overpass.py`)

```python
BUILDING_QUERY = (
    "[out:json][timeout:60];"
    "(way[\"building\"]({s},{w},{n},{e});"
    "relation[\"building\"]({s},{w},{n},{e}););"
    "(._;>;);out body;"
)

def _height_m(tags: dict, default: float = 9.0) -> float:
    for key in ("height", "building:height"):
        if key in tags:
            raw = tags[key].lower().replace("m", "").strip()
            try:
                return float(raw)
            except ValueError:
                pass
    if "building:levels" in tags:
        try:
            return float(tags["building:levels"]) * 3.0
        except ValueError:
            pass
    return default

def fetch_buildings(bbox: BBox) -> dict:
    q = BUILDING_QUERY.format(s=bbox.south, w=bbox.west, n=bbox.north, e=bbox.east)
    return get_json(OVERPASS_URL, params={"data": q}, timeout=90.0)

def buildings_to_shells(osm: dict, frame: LocalFrame, default_h: float = 9.0):
    """Return list of {footprint_enu: [(e,n),...], height_m: float, osm_id: int}."""
    nodes = {int(e["id"]): (float(e["lon"]), float(e["lat"]))
             for e in osm.get("elements", []) if e.get("type") == "node"}
    shells = []
    for el in osm.get("elements", []):
        if el.get("type") != "way" or "building" not in el.get("tags", {}):
            continue
        ll = [nodes[n] for n in el.get("nodes", []) if n in nodes]
        if len(ll) < 3:
            continue
        enu = [(frame.to_enu(lat, lon)[0], frame.to_enu(lat, lon)[1]) for lon, lat in ll]
        # drop closing duplicate
        if enu[0] == enu[-1]:
            enu = enu[:-1]
        shells.append({
            "osm_id": int(el["id"]),
            "footprint_enu": enu,
            "height_m": _height_m(el.get("tags", {}), default_h),
            "tags": el.get("tags", {}),
        })
    return shells

def extrude_shell(footprint_enu, height_m):
    """Verts + faces for open box (no floor). footprint (e,n) → xyz (e, h, n) or (e,n,h) — pick Studio convention."""
    bottom = [(e, 0.0, n) for e, n in footprint_enu]
    top = [(e, height_m, n) for e, n in footprint_enu]
    verts = bottom + top
    n = len(footprint_enu)
    faces = []
    # walls
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, j + n, i + n))  # quad
    # roof
    faces.append(tuple(range(n, 2 * n)))
    return verts, faces
```

**Implement first:** `fetch_buildings` + `buildings_to_shells` beside roads; extrude to OBJ/Three BufferGeometry; skip relation inners v1; default 9 m.

## Explicit non-goal
Full Simple-3D-Buildings roof shapes — LoD1 boxes are enough for textured-facade targets.
