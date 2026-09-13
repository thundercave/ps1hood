# Studio layer alignment survey (`ps1hood`)

**Date:** 2026-09-13 (Europe/Amsterdam)  
**Scope:** existing `/workspace/ps1hood` only — how Studio places BAG, satellite, Street View cameras, and recon cloud; likely BAG/sat vs SV/cloud mismatch causes.  
**No PR** — survey only.

---

## 1. Files / symbols that write or load each layer

| Layer | Write (pipeline / capture) | Load / place (Studio) |
|-------|----------------------------|------------------------|
| **scene.json** | `reconstruct/export.py` → `scene_payload()` + `write_scene()`; called from `pipeline.stage_reconstruct` → `recon/scene.json` | `studio/app.py` `GET /api/runs/<name>/scene`; `studio/static/viewer.html` fetches it |
| **BAG shells** | `capture/bag.py` (`fetch_bag`, `rows_to_buildings`, `buildings_enu`, `live_buildings`); `pipeline.stage_bag` → `bag/buildings.json`; slim copy into `live.json` + `scene.json.buildings` via `_write_live` / `scene_payload` | **Live pane:** `index.html` `rebuildLive()` → `bagGroup` from `/api/runs/.../live`. **Viewer:** `viewer.html` loops `sceneJson.buildings` (always on; no toggle) |
| **Satellite / ortho** | `capture/satellite.py` `fetch_satellite` → `satellite/ortho.jpg` + `ortho.json` (`crs: EPSG:4326`); `Ortho` ENU↔px helper | `app.py` `GET .../satellite.jpg` serves `ortho.jpg`. **Viewer** builds a textured ground `PlaneGeometry` from cameras (not Ortho ENU). Facades ground bake: `facades._ground_satellite_texture` (correct Ortho crop) |
| **ENU origin** | `geo.LocalFrame.from_bbox` (bbox centre, `alt0=0`); written as `scene.json.origin` / `live.json.origin` | Viewer remaps ENU→Three with `to3(e,n,u)=(e,u,-n)` + `remapEnuGeometry`; does **not** re-derive LocalFrame in JS (assumes stored ENU metres) |
| **SV camera poses** | `align/pose_graph.initial_poses` / `refine_poses`; optional `align/bag_edges.snap_camera_to_bag`; `pipeline.stage_align` → `align/poses.json`, `align/cameras.json`; exploded into scene cameras | `viewer.html` / `index.html`: position `to3(e,n,u)`, `rotation.y = -heading°`, billboard from cropped JPEG |
| **Recon cloud** | Flow / COLMAP posed / MapAnything → `recon/cloud.ply` (Studio load path); copies also `cloud_photo.ply`, `cloud_mapanything.ply` | `viewer.html` PLYLoader + `remapEnuGeometry`; toggle `#togCloud` |
| **Facades (related)** | `reconstruct/facades.extract_facades` → `facades.obj/.mtl` + textures | Viewer MTL/OBJ load + ENU remap; toggle `#togFacades` |

Align priors (not Studio hero, but affect poses): `align/satellite_align.py`, `align/bag_edges.py`, `align/seat.py`, OSM roads via `overpass` / `snap_to_polylines_enu`.

---

## 2. CRS / frame each layer assumes

| Layer | Assumed frame | Notes |
|-------|---------------|--------|
| **LocalFrame / scene cameras / cloud / facades** | **ENU metres**, origin = bbox centre WGS84, `alt0=0` (ellipsoid). X=east, Y=north, Z=up | Sacred product frame (`docs/correct-geometry-ghost-duplicates.md`, MapAnything lock) |
| **BAG** | Source **EPSG:28992 / RD New** XY (+ NAP heights in attributes / geom Z); converted `rd_to_wgs84` then `LocalFrame.to_enu(lat, lon, z)` | Z treated as WGS84 ellipsoid altitude numerically (NAP≠ellipsoid; OK-ish only where NAP≈0 relative to cameras’ forced `u=2.5`) |
| **Satellite ortho file** | **EPSG:4326** lon/lat bbox export (Esri World Imagery); pixels linear in lon/lat | Meta in `ortho.json`. Python `Ortho` maps via **ENU of bbox corners** (not raw lon/lat in the viewer) |
| **Viewer sat ground plane** | Intended ENU, but **implemented as camera-hull AABB + full ortho texture** | **Not** georeferenced to `satellite.bbox` ENU — see §4 |
| **Three.js display** | Remap `(e,n,u) → (e, u, -n)` | Shared by BAG, cams, cloud, facades |

Smoke run check (`runs/smoke`): Ortho ENU centre ≈ `(0,0)`; viewer plane centre from cameras ≈ `(-12.5, +8.0)` m → **~12.5 m E / 8 m N shift** of sat under the same ENU cameras.

---

## 3. Known TODO / FIXME about alignment

- **No literal `TODO`/`FIXME` in `studio/`** about layer CRS.
- Product docs (effective TODOs):
  - **Hide 3DBAG cardboard shells from hero Studio view** — `docs/compare-and-pathforward.md` §7; `docs/overpass-extruded-shells.md`; `docs/photo-consistent-facades.md` (“Not the hero”); `docs/path-alpha-planarize-recipe.md` checklist “Studio opens facades.obj without BAG shells”.
  - **MapAnything wrong-frame** (raw `pts3d` as ENU) — documented + **PR #14 fix landed**; gate notes in `docs/mapanything-densify.md` / compare doc. Still a failure mode if an old `cloud.ply` is left in place.
  - Align stage logs **“SV vs 3DBAG centroid … m apart”** (`pipeline.stage_align`) — residual after edge snap, not a Studio TODO.

Local smoke `bag/raw.json`: `"dump 0.0% downloaded"` → empty buildings (BAG layer absent in these runs).

---

## 4. Top 3 hypothesized causes of BAG/sat vs SV/cloud mismatch

1. **Viewer satellite plane is not placed in Ortho ENU (highest confidence).**  
   `viewer.html` sizes/centres the ground quad from **camera e/n hull + 16 m pad**, then maps the **entire** `ortho.jpg` (full WGS84 bbox) across that quad. Correct placement = ENU of `satellite.bbox` corners (same as `Ortho.sw/sh/ee/nn`). Smoke: **~12–13 m planar shift**. Facades ground bake *does* use `Ortho.enu_to_px` — so sat plane and `facades.obj` ground can disagree with each other.

2. **BAG still drawn as always-on overlay while product geometry is photo ENU.**  
   Cameras are optionally snapped *to* BAG edges (`snap_camera_to_bag`); cloud/facades come from SV multi-view in align ENU. Residual snap / incomplete dump / GPS prior leaves BAG shells offset from photo cloud. Docs already say BAG is align-prior only — viewer still renders `sceneJson.buildings` with no toggle (unlike cloud/facades).

3. **Vertical / altitude convention drift (BAG NAP-as-alt vs camera `u=2.5`).**  
   BAG vertices use NAP (or geom Z) as `to_enu(..., alt=z)`; SV poses force `u = camera_height_m` (2.5) above ellipsoid origin. Where local NAP maaiveld ≠ ~0, BAG sits high/low vs cameras/cloud. Secondary vs (1); matters more for BAG vs cloud than for sat (sat is forced to `u=0` plane).

Honorable mention: stale MapAnything cloud in wrong frame (pre–PR #14) would float vs sat/BAG even if sat were fixed.

---

## 5. Smallest fix direction

**Prefer: transform / place overlays into photo ENU — and default-hide BAG.**

1. **Minimal sat fix (do this first):** in `viewer.html`, build the ground plane from `sceneJson.satellite.bbox` + `origin` (or precompute ENU corners into `scene.json`) using the same corner→ENU mapping as `Ortho` — size `(ee−sw)×(nn−sh)`, centre `((sw+ee)/2,(sh+nn)/2,0)`, keep `(e,u,-n)` remap. Do **not** size from camera hull. Optionally add `enu_sw`/`enu_ne` in `scene_payload` so JS need not reimplement LocalFrame.
2. **Hide BAG by default** in viewer (checkbox off) / omit `buildings` from hero scene payload; keep live `index.html` BAG for align debugging only.
3. Avoid inventing a second “photo ENU” Sim(3) for overlays — cameras/cloud already define the frame; sat/BAG should *join* that ENU, not the reverse.

**Not recommended as first move:** hide sat entirely, or re-solve cameras into BAG/RD — product rule is photo ENU sacred.

---

## Quick symbol index

- Write scene: `ps1_hood.reconstruct.export.scene_payload` / `write_scene`
- Serve: `ps1_hood.studio.app` (`/scene`, `/cloud.ply`, `/satellite.jpg`, `/live`, facades)
- Place: `studio/static/viewer.html`, `studio/static/index.html` (`rebuildLive`)
- ENU: `ps1_hood.geo.LocalFrame`
- Ortho ENU↔px: `ps1_hood.capture.satellite.Ortho`
- BAG→ENU: `ps1_hood.capture.bag.rows_to_buildings` / `buildings_enu`
