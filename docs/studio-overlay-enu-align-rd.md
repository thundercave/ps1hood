# Studio overlay ENU align — R&D pack

**Date:** 2026-09-13 (Europe/Amsterdam)  
**Goal:** One shared ENU frame so BAG / sat / SV cams / recon cloud agree on-street to ~metre on smoke-dense; later scene stitching does not fight overlays.  
**Sacred:** photo-derived geometry (MA ENU cloud ~579k + hybrid façades **10/8/0.418**) stays hero — do **not** invent OSM/BAG as mesh. Overlays transform to match SV cameras / recon ENU, or gated off until registered.

**Authorities folded here:** Stippy Studio/BAG path digest (verbatim §0) + Chief survey [`/workspace/studio-layer-alignment-survey.md`](studio-layer-alignment-survey.md).  
**Shipping now:** Chief PR with the three moves in §3 — this pack **endorses** them as first PR; ranked causes / acceptance / follow-ups only. Do not reopen scope.

---

## 0) Stippy digest (authoritative placement — fold verbatim)

- **scene.json:** `origin={lat,lon}` = `LocalFrame` centre; cameras `e,n,u`; buildings ENU verts; satellite/cloud meta.
- **Viewer:** `to3(e,n,u)=(e,u,-n)`; BAG + cloud + façades remapped; sat ground plane from ortho.
- **BAG:** EPSG:28992 → WGS84 → `LocalFrame` ENU; Z from `b3_h_maaiveld` / dak (NAP-ish).
- **Sat:** EPSG:4326 ortho → `LocalFrame` pixel↔ENU.
- **Cams:** GPS→ENU, `u=2.5` m default; `align/bag_edges` + `align/satellite_align` exist.
- **No CRS field in scene.json** — mismatch likely from stale origin / BAG Z / sat plane vs cloud height / remap bugs.

---

## 1) How Studio currently places layers

| Layer | Written by | Loaded / placed |
|-------|------------|-----------------|
| **scene.json** | `reconstruct/export.scene_payload` + `write_scene` from `pipeline.stage_reconstruct` → `recon/scene.json` | `GET /api/runs/<name>/scene` → `viewer.html` |
| **BAG shells** | `capture/bag.rows_to_buildings` / `buildings_enu` → `bag/buildings.json`; slim via `live_buildings` into `live.json` + `scene.buildings` | **Live:** `index.html` `rebuildLive` → `bagGroup`. **Viewer:** always-on loop over `sceneJson.buildings` (no toggle today) |
| **Satellite** | `capture/satellite.fetch_satellite` → `ortho.jpg` + `ortho.json` (`crs: EPSG:4326`); Python `Ortho` does correct ENU↔px | Viewer textured ground plane (**bug:** sized from camera hull, not Ortho ENU — Chief §4). Facades ground bake uses `Ortho` correctly |
| **SV cameras** | `align/pose_graph` + optional `bag_edges.snap_camera_to_bag` → poses/cameras; scene cameras `e,n,u` | `to3(e,n,u)`, `rotation.y = -heading°`, JPEG billboard |
| **Recon cloud / façades** | Flow / MA / COLMAP → `recon/cloud.ply`; façades OBJ/MTL | PLY/OBJ + `remapEnuGeometry`; toggles `#togCloud` / `#togFacades` |

**Display remap (shared):** ENU `(e,n,u)` → Three `(e, u, -n)`. Product frame = photo ENU (cameras + cloud + façades).

**smoke-dense snapshot:** `origin` matches `LocalFrame.from_bbox`; `buildings: []` (3DBAG dump 0%); cameras `u=2.5`; `bag_snapped` / `residual_m` null; sat meta present. Chief smoke check: Ortho ENU centre ≈ `(0,0)` vs viewer plane centre from cams ≈ `(-12.5, +8)` m → **~12–13 m planar sat shift**.

---

## 2) Ranked causes of misalignment

1. **Viewer sat plane ≠ Ortho ENU (highest confidence — Chief #1).**  
   `viewer.html` sizes/centres ground from **camera e/n AABB + 16 m** while stretching the **full** `ortho.jpg` (WGS84 bbox). Correct = ENU of `satellite.bbox` corners (`Ortho.sw/sh/ee/nn`). Smoke: ~12–13 m shift; smoke-dense similar (~1.5–1.8× scale + ~1.5 / 7.8 m centre Δ). Facades sat ground bake already correct → sat plane can disagree with `facades.obj` ground.

2. **BAG always drawn in product viewer while photo cloud is hero (Chief #2).**  
   Align may snap cams *to* BAG; product mesh is photo ENU. Residual snap / empty dump / GPS prior → shells fight cloud. Docs already: BAG = align prior only; viewer still renders buildings with no off-default.

3. **BAG Z / NAP-as-ellipsoid vs camera `u=2.5` (Chief #3).**  
   `to_enu(lat,lon,z)` with NAP / geom Z as `alt`; cams force `u=camera_height_m`. Where maaiveld ≉ 0 relative to ellipsoid origin, BAG high/low vs street. Secondary to (1); sat forced to `u=0`.

4. **Stale / incomplete scene.json (follow-up).**  
   No CRS / `overlay_registered` / shared-origin assert; old MA wrong-frame `cloud.ply` (pre-PR #14); empty BAG when dump incomplete; unused `const bw = sceneJson.bbox` in viewer is the smoking gun for (1).

5. **Y-up vs Z-up / remap bugs (lower alone).**  
   Shared `to3` is consistent for BAG/cams/cloud/façades. Historic Overpass extrude snippet used `(e,h,n)` Y-up — do not mix. Unlikely sole cause of ~10 m planar sat error.

6. **Re-solving cameras into BAG/RD (anti-pattern).**  
   Would fight photo ENU sacred rule — **out of first-PR scope** (Chief: do not).

---

## 3) First PR scope (smallest) — **ENDORSE Chief shipping PR**

Chief is shipping these three; this pack endorses them as the complete first PR. No expand.

| # | Change | Why |
|---|--------|-----|
| **1** | **Sat plane from `satellite.bbox` ENU** (same corner→ENU as `Ortho`), not camera hull | Fixes ~12 m smoke shift; joins photo ENU |
| **2** | **Default-hide BAG** in product viewer (checkbox off / omit from hero) | Sacred: unregistered cadastral shells off product view; live pane may keep BAG for align debug |
| **3** | **Do not re-solve cams into BAG** | Photo ENU / SV cameras stay frame anchors; overlays join them |

Optional tiny helpers inside same PR only if already in Chief branch: precompute `enu_sw`/`enu_ne` into `scene_payload`; assert `scene.origin` == `LocalFrame.from_bbox` centre. No Sim(3) second frame; no NAP datum rewrite in PR1.

---

## 4) Concrete fix recipe (PR1 + later)

### PR1 (now — Chief)

```text
viewer sat:  size=(ee−sw)×(nn−sh), centre=((sw+ee)/2,(sh+nn)/2,0) from satellite.bbox ENU
             keep to3(e,u,-n); stop using camera AABB for ground
BAG:         default hidden in viewer.html (togBag off / omit hero buildings)
poses:       do NOT snap product cameras into BAG/RD for Studio display
```

Wire existing tools later (not PR1 blockers):

- `align/satellite_align.align_camera_to_satellite` — already ENU via `Ortho`; refine poses when BAG absent (`use_satellite=not buildings`).
- `align/bag_edges.snap_camera_to_bag` — align-stage prior only; product view must not require it.
- Gate: `overlay_registered: false` until sat plane ENU + optional rigid BAG→photo ENU residual ≤ ~1–2 m.

### Follow-ups (after PR1 — do not block ship)

1. **NAP / Z:** relative ground: subtract local `h_maaiveld` (or NAP→ellipsoid via PROJ) so BAG `u` sits near recon `ground_z` / cam `u−2.5`.  
2. **Stale scene.json:** write `crs: "ENU+LocalFrame"`, `origin`, `satellite.enu_sw/ne`, `overlay_registered`; fail-loud if origin drifts from bbox centre.  
3. **Rigid BAG register (optional):** XY(+yaw) only onto photo ENU using bag_edges residual / façade plane anchors — or keep BAG gated off.  
4. **Meter gate:** assert cam centroid vs Ortho centre, BAG footprint vs cam hull, cloud z p50 vs cam u — see §5.

---

## 5) Acceptance tests / Studio checklist (smoke-dense)

**Automated / script**

- [ ] Sat plane centre vs Ortho ENU centre ‖Δxy‖ **≤ 1 m** (was ~12 m).  
- [ ] Sat plane size vs `|ee−sw|×|nn−sh|` within **~2%**.  
- [ ] `scene.origin` == `LocalFrame.from_bbox(bbox)` (lat/lon).  
- [ ] Product viewer: BAG **not visible** by default (`togBag` unchecked or no buildings mesh).  
- [ ] Cloud + façades still load; MA product / hybrid **10/8/0.418** untouched.  
- [ ] No code path that re-writes product poses from BAG for Studio hero.

**Visual Studio (smoke-dense)**

- [ ] Ortho roads/roofs sit under SV camera arrows to ~metre.  
- [ ] Photo cloud hugs street; façades line windows (hero).  
- [ ] Enabling BAG (debug) may still show residual height/XY — expected until follow-up NAP/register; must not be required for product read.

**Gate phrase:** *BAG edges / sat / cams / cloud agree on street to ~metre on smoke-dense* — sat+cams+cloud first (PR1); BAG edges once registered or left gated.

---

## 6) Report card

| Item | |
|------|--|
| **Pack path** | `/workspace/studio-overlay-enu-align-rd.md` |
| **Survey** | `/workspace/studio-layer-alignment-survey.md` (Chief) |
| **Top cause** | Viewer sat plane sized from camera hull, not `satellite.bbox` Ortho ENU (~12 m smoke shift) |
| **Top fix (PR1)** | Place sat from Ortho ENU + default-hide BAG + do not re-solve cams into BAG (Chief shipping) |
| **Sacred** | MA ENU cloud + hybrid façades hero; overlays join photo ENU or stay gated |

