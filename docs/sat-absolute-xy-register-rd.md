# Sat absolute XY register — R&D pack (PRIORITY)

**Date:** 2026-09-13 (Europe/Amsterdam)  
**Block:** PS1 Hood smoke-dense / product  
**Sacred (locked):** **WGS84 Esri ortho = absolute real-world XY.** All sources confirm each other against sat. SV currently BAG-aligned → must **re-register to sat**. BAG untrusted (below-ground / poor sat follow) — optional off or transform into same frame; **NOT hero mesh**. Photo multi-view keeps **relative 3D shape**; only georef/nudge to sat (Sim(2) / SE(2) XY+yaw, or seat against Ortho).

**Product today:** hybrid façades **10/8/0.418**; MA cloud **579613** ENU; Studio sat plane Ortho ENU (**PR #27** merged).  
**Prior packs:** [`studio-overlay-enu-align-rd.md`](studio-overlay-enu-align-rd.md), [`studio-layer-alignment-survey.md`](studio-layer-alignment-survey.md), `align/satellite_align.py`, `align/bag_edges.py`.

**Strategy flip vs overlay pack:** PR #27 made the *viewer* place sat correctly in LocalFrame ENU. This pack makes *poses + recon* join **sat street/rooflines**, not BAG edges. Photo relative structure stays; one rigid/SE(2) seats the whole graph on Ortho.

---

## 0) Where the pipeline snaps SV cams to BAG (disable / replace)

### Call chain (main @ `thundercave/ps1hood`)

| Step | Symbol | Behaviour |
|------|--------|-----------|
| 1 | `pipeline.stage_align` | GPS/OSM prior → `refine_poses` → optional BAG edge snap → explode cameras |
| 2 | `pose_graph.initial_poses` | GPS → ENU; OSM `snap_to_polylines_enu` (road centreline, not BAG) |
| 3 | `pose_graph.refine_poses(..., use_satellite=not buildings, use_features=not bool(buildings))` | **Sat NCC + feature SE(2) bundle ONLY when BAG buildings list is empty** |
| 4 | `seat.push_out_of_footprints` | Runs whenever footprints exist (BAG-derived), even before snap |
| 5 | `bag_edges.snap_camera_to_bag` | Per-pano coarse→fine edge DT vs projected 3DBAG wall/roof segs; overwrites `e,n,u,heading` |
| 6 | Write | `align/poses.json`, `align/cameras.json`, debug `overlay.jpg` |

### Critical inverted gate (today)

```text
# pipeline.stage_align (approx)
poses = refine_poses(..., use_satellite=not buildings, use_features=not bool(buildings))
...
if photo is not None and buildings:
    hit = snap_camera_to_bag(...)   # BAG becomes the XY prior
    pose[e/n/u] = hit[...]
```

- **BAG present** → sat refine **OFF**, feature bundle **OFF**, bag snap **ON**. Product SV frame = BAG-edge prior.  
- **BAG empty** (smoke-dense dump 0%) → sat refine **ON** (per-cam NCC), bag snap skipped. Smoke shows `sat_score` ~0.26 mean (range −1…0.76), GPSΔ ~0.7–9.5 m — sat path exists but is not the intentional product default.

### How to disable BAG snap / replace with sat

| Knob | Today | Target |
|------|-------|--------|
| Bag edge snap | Implicit `if buildings:` | Only when `--align-prior bag` |
| Sat refine | `use_satellite=not buildings` | Always on for `--align-prior sat` (default) |
| Feature SE(2) bundle | Off when BAG | On for sat prior (preserve relative) |
| `push_out_of_footprints` | Always if footprints | Off under sat prior (or after global seat, soft) |
| CLI | `ps1hood align <run>` no prior flag | `--align-prior {sat,bag}` (default **sat**) |

No need to delete `bag_edges.py` — gate it. Live pane may still show BAG for debug; product hero stays photo + sat.

---

## 1) Register LocalFrame / poses / cloud / façades to sat ortho

### Absolute XY authority

- **Ortho:** Esri World Imagery export `EPSG:4326` → `Ortho` maps lon/lat bbox corners through `LocalFrame` → ENU (`sw,sh,ee,nn`). PR #27 already places the Studio ground plane from these corners.  
- **LocalFrame origin** stays bbox centre (`scene.origin`) — do **not** invent a second CRS. Registration = SE(2)/Sim(2) **inside** that ENU so cams sit on sat streets.  
- BAG (RD New / NAP) is optional overlay: either hidden, or same SE(2) applied so it no longer fights sat.

### Existing tools to reuse

| Tool | Role for sat-absolute |
|------|----------------------|
| `align_camera_to_satellite` + Ortho NCC | Per-cam SE(2) observation: render sat ground into cam, maximize NCC vs photo lower half |
| `render_satellite_into_camera_fast` | Synth view for NCC |
| `write_debug_overlay` | Cam arrows on ortho — acceptance visual |
| Road edges (OSM / Canny on Ortho) | Optional lateral constraint (kerb/road paint) when NCC weak |
| Roof / façade shadows on Ortho | Edge DT analog of `collect_bag_edges` but from sat pixels (follow-up) |
| `_bundle_se2` | Soft GPS + neighbour relative — keep relative graph |

### Global Sim(2) / SE(2) vs per-cam (ranked)

1. **Preferred — one global SE(2) (or Sim(2) with s≈1):**  
   - Build relative graph first (GPS/OSM + feature `_bundle_se2`, **no** bag snap).  
   - Collect per-cam sat NCC peaks as *observations* (Δe, Δn, Δyaw), or match cam→road/roof edges on Ortho.  
   - Solve **one** `(tx, ty, yaw[, s])` that seats the whole pose set on Ortho.  
   - Apply same transform to cloud XYZ + façade verts (see §2).  
   - Preserves photo multi-view relative structure (sacred).

2. **Acceptable PR1 shortcut — sat refine ON + tight coupled nudge:**  
   - Re-enable `use_satellite=True` always under sat prior.  
   - Cap per-cam shift (`max_align_shift_m` already 8 m / heading 15°) and **re-run soft `_bundle_se2` after** sat obs so neighbours cannot drift apart.  
   - Still write a **summary** SE(2) into `scene.json` (centroid before→after + mean yaw) for reconstruct re-apply.

3. **Anti-pattern:** independent per-cam bag snap **or** unconstrained per-cam sat with features off — breaks relative 3D; cloud/façades no longer match cams after the fact.

### Seat recipe (conceptual)

```text
prior:     GPS → LocalFrame ENU; optional OSM centreline (not BAG)
relative:  feature SE(2) bundle (_bundle_se2)  OR keep MA/COLMAP relative locks
absolute:  Ortho NCC / road-roof edges → ONE SE(2) T_sat
apply T:   cams, cameras.json, cloud.ply, facades.obj, planes.json
BAG:       hide OR apply same T (debug only)
persist:   scene.json.georef = { prior: "sat", T_sat, scores }
```

Sim(2) only if Ortho pixel scale vs ENU metres drifts (rare if `Ortho.enu_corners` correct); default **SE(2)** (tx, ty, yaw), `u` / pitch untouched.

---

## 2) Keep photo multi-view relative structure

Sacred: MA ENU cloud + hybrid façades encode **relative** street geometry from locked SV poses. Absolute fix must be **one rigid (or SE(2)) motion of the whole set**.

| Asset | Apply `T_sat` how |
|-------|-------------------|
| `align/poses.json` / `cameras.json` | `(e',n') = R(yaw)·(e,n) + (tx,ty)`; `heading' = heading + yaw` |
| `recon/cloud.ply` / MA ply | Same XY; leave `u` |
| `facades.obj` + `planes.json` | Transform verts / plane `(n,d)` consistently |
| `scene.json` cameras | Re-export from transformed poses |
| LocalFrame / `origin` / Ortho meta | **Unchanged** (absolute sat already in that ENU) |
| BAG buildings | Omit from hero, or transform identically if debug-on |

Do **not** re-run MapAnything / Path α for XY seat alone — transform product artefacts. Re-run align→reconstruct only when relative graph itself changes.

`lerp_pose` / known-pose densify stay valid because relative cam–cam geometry is preserved under one SE(2).

---

## 3) Acceptance (smoke-dense + Studio)

**Automated**

- [ ] `--align-prior sat` (default): `bag_snapped` never set; no call to `snap_camera_to_bag`.  
- [ ] `use_satellite=True` (or global SE(2) path) runs; `scene.georef.prior == "sat"`.  
- [ ] Cam arrows on `align/overlay.jpg` sit on Ortho roads to **≲ ~1–2 m** median; max outlier gated.  
- [ ] Mean `|Δxy|` cam vs nearest Ortho road/roof edge ≤ **~1 m** (or NCC `sat_score` gate + visual).  
- [ ] After `T_sat`, cloud centroid XY vs cam hull ≤ prior relative gap (no new tens-of-metres drift).  
- [ ] Façades still load; hybrid **10/8/0.418** quality-keep unchanged (geometry moved rigidly only).  
- [ ] BAG default hidden (PR #27); if shown, same frame or clearly debug.

**Visual Studio**

- [ ] Ortho streets/rooflines under SV arrows ~metre (PR #27 plane + this register).  
- [ ] MA cloud hugs sat street; façades on building footprints as seen in Ortho.  
- [ ] Enabling BAG debug: either coincident or off — never the authority.

**Gate phrase:** *SV cams + façades + MA cloud on sat street/rooflines ~metre; BAG off or same frame.*

---

## 4) First PR scope — `--align-prior sat` (default)

Smallest shippable; no clone required for review — wire existing modules.

### CLI / config

```text
ps1hood align <run> --align-prior sat|bag     # default: sat
ps1hood run   <run> --align-prior sat|bag     # thread into stage_align
```

- Persist on `ProjectSpec` / run meta: `align_prior: "sat"|"bag"`.  
- Settings already have `max_align_shift_m=8`, `max_align_heading_deg=15` — reuse for sat search.

### Code changes (file-level)

1. **`pipeline.stage_align`**  
   - Branch on `align_prior`:  
     - **`sat` (default):** `use_satellite=True`, `use_features=True`; **skip** `snap_camera_to_bag`; skip or soften `push_out_of_footprints`; emit sat residuals/`sat_score`.  
     - **`bag`:** preserve today’s behaviour (sat off when buildings, bag snap on).  
   - After sat path: compute summary SE(2) `T_sat` (centroid + mean yaw from pre/post or from explicit global solve).  
   - Write `align/georef.json` + pass into reconstruct.

2. **`pose_graph.refine_poses`**  
   - Stop deriving `use_satellite` from `not buildings` at call site.  
   - Optional: feed sat hits as stronger priors into `_bundle_se2` (weight sat obs) so relative graph absorbs absolute seat.

3. **`cli.py`**  
   - `--align-prior` on `align` and `run`.

4. **`reconstruct/export.scene_payload`**  
   - Add:
     ```json
     "georef": {
       "prior": "sat",
       "frame": "ENU+LocalFrame",
       "T_sat": {"tx_m": ..., "ty_m": ..., "yaw_deg": ..., "s": 1.0},
       "sat_score_mean": ...,
       "overlay_registered": true
     }
     ```
   - Keep PR #27 `satellite.enu`.

5. **Tests**  
   - Unit: sat prior never calls bag snap (mock); bag prior still can.  
   - `scene_payload` includes `georef`.  
   - Optional: smoke fixture — after sat align, overlay centroid within N metres of Ortho road mask (if fixture available).

### Explicitly out of first PR

- Full roof-edge extractor from Ortho pixels.  
- NAP / BAG Z rewrite.  
- Re-planarize / ZNCC Path α.  
- Per-cam free Sim(3) / vertical seat.  
- Making BAG hero.

### Suggested PR title

`Align prior sat (default): disable BAG snap, sat refine + georef in scene.json`

---

## 5) Follow-ups (after PR1)

1. **True global SE(2) solver** — joint Ortho NCC / road edges over all cams (replace mean-of-per-cam).  
2. **Apply `T_sat` to existing product PLY/OBJ** without full re-recon (one-shot `scripts/apply_georef.py`).  
3. **Ortho road/roof edge DT** (BAG-edge twin on sat).  
4. **Metre CI gate** on smoke-dense overlay.  
5. Optional BAG→sat residual register for debug overlay only.

---

## 6) Report card (Chief)

| Item | |
|------|--|
| **Pack path** | `/workspace/sat-absolute-xy-register-rd.md` |
| **Top bug** | `stage_align` sets `use_satellite=not buildings` and snaps cams to BAG whenever 3DBAG exists — **inverts** sat-absolute XY |
| **Top fix** | Default `--align-prior sat`: disable `snap_camera_to_bag`, always run Ortho NCC / SE(2) seat, write `georef.T_sat` into `scene.json`; keep photo relative via one transform (+ feature bundle) |
| **Depends** | PR #27 (sat plane Ortho ENU) — done |
| **Sacred** | Sat ortho = absolute XY; photo relative 3D preserved; BAG not hero |

