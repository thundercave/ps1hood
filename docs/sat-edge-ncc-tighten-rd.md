# Sat edge+NCC tighten — R&D pack (PRIORITY follow-up)

**Date:** 2026-09-13 (Europe/Amsterdam)  
**Block:** PS1 Hood smoke-dense / product  
**Depends:** **PR #28 MERGED** (`feat/align-prior-sat`) — `--align-prior sat` default, `georef.T_sat`, seat cloud/façades, BAG snap gated.  
**Prior pack:** [`sat-absolute-xy-register-rd.md`](sat-absolute-xy-register-rd.md)

**Sacred (locked):** WGS84 Esri ortho = absolute real-world XY. Photo multi-view keeps **relative 3D** (one SE(2)). BAG not hero. This pack tightens *how well* we lock to sat — not who is authority.

**PC after PR #28 (smoke-dense):**
| Metric | Result |
|--------|--------|
| `prior` | `sat` |
| `bag_snapped` | **0** |
| `T_sat` | tx=**−3.06** m, ty=**1.02** m, yaw=**3.89°** |
| `T_applied` | ~**−3 m / 1.2°** (artefact seat) |
| `sat_score_mean` | **0.26** (weak; gate accept is 0.08) |
| Studio | cams + façades on Ortho ~metre; BAG off |
| Residual | weak photometric NCC + **MA floaters outside sat tile** |

**Goal:** raise registration sharpness with **road/roof edges** fused into sat cost (not NCC alone), plus **clip/gate MA pts outside Ortho ENU bbox ± margin**.

---

## 1) Why mean NCC ≈ 0.26 (and why seat still looked ~metre)

`align_camera_to_satellite` maximises photometric NCC between the photo lower half and a **ground-plane warp** of Ortho (`render_satellite_into_camera_fast` → `ncc`). Accept if score > **0.08**; mean on smoke landed **0.26**.

| Cause | Mechanism |
|-------|-----------|
| **Grazing SV geometry** | Cameras look nearly horizontal; only a thin ground band (code uses `y0 = 0.45·H`) sees sat. Most façade pixels never enter the NCC mask → low SNR. |
| **Trees / cars / shadows** | Ortho top-down ≠ SV side-look. Moving objects, crowns, and time-of-day shade dominate gray correlation even when street centreline is right. |
| **Ortho blur / compression** | Esri export ~1–2 px/m; JPEG + remap at 320 px search scale flattens texture → NCC plateau, not a sharp peak. |
| **Ground-only warp** | Synth assumes flat ground at `camera_height_m`. Raised kerbs, pitched roofs, and any non-ground never match → systematic residual. |
| **Appearance ≠ geometry** | NCC rewards similar brightness, not shared edges. A 1–2 m lateral miss can still score ~0.2–0.3 if asphalt looks like asphalt. |

**Why Studio still looked ~metre:** GPS/OSM prior + feature `_bundle_se2` already put cams near streets; PR #28’s SE(2) seat moved the whole set ~3 m / few degrees. Weak NCC was enough to *nudge* but not enough to *pin* kerb/roofline. Tightening needs a **geometric** cost on road/roof edges.

Score scale today: −1 (empty/fail) … ~0.76 (rare peak). Mean 0.26 ≫ accept 0.08 ⇒ almost every cam “wins,” but the objective is soft — classic weak-prior regime.

---

## 2) Edge-based cost: Canny photo ↔ Ortho road/roof; fuse with NCC

Reuse the **BAG edge DT pattern** (`bag_edges.photo_edge_dt` + projected segments) but source edges from **sat**, not 3DBAG.

### 2a) Ortho edge map (absolute)

```text
ortho_gray → GaussianBlur → Canny (tune; start 40/120)
optional: morphology close to link broken kerbs
optional mask: road+roof only (see §3)
→ distanceTransform (same as photo_edge_dt)
```

**Road lines:** prefer Canny on Ortho asphalt/paint; optional OSM centreline rasterised as weak prior (not BAG walls).  
**Roof lines:** Ortho roof ridges / building outlines visible in Esri — extract long high-contrast edges with length + angle filters (skip short tree noise). Do **not** use 3DBAG verts as hero; sat pixels are authority.

### 2b) Photo edge map

Already exists: `photo_edge_dt(photo_bgr)` (Canny 60/160 → DT). For sat fuse, restrict DT sampling to:
- lower half / ground FOV for road edges, and/or
- mid-band for roof ridges projected into cam (optional second term).

### 2c) How to score a SE(2) candidate

Two practical options (PR1 pick A):

| Option | Idea | Pros |
|--------|------|------|
| **A. Synth-edge NCC / Chamfer** | Render Ortho **edge image** into cam with same `render_satellite_into_camera_fast` (or remap edge DT); score Chamfer = mean DT at photo edge pixels (or 1 − normalised Chamfer) | Reuses warp path; no 3D segments |
| **B. Project Ortho polylines** | Vectorise Ortho Canny → ENU polylines → project like `collect_bag_edges` / `_project_segments` → sample `photo_edge_dt` | Closer twin of BAG snap; sharper on long façades |

### 2d) Fuse with photometric NCC

```text
score = w_ncc * ncc(photo, synth_rgb)
      + w_edge * edge_agree(photo_edges, synth_edges_or_projected)
```

Suggested start: `w_ncc=0.35`, `w_edge=0.65` (edges dominate when photo/sat appearance diverge).  
Gate: require `edge_agree` above a floor **or** fused score > τ; stop accepting pure NCC peaks with collapsed edges.

Emit telemetry: `sat_score` (fused), `sat_ncc`, `sat_edge`, keep mean of fused as `sat_score_mean`.

**Code hooks:**
- `align/satellite_align.py` — add `edge_agree`, optional edge render; extend `align_camera_to_satellite` return dict.
- `align/bag_edges.py` — reuse `photo_edge_dt` (import); do **not** call `snap_camera_to_bag` under sat prior.
- New thin helper e.g. `align/sat_edges.py` — Ortho Canny + optional polyline extract + mask.

---

## 3) Global SE(2) vs per-cam; multi-scale; mask to road+roof

### Ranked solve strategy

1. **Preferred — one global SE(2) on fused cost**  
   Keep relative graph (feature `_bundle_se2` / locked MA poses). Collect per-cam fused peaks as observations `(Δe, Δn, Δyaw, weight=sat_edge)`. Solve **one** `(tx, ty, yaw)` that maximises Σ weights · score_i(T · pose_i) — or fit SE(2) from weighted peaks then polish with 1–2 global iterations. Apply same `T` to cams + cloud + façades (PR #28 path). Preserves photo relative 3D.

2. **PR1 shortcut — per-cam fused search + re-bundle**  
   Replace NCC-only loop in `refine_poses` with fused score; keep `max_shift_m=8` / `max_heading_deg=15`; **re-run `_bundle_se2` after** sat obs (already does). Write `T_sat` via existing `summarize_se2` / `fit_se2`. Good enough if global solver slips.

3. **Anti-pattern:** unconstrained per-cam sat with features off, or re-enabling BAG snap.

### Multi-scale

Already coarse→fine metres/heading in `align_camera_to_satellite`. Extend:
- Pyramid Ortho + photo at ~0.5 / 1 / 2 m/px equivalent (or 160 → 320 → full search window).
- Run edge term at coarser scales first (edges survive downsample better than texture).
- Fine stage: ±2 m / ±4° with 0.4 m / 1° steps on fused score.

### Mask to road + roof (critical for 0.26→useful)

Build Ortho mask `M`:
- **Road:** low-sat / gray band + OSM road buffer raster (optional), or simple HSV/gray threshold + morphology.
- **Roof:** brighter planar patches / building-coloured regions; exclude vegetation green and deep shadow if possible.
- Apply `M` inside remap / NCC / edge agree so trees and courtyard clutter do not win the peak.

Photo side: optional sky/car mask later; PR1 can skip if Ortho mask + lower-half is enough.

---

## 4) Floater gate: drop/hide MA pts outside satellite.bbox ENU ± margin

**Chief: explicit in this pack.** PC residual after seat: MA cloud has points **outside the Ortho tile** (sky rays, distant façades, triangulation outliers) that float past the sat plane and distract Studio.

### Authority bbox

```text
Ortho.enu_corners(bbox, frame) → (sw, sh, ee, nn)
# same as Ortho.__init__ / scene.satellite.enu (PR #27)
margin_m = 2.0   # start; CLI --sat-cloud-margin-m
keep if:
  (sw - margin) <= e <= (ee + margin)
  (sh - margin) <= n <= (nn + margin)
# u unrestricted (vertical floaters are a separate Path α concern)
```

Source of truth: `scene.satellite.bbox` / `Ortho.bbox` → ENU via `LocalFrame`, **not** camera-hull AABB.

### Where to clip (product)

| Stage | Action |
|-------|--------|
| **After `seat_recon_artefacts` / `apply_se2_to_ply`** | Filter `recon/cloud.ply` (and `cloud_photo.ply` if present) in-place or write `cloud_satclipped.ply` + point Studio at clipped |
| **`scripts/apply_georef.py`** | Optional `--clip-sat-bbox` after SE(2) |
| **Studio viewer** | GPU/CPU hide: discard pts with ENU outside sat plane extents (defence in depth if PLY not re-exported) |
| **MapAnything export** (follow-up) | Optional early clip so Path α never sees out-of-tile junk |

### Semantics

- **Drop** (preferred for product PLY): rewrite vertex list; update `georef.cloud_clipped = { kept, dropped, margin_m, bbox_enu }`.
- **Hide** (viewer-only): leave full PLY for debug toggle “show floaters”.
- Do **not** clip façades.obj by sat bbox aggressively (walls near tile edge may straddle); optional soft warn if >X% verts outside.
- BAG remains off / not used for this gate.

### Hook sketch

```python
# align/georef.py (or reconstruct/export)
def clip_ply_to_ortho_enu(path, sw, sh, ee, nn, margin_m=2.0) -> dict:
    # read xyz; keep inside expanded AABB; rewrite PLY; return counts
```

Call from `pipeline.stage_align` after seat when `align_prior == "sat"`, and from `apply_georef.py`.

---

## 5) First PR scope + acceptance

### In scope (smallest shippable)

1. **`align/sat_edges.py`** (new) — Ortho Canny + optional road/roof mask; edge remap or Chamfer vs `photo_edge_dt`.  
2. **`align/satellite_align.py`** — fused `score = w_ncc·ncc + w_edge·edge`; return `ncc`, `edge`, `score`; multi-scale fine step.  
3. **`pose_graph.refine_poses`** — use fused score; keep accept floor but prefer edge-aware τ; telemetry on poses.  
4. **Floater gate** — `clip_ply_to_ortho_enu` after sat seat; `georef` records clip stats; CLI `--sat-cloud-margin-m` (default 2).  
5. **Tests** — fused score prefers edge-aligned shift on synthetic stripe; clip drops pts outside ENU ± margin; sat prior still never bag-snaps.  
6. **Docs** — one § in `compare-and-pathforward.md` (pathforward).

### Out of scope (this PR)

- True joint global SE(2) optimiser over all cams (follow-up; PR1 = per-cam fused + existing `fit_se2`).  
- BAG roof edges as authority.  
- Re-run MapAnything / Path α.  
- NAP / Z rewrite.  
- Full semantic segmentation of Ortho (hand thresholds + OSM road buffer enough).

### Suggested PR title

`Sat edge+NCC fuse + clip MA cloud to Ortho ENU bbox`

### Acceptance (smoke-dense + Studio)

**Automated**
- [ ] `sat_score_mean` (fused) **↑** vs 0.26 baseline on same run (target ≥ **0.40** or clear edge-term lift in telemetry).  
- [ ] `georef.prior == "sat"`, `bag_snapped == 0`.  
- [ ] After clip: **0** (or negligible) MA pts outside Ortho ENU ± `margin_m`; `georef.cloud_clipped.dropped` reported.  
- [ ] Relative structure preserved: one SE(2) only; feature bundle still on.  
- [ ] Unit tests green (fuse + clip + no bag snap).

**Visual Studio**
- [ ] Street centreline / kerb and **roofline** under cam arrows ≤ **~1 m** (tighter than “looks about right”).  
- [ ] MA cloud hugs sat street **inside** tile; no halo of floaters past Ortho edges.  
- [ ] Façades still seated; BAG off.

**Gate phrase:** *Edge-fused sat lock; street/roofline ≤1 m; MA clipped to sat bbox; BAG not hero.*

---

## 6) Compare / pathforward (short)

PR #28 flipped authority to sat and seated the product ~metre — **done**. Remaining gap is **objective sharpness** (NCC 0.26) and **out-of-tile MA clutter**, not BAG snap.

| Track | Role after this pack |
|-------|----------------------|
| **Sat absolute XY** | Authority fixed (PR #27 plane + PR #28 prior). This PR = tighten cost + clip. |
| **Path α** | Planarize/ZNCC on **clipped** MA inside sat tile — fewer bogus peels from floaters. |
| **Studio** | Cleaner walkable block: cams on kerb, cloud bounded by Ortho, façades on sat footprints. |
| **BAG** | Debug overlay only; never edge authority. |

Sequence: **edge+NCC fuse + floater clip (this PR)** → optional global SE(2) polish → Path α coverage on cleaned cloud → PS1 mesh/texture.

---

## Report card (Chief)

| Item | |
|------|--|
| **Pack path** | `/workspace/sat-edge-ncc-tighten-rd.md` |
| **Top fix 1** | Fuse **Ortho road/roof Canny (Chamfer/edge-remap)** with photometric NCC in `satellite_align` / `refine_poses` — fix weak mean NCC 0.26 |
| **Top fix 2** | **Clip/gate MA cloud** to `Ortho.enu_corners` (satellite.bbox ENU) ± margin after sat seat — kill floaters outside sat tile |
| **Depends** | PR #28 (sat prior + `T_sat` seat) — merged |
| **Sacred** | Sat ortho = absolute XY; photo relative 3D; BAG not hero |
