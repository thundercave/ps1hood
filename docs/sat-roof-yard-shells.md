# Sat-locked roof / yard shells (PR-A)

**Date:** 2026-09-20 · **Product:** MA ENU cloud + hybrid façades 10/8 · sat absolute XY · BAG off  
**Goal:** fill SV-blind roofs / backyards with flat shells — completeness before PS1 polish.  
**Sacred:** sat = absolute XY (Ortho ENU) · photo multi-view = relative 3D · **no BAG hero**.

Pack: `/workspace/complete-scene-no-holes-rd.md` (PR-A only).

---

## What it does

1. Load `runs/<name>/satellite/ortho.jpg` + `ortho.json` into `Ortho` (ENU).
2. Segment roof / yard footprints: **Canny barriers + flood from image border** (street/open), classify green interiors as yards, remainder as roofs.
3. Flat **AABB shell quads** in Ortho ENU XY.
4. **Z (roofs)** = high percentile of MA z in footprint (if clearly above ground / cam slab), else façade top from `planes.json`, else `ground_z + 8 m`. **Never** street/cam height. **Yards** = low ground band (`ground_z + ~0.15`), thin near ground.
5. **Footprints** = eroded mask + AABB inset (~0.75 m). Reject shells that overlap sat street/asphalt or camera XY corridor (± margin). Skip roofs whose Z still intersects the street-view slab at cam height.
6. Texture from **sat crop** of each footprint (nearest in Studio).
7. Write `recon/roofs.obj` + `roofs.mtl` + `textures/roof_*.jpg` + `roofs.json`.

Does **not** touch hybrid façades / unclipped MA cloud. Does **not** invent BAG/OSM extrusions. Sat XY lock stays sacred.

### Post-PR #32 fix (`fix/sat-roofs-z-footprint`)

PC after merge saw ≥1 slab too low / over-wide cutting street-level view (`ma_median` of ground points in a spilled footprint). Fix: roof Z = MA **high** percentile or façade top; shrink footprints; reject street/cam overlap; fail-skip street-slab shells.

---

## Usage

```bash
# After sat + (optional) façades / MA cloud exist:
uv run ps1hood roofs smoke-dense

# Or fold into façades:
uv run ps1hood facades smoke-dense --sat-roofs
```

Studio: hard-reload viewer → **roofs** toggle (on by default). Leave **BAG** off.

Gate log: `mean_edge_m` / `gate_target_ok` (target ≤ ~1 m vs Ortho Canny). Counts: `n_roof`, `n_yard`, `textured`.

Fail-loud: missing `ortho.json` / empty sat image / zero footprints after segmentation.

---

## PC after merge

```bash
ps1hood roofs smoke-dense
```

Studio hard-reload; BAG off; expect roofs/yards closing sky holes over buildings and empty backyards/parking pads (street surface stays sat ground plane). **No street-cutting slabs** at cam height — roofs at façade-top / MA high median; yards near ground; footprints inset off the road/camera corridor.

---

## Out of scope (this PR)

FILM densify · Path α peels · PS1 polish · hungry cloud clip · BAG hero · façade gap fill (PR-C)
