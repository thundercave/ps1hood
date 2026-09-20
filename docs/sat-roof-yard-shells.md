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
4. **Z** = MA `cloud.ply` median inside polygon, else façade top from `planes.json`, else `ground_z + 8 m` (yards → `ground_z + 0.15`).
5. Texture from **sat crop** of each footprint (nearest in Studio).
6. Write `recon/roofs.obj` + `roofs.mtl` + `textures/roof_*.jpg` + `roofs.json`.

Does **not** touch hybrid façades / unclipped MA cloud. Does **not** invent BAG/OSM extrusions.

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

Studio hard-reload; BAG off; expect roofs/yards closing sky holes over buildings and empty backyards/parking pads (street surface stays sat ground plane).

---

## Out of scope (this PR)

FILM densify · Path α peels · PS1 polish · hungry cloud clip · BAG hero · façade gap fill (PR-C)
