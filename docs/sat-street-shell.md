# Sat street / ground shell (smoke-dense)

**Date:** 2026-09-20 · Fill **parking / road holes** under the drive  
**Sacred:** sat absolute XY · no free-pose · **don’t clobber** `facades.obj` / `roofs.obj` / planes  
**Reuse:** `segment_roof_yard_mask` → `street_mask` + Ortho crop (same as PR-A roofs)

Pack: `/workspace/sat-street-shell-rd.md`.

---

## What it does

1. Load Ortho → `segment_roof_yard_mask` → `street_mask`.
2. **Punch hard:** `support = street − dilate(roof ∪ yard, ~1.5 m)` so walls don’t get a ground slab.
3. **Optional:** OR cam XY corridor ±4 m (also punched by buildings) so the drive path is covered even if the mask is thin.
4. Contours → tiles `kind=street` (AABB), split long corridors at ≤ ~25 m; reject tiny (&lt;8 m²) and near-full-tile blobs.
5. **Z = ground:** `median(cam_u) − camera_height_m` (≈ cam_u−2.5), else `planes.json` ground_z / yard low-band. Never MA high-z; never BAG.
6. Sat-crop texture per tile → `recon/street.obj` + `street.mtl` + `textures/street_*.jpg` + `street.json`.

Does **not** merge into `roofs.obj` or rewrite façades / planes. Idempotent overwrite of `street.*` only (optional `street.obj.bak`).

---

## Usage

```bash
uv run ps1hood street smoke-dense \
  --min-area-m2 8 \
  --building-dilate-m 1.5 \
  --cam-corridor-m 4
```

Studio: hard-reload → **street** toggle (default **OFF** overlay; cloud-first hero — façades/roofs also default-off).

---

## Acceptance

- Parking / road holes show sat-textured ground under cams
- No street quad under building footprints (visual + AABB ∩ roof empty)
- Façades 18/18 + roofs unchanged
- Z ≈ ground (not floating at cam_u / roof)

## Non-goals

Cloud-gate peels · sat_edge · Mapillary · BAG extrude · free-pose

---

## One-liner

*Ortho street_mask − buildings → flat ENU ground quads at ground_z, sat-textured into `street.obj` — fill road/parking holes without touching façades or roofs.*
