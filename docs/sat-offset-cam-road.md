# Sat-offset from cam track → street centerline (`measure-cams`)

**Date:** 2026-09-20 (Europe/Amsterdam)  
**Pack:** `/workspace/sat-offset-cam-road-rd.md`  
**Sacred:** sat absolute XY · **one rigid SE(2)** · no free-pose · bak · skip sat-native roofs/street  
**Reuse:** `apply_forced_se2` / `bak_force` / `fit_se2` (`align/sat_offset.py`)

Façade↔Canny pairs were noisy; **cam-on-road** is the hard constraint. Fit SE(2) so unique pano XY (red track) land on the Ortho `street_mask` medial / centerline (dotted-line analogue).

## CLI (PC)

```bash
# 1) Measure only — writes align/T_cam_road.json (does NOT apply)
uv run ps1hood sat-offset measure-cams smoke-dense \
  --out align/T_cam_road.json \
  --search-r 15
# dumps mean_nn_m, tx, ty, yaw, rms, n_cams, ||t||; optional --overlay PNG

# Alias:
uv run ps1hood sat-offset measure smoke-dense --from cams-road

# 2) After Studio preview OK — bak then apply (skips roofs/street)
uv run ps1hood sat-offset apply smoke-dense --from align/T_cam_road.json \
  --targets cams,cloud,facades,planes \
  --skip roofs,street
```

**Gate / confirm:** measure reports `rms` / `n_cams` / `||t||` and sets `applied=false`. Do **not** auto-apply — separate `apply` step (or Studio confirm).

Undo: restore `align.bak_force_<ts>/` over `align/` and `recon/bak_force_<ts>/` into `recon/`.

## Internals

| Step | Symbol |
|------|--------|
| Unique cam XY | `load_cam_xy_enu` / `unique_cam_xy_from_project` |
| Ortho street mask | `segment_roof_yard_mask` → `street_m` |
| Medial / centerline | `street_mask_centerline` (ximgproc.thinning \| morph skeleton \| DT ridge) → `street_centerline_enu` |
| Correspondences | each cam → nearest centerline pt (`min_nn`…`search_r`, default 0.5…15 m) |
| Fit SE(2) | `fit_se2`; if \|yaw\| ≤ 2° → translation-only (yaw=0) |
| Gates | n_pairs ≥ 4 · rms ≤ 2 m · \|yaw\| ≤ 10° · \|\|t\|\| ≤ 12 m |
| Persist | `align/T_cam_road.json` · `source=cam_street_centerline` |
| Apply | existing `bak_force` → `apply_forced_se2` (cams+cloud+facades+planes; skip roofs,street) |

**Do not** use full Ortho Canny (road paint + roofs). Street-mask medial is the target.

## Acceptance

- Red cam track sits on dotted / street medial (±~1 m)
- Façades/cloud move with cams; roofs/street unchanged
- Bak restores prior
- One rigid SE(2) only — no free-pose

## Non-goals

Façade-Canny auto apply · free-pose · wipe 18 · Mapillary

## One-liner

*Fit SE(2) from cam XY → Ortho street_mask centerline; bak; apply to cams+cloud+facades+planes; skip roofs/street — put the drive back on the sat road.*
