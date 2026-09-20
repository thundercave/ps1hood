# Sat-offset from cam track → street centerline (`measure-cams`)

**Date:** 2026-09-20 (Europe/Amsterdam)  
**Pack:** `/workspace/sat-offset-cam-road-fix-rd.md` (fix for multi-branch NN)  
**Sacred:** sat absolute XY · **one rigid SE(2)** · no free-pose · bak · skip sat-native roofs/street  
**Reuse:** `apply_forced_se2` / `bak_force` / `fit_se2` (`align/sat_offset.py`)

Façade↔Canny pairs were noisy; **cam-on-road** is the hard constraint. Fit SE(2) so unique pano XY (red track) land on the Ortho `street_mask` medial / centerline (dotted-line analogue).

**Failed measure symptom:** `||t||≈1.9 m` with `rms≈4.1 m` ⇒ multi-branch skeleton NN, **not** a near-miss. **Do not force-apply that T.**

## CLI (PC)

```bash
# 1) Measure only — corridor crop + continuity match → align/T_cam_road.json (does NOT apply)
uv run ps1hood sat-offset measure-cams smoke-dense \
  --corridor-m 15 \
  --translation-only \
  --max-mad-m 1.5 \
  --out align/T_cam_road.json \
  --overlay
# dumps mean_nn_m, median_abs_m, mad_m, tx, ty, yaw, rms, n_cams, ||t||; applied=false

# Alias:
uv run ps1hood sat-offset measure smoke-dense --from cams-road

# 2) After Studio preview OK — bak then apply (skips roofs/street)
uv run ps1hood sat-offset apply smoke-dense --from align/T_cam_road.json \
  --targets cams,cloud,facades,planes \
  --skip roofs,street
```

**Gate / confirm:** measure reports `rms` / `mad` / `n_cams` / `||t||` and sets `applied=false`. Do **not** auto-apply — separate `apply` step (or Studio confirm). Reject if `mad > 1.5 m` even when `||t||` is small.

Undo: restore `align.bak_force_<ts>/` over `align/` and `recon/bak_force_<ts>/` into `recon/`.

## Auto fixes (this PR)

| Fix | What |
|-----|------|
| **Corridor crop** | `street_m &= dilate(cam disks, corridor_m≈15)` **before** skeleton/medial |
| **Continuity match** | Order cams along travel; attach to one branch via road-tangent prediction (no NN jump to parallel parking/side) |
| **MAD gate** | Report `median_abs_m` + `mad_m`; reject if `mad > max_mad_m` (default 1.5 m) |
| **Translation-only** | `--translation-only` / `--se2-mode translation` when yaw is noise |

## Studio fallback (if A–C still fail)

```text
POST /api/runs/<name>/sat-offset/cam-road-pairs
  { "pairs": [ {red:{e,n}, yellow:{e,n}}, ... ] }          # ≥4 cam↔road-center picks
  { "polyline": [ {e,n}, ... ] }                            # drawn centerline; cams snap to it
→ align/T_pick_cam_road.json (preview only)
POST …/sat-offset/apply { "confirm": true, "from": "T_pick_cam_road" }
```

## Internals

| Step | Symbol |
|------|--------|
| Unique cam XY | `unique_cam_xy_from_project` / `order_cams_along_path` |
| Ortho street mask | `segment_roof_yard_mask` → `street_m` |
| Corridor crop | `crop_street_mask_to_cam_corridor` (`cam_xy_mask`, R≈12–20 m) |
| Medial / centerline | `street_mask_centerline` → `street_centerline_enu` |
| Correspondences | `match_cams_to_centerline_continuity` (fallback plain NN) |
| Fit SE(2) | `fit_se2`; if \|yaw\| ≤ 2° or `--translation-only` → translation-only |
| Gates | n_pairs ≥ 4 · rms ≤ 2 m · mad ≤ 1.5 m · \|yaw\| ≤ 10° · \|\|t\|\| ≤ 12 m |
| Persist | `align/T_cam_road.json` · `source=cam_street_centerline` · `applied=false` |
| Apply | existing `bak_force` → `apply_forced_se2` (cams+cloud+facades+planes; skip roofs,street) |

**Do not** use full Ortho Canny (road paint + roofs). Street-mask medial is the target.

## Acceptance

- Preview: transformed red track on dotted / street center (±~1 m)
- rms ≤ ~1.5–2 m **and** mad ≤ ~1–1.5 m before apply
- No apply of the current failed 1.9 m / 4.1 m T
- Façades/cloud move with cams; roofs/street unchanged
- Bak restores prior
- One rigid SE(2) only — no free-pose

## Non-goals

Façade-Canny auto apply · free-pose · wipe 18 · Mapillary · force-apply failed measure

## One-liner

*rms ≫ \|\|t\|\| = multi-branch skeleton NN — corridor-crop the mask, continuity-match along the cam path, gate on MAD; measure → T_cam_road.json only; apply separate with bak; do not force-apply the failed 1.9 m T.*
