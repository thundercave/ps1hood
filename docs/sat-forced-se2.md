# Forced SE(2) from façade → Ortho Canny (`sat-offset`)

**Date:** 2026-09-20 (Europe/Amsterdam)  
**Pack:** `/workspace/sat-forced-se2-from-offset-rd.md`  
**Sacred:** one rigid SE(2) · no free-pose · bak first · prefer sat XY · skip sat-native roofs/street

Edge re-seat can no-op when fused photo NCC sits in a local max. This tool measures a **different cost** — top-down façade long edges vs Ortho Canny Chamfer — fits one SE(2), then bak+applies it to cams + cloud + façades/planes together.

## CLI (PC)

```bash
# 1) Measure red façade edges → yellow sat Canny; write align/T_force.json
uv run ps1hood sat-offset measure smoke-dense --out align/T_force.json
# prints tx ty yaw rms n_pairs

# Optional Chamfer multi-start polish (escape NCC basin when measure is thin):
uv run ps1hood sat-offset measure smoke-dense --multistart

# 2) Bak then apply one T (skips roofs/street by default)
uv run ps1hood sat-offset apply smoke-dense --from align/T_force.json \
  --targets cams,cloud,facades,planes \
  --skip roofs,street
```

Undo: restore `align.bak_force_<ts>/` over `align/` and `recon/bak_force_<ts>/` files into `recon/`.

Equivalent low-level (after writing `georef.T_applied`):

```bash
uv run python scripts/apply_georef.py smoke-dense --which T_applied
# seat_recon_artefacts today skips roofs ✓
```

## Internals

| Step | Symbol |
|------|--------|
| Long façade edges from `planes.json` quads | `facade_long_edges` |
| Ortho Canny ENU samples + DT | `sat_canny_edge_points_enu` / `sat_canny_dt_metres` (`ortho_canny`) |
| Pair midpoints within search R (default 8 m) | `match_facade_to_sat` |
| Fit SE(2) | `align.georef.fit_se2` |
| Gates | rms ≤ 1.5 m · n_pairs ≥ 4 · \|yaw\| ≤ 15° · \|\|t\|\| ≤ 10 m |
| Bak | `bak_force` → `align.bak_force_<ts>/` + `recon/bak_force_<ts>/` |
| Apply | `apply_forced_se2` → `apply_se2_pose` + ply/obj/planes helpers (same as `seat_recon_artefacts`) |
| Persist | `align/T_force.json`, `georef.T_force` + `georef.T_applied` |

Optional multi-start (`multistart_facade_chamfer`, CLI `--multistart`): grid Δe,Δn ∈ ±6 m, Δyaw ∈ ±8° minimizing mean Ortho-Canny DT on façade edge samples — so future auto-align can leave the NCC no-op basin.

## Non-goals

Per-cam free-pose · wipe 18 · hungry cloud clip · blaming sat tile (PC: tile OK) · Studio pick (manual backup, later)

## Acceptance

- After apply: Studio yellow≈red within ~1 m on long walls  
- Relative photo structure unchanged (one SE(2))  
- roofs stay put if already on sat; façades/cams/cloud moved together  
- bak restores prior  
