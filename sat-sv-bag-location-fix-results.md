# sat / SV / BAG location fix results

**Date:** 2026-09-27 (Europe/Amsterdam)  
**Branch:** `fix/sv-raw-gps-seat-no-osm-snap` @ `4689bec` (+ log-message tidy)  
**Dataset:** `runs/smoke-dense`  
**Scope:** SV pose seating only. `rd_to_wgs84` / BAG CRS unchanged.

## Spec

1. Disable OSM `snap_to_polylines_enu` in `pose_graph.initial_poses` — seat on raw Google GPS.
2. Keep sat NCC refine, capped at **1.0 m** from `e_gps`/`n_gps`; reject beyond that and keep GPS.
3. Same `LocalFrame` for Ortho (EPSG:4326) and SV poses (no second absolute).
4. Re-run smoke-dense align; report seated ENU vs Google GPS (target **&lt; 1 m**).
5. This file: before/after numbers.

## Code changes

| File | Change |
|------|--------|
| `src/ps1_hood/align/pose_graph.py` | No OSM snap; `MAX_SAT_SHIFT_M` / `MAX_GPS_DRIFT_M` = 1.0; reject bad NCC; post-bundle clamp |
| `src/ps1_hood/align/satellite_align.py` | Search candidates gated vs GPS prior; default `max_shift_m=1.0`; stage spans ≤ prior cap |

## Before (OSM snap + wide sat search)

### Box clone (`align/poses.json` pre-fix)

| Metric | Value |
|--------|-------|
| n cams | 13 |
| mean Δgps→seat | **6.220 m** |
| max Δgps→seat | **9.462 m** |
| snapped | 13/13 True |

### Host PC (Lego dump, HEAD `ca4a8cb`, 10 unique panos)

| Metric | Value |
|--------|-------|
| mean Δgps→proj | **5.485 m** |
| max Δgps→proj | **10.07 m** |
| snapped | all True |
| sat_score mean | ~0.21 |
| T_sat | tx −0.32 m, ty −1.67 m, yaw +10.5° |

## After (raw GPS + 1 m sat cap)

### Box re-align (`uv run ps1hood align smoke-dense --align-prior sat`)

| Metric | Value |
|--------|-------|
| n cams | 13 |
| mean Δgps→seat | **0.422 m** |
| max Δgps→seat | **0.978 m** |
| snapped | **0/13** |
| Target &lt; 1 m | **met** |

Rejects / clamps (log):

- sat NCC rejected (low score): several panos (kept GPS)
- pose clamp after bundle: `1-hH0xS8V_…` 1.65 m → GPS; `2n4pk_dOww3i…` 1.21 m → GPS
- New `T_sat`: tx 0.12 m, ty −0.18 m, yaw 6.73° (sat_score_mean −0.047)

Product SE(2) re-applied to `cloud.ply` / `cloud_photo.ply` / façades / planes.

### Host PC

Pending Lego checkout of `fix/sv-raw-gps-seat-no-osm-snap` + same align on `/home/sander/ps1-hood/runs/smoke-dense`. Expect same order of magnitude once HEAD matches.

## MapAnything cloud vs sat roads

GPS seat is now within 1 m of Google raw GPS in the Ortho LocalFrame, so the dual-absolute that pulled cams into buildings is removed. Full visual “no cars through buildings” still needs a Studio look on the PC after host pull (Google GPS itself can sit a few metres off painted road centerline; that is GPS noise, not OSM snap).

## Not changed

- `rd_to_wgs84` / BAG CRS (EPSG:28992) — confirmed correct
- BAG as pose prior (already off under `--align-prior sat`)

## Next

1. PC: `git fetch && git checkout fix/sv-raw-gps-seat-no-osm-snap && uv run ps1hood align smoke-dense --align-prior sat`
2. Open Studio; confirm cloud/cams vs sat roads
3. Merge PR when host confirms
