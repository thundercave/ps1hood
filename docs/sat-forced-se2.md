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



## Studio yellow↔red corner picks (manual backup)

Auto Chamfer NN pairs can latch kerb/asphalt instead of building outline — low RMS ≠ correct wall↔wall. Prefer Studio picks when the visual slip (~2–4 m) does not match auto `T_force`.

**Do not auto-apply Chamfer `T_force`.** Fit is preview-only; apply only after confirm.

### Studio (recommended)

1. Open `http://127.0.0.1:8765/viewer?run=<run>` · enable **facades** · toggle **sat-offset pick**
2. Top-down: click **yellow** sat corner, then matching **red** façade corner (≥3 pairs)
3. **fit preview** → shows `tx ty yaw rms` + cyan mapped overlay (writes `align/T_pick.json`, does **not** apply)
4. If overlay looks wall↔wall (~1 m), **apply T** → confirm → bak then cams+cloud+facades+planes (skips roofs/street)

### CLI / API

```bash
# pairs.json = [{ "yellow": {"e":…,"n":…}, "red": {"e":…,"n":…} }, ...]
uv run ps1hood sat-offset fit-pairs smoke-dense --pairs pairs.json
# preview only → align/T_pick.json + T_pick_pairs.json

# only after visual confirm:
uv run ps1hood sat-offset apply smoke-dense --from align/T_pick.json   --targets cams,cloud,facades,planes --skip roofs,street
```

```http
POST /api/runs/<name>/sat-offset/pairs   { "pairs": [ { "yellow": {e,n}, "red": {e,n} }, ... ] }
→ T + preview, applied=false

POST /api/runs/<name>/sat-offset/apply   { "confirm": true, "from": "T_pick" }
→ bak + apply (same as CLI apply)
```

Gates for picks: n≥3 · rms≤1.5 m · |yaw|≤15° · ||t||≤10 m.

## Non-goals

Per-cam free-pose · wipe 18 · hungry cloud clip · blaming sat tile (PC: tile OK) · **auto-applying Chamfer `T_force`** (NN pairs can latch road/tree — preview Studio picks first)

## Acceptance

- After apply: Studio yellow≈red within ~1 m on long walls  
- Relative photo structure unchanged (one SE(2))  
- roofs stay put if already on sat; façades/cams/cloud moved together  
- bak restores prior

## Cam track → street centerline (`measure-cams`)

When panos sit **off** the sat road, fit from unique cam XY → Ortho `street_mask` medial (not full Canny):

```bash
uv run ps1hood sat-offset measure-cams smoke-dense --out align/T_cam_road.json --search-r 15
uv run ps1hood sat-offset apply smoke-dense --from align/T_cam_road.json \
  --targets cams,cloud,facades,planes --skip roofs,street
```

Gates: n≥4 · rms≤2 m · |yaw|≤10° · ||t||≤12 m · `source=cam_street_centerline`. Measure does **not** apply.

Full notes: [`sat-offset-cam-road.md`](sat-offset-cam-road.md).

**Studio draw:** when auto measure still fails, toggle **cam-road polyline** in the viewer — draw a yellow dotted sat centerline → fit preview (`T_pick_cam_road`) → confirm apply. No new auto measure.
