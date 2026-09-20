# Soft support gate for off-tile / sky-halo floaters

**Date:** 2026-09-20 · Façades **untouched** · **opt-in** · no free-pose · **not** Ortho XY±2 m  
**Companion:** [`roof-floater-zclean.md`](roof-floater-zclean.md) (above-shell spikes)

---

## Why zclean missed them

Z-clean only acts **inside** roof footprints (`z > shell_z+margin`). Off-tile MA smear sits in empty ENU beside the block → never enters a roof AABB → drop count 0. Need a **support** gate, not another roof-Z pass.

---

## Soft support gate

Build a **keep mask** from sat layers in `recon/roofs.json` (roof ∪ yard ∪ street), plus optional cam corridor ±6 m:

```text
support = dilate(roof ∪ yard ∪ street, radius_m ≈ 10)   # ENU AABB expand
# + cam pose pads ± cam_corridor_m (default 6)

for each cloud point p:
  if p_xy in support:          KEEP   # real scene incl. yards
  elif p.z > local_ground_z + z_out_m:   # default z_out_m=8
    DROP   # sky smear off-tile
  elif dist(p_xy, support) > far_m:         # default far_m=15
    DROP   # empty halo far from any footprint
  else:
    KEEP   # soft fringe (trees leaning out, etc.)
```

`local_ground_z`: `cam_u_median − 2.5`, else median yard shell z, else cloud p10 — **not** BAG.

**Not this:** `clip to Ortho.enu_corners ± 2 m` (hungry; kills yards at tile rim).

| Mode | Yard safe? | Kills off-tile halo? |
|---|---|---|
| Ortho±2 m XY | **No** (prior −42%) | Yes |
| Roof-Z only | Yes | **No** (PC dropped=0) |
| **Support dilate + far/z_out** | **Yes** | **Yes** |

---

## CLI (opt-in)

```bash
# densify unchanged — offtile is a separate opt-in step (no densify)
uv run ps1hood cloud-offtile smoke-dense \
  --support-dilate-m 10 \
  --far-m 15 \
  --z-out-m 8

# or extend zclean:
uv run ps1hood cloud-zclean smoke-dense --margin-m 1.5 --offtile \
  --support-dilate-m 10 --far-m 15 --z-out-m 8
```

Write `recon/cloud_offtile.ply` (+ bak of prior offtile) + `align/georef.json` →
`cloud_offtile = {kept, dropped_far, dropped_z, dilate_m, far_m, z_out_m, …}`.  
Studio toggle **cloud** / **zclean** / **offtile**. **Never** rewrite `facades.obj`.

---

## Acceptance

- Visible off-tile / sky-halo pts gone or ≪ prior  
- Yard + street points inside dilated support kept  
- Façades 18/18 unchanged  
- Reversible (flag off / bak)  

## Non-goals

Hungry default XY clip · free-pose · garage Mapillary · sat_edge peels

---

## One-liner

*Floaters are off-roof, not above-shell — opt-in drop pts far from dilated roof∪yard∪street (or z≫local ground); keep yards; never Ortho±2 m default.*
