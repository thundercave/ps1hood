# Soft Z cloud clean vs sat roof shells

**Date:** 2026-09-20 · Façades **18/18** held · far-side garage = coverage ceiling (Mapillary side note — **not** this PR)  
**Sacred:** sat absolute XY · no free-pose · **opt-in** · **not** hungry XY clip (user rejected Ortho±2 m as default)

---

## Problem

MA / recon cloud still shows **spikes above roofs** and smear at depth edges. Prior `--cloud-clip-sat` XY±2 m deleted real yards (−42%) without fixing Z floaters. Need a **vertical** gate tied to sat roof shells already in product (`ps1hood roofs` → `recon/roofs.json`).

---

## Soft Z gate

Reuse `assign_shell_z` / roof AABB footprints (`kind=roof`):

```text
for each point p in cloud:
  find roof region R with p_xy inside AABB (inset 0.5 m)
  if none: keep (yards/street untouched)
  if p.z > R.shell_z + margin_m:   # default margin_m = 1.5
    drop
  if p.z < ground_z − 1.0: optional sink drop (off by default)
```

`shell_z` = existing roof high-percentile / façade-top assignment — never invent Z from BAG.

**Write:** `recon/cloud_zclean.ply` (+ `.bak` of prior zclean) + `align/georef.json` → `cloud_zclean = {kept, dropped, margin_m, …}`.  
**Default product** stays unclipped `cloud.ply` (no hungry XY clip).  
Studio: hero = sat shells + façades; toggle **cloud** (raw) vs **zclean**.

---

## CLI (opt-in)

```bash
# densify unchanged — soft Z is a separate opt-in step
uv run ps1hood densify smoke-dense --backend mapanything ...

uv run ps1hood cloud-zclean smoke-dense --margin-m 1.5

# or after sat seat:
uv run ps1hood align smoke-dense --align-prior sat --cloud-zclean --zclean-margin-m 1.5
```

**Do not** re-enable `--cloud-clip-sat` as default. XY clip stays opt-in and separate.

---

## What not to do

| Avoid | Why |
|---|---|
| Hungry XY Ortho±2 m default | Killed yards; user restored unclipped |
| Clip façades.obj by Z | Walls OK; only cloud |
| Free-pose / re-recon for floaters | Sacred poses |
| Block on garage Mapillary | Roofs / floaters ship first |

---

## Mapillary note (roof_099_e0 — don’t block)

Google probes = no SV. Optional later: `discover --source mapillary` small bbox on need probes; if images face garage, sat-align + photo façades. **Not required for this PR.**

---

## Acceptance

- Sky spikes above sat roof shells gone (or ≪ prior) at margin 1.5 m  
- Yard / street points outside roof AABBs unchanged  
- Façades 18/18 untouched  
- Stats in georef; reversible via bak / flag off  
