# Studio hero — default-hide raw MA cloud

**Date:** 2026-09-20 · **STOP** further cloud-gate peels (offtile killed high-z; long arms remain in soft fringe)  
**Sacred:** **don’t delete** product `cloud.ply` / zclean / offtile · façades + sat roofs stay · sat XY · no free-pose

---

## Goal

Product Studio view = **complete-looking scene** without MA halo arms:

| Layer | Default |
|---|---|
| Façades (`facades.obj`) | **ON** |
| Sat roofs/yards (`roofs.obj`) | **ON** |
| BAG | **OFF** (already) |
| Raw MA / recon `cloud.ply` | **OFF** |
| `cloud_zclean.ply` | OFF (debug) |
| `cloud_offtile.ply` | OFF (debug toggle) |
| SV cams | ON (debug ok) |

User turns **cloud (raw MA)** on when diagnosing densify — not for the hero product look.

## Wire

`viewer.html`: `#togCloud` unchecked; after `loadAsync(cloud.ply)` set `cloudPoints.visible = false`. Façades + roofs stay checked; BAG / zclean / offtile stay off. No server delete; API paths unchanged; `cloud.ply` remains on disk for Path α / densify.

## Acceptance

- Cold open Studio smoke-dense: façades + sat roofs, **no** point-cloud arms
- Toggle cloud → raw PLY appears; untoggle → gone
- Files on disk unchanged

## One-liner

*Studio default-hide raw MA cloud; hero = façades + sat roofs/yards; cloud stays on disk for a debug toggle — stop peeling the fringe.*
