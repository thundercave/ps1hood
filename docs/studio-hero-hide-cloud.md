# Studio hero — cloud-first (raw MA ON)

**Date:** 2026-09-20 · User: **cloud-first** (overrides prior default-hide-cloud / PR #48)  
**Sacred:** sat XY · **don’t delete** `planes.json` / façades / roofs / street on disk · no free-pose

---

## Goal

Product Studio cold-open = **dense photo / MapAnything point cloud**. Planar façades, sat roofs, and street shells are optional overlays (debug / sculpt / compare) — they read as non-constructive cardboard next to the cloud.

| Layer | Default |
|---|---|
| Raw MA / recon `cloud.ply` | **ON** |
| Façades (`facades.obj`) | **OFF** (overlay toggle; sculpt auto-enables) |
| Sat roofs/yards (`roofs.obj`) | **OFF** (overlay) |
| Sat street/ground (`street.obj`) | **OFF** (overlay) |
| BAG | **OFF** |
| `cloud_zclean.ply` | OFF (debug) |
| `cloud_offtile.ply` | OFF (debug) |
| SV cams | ON (debug ok) |

## Wire

`viewer.html`: `#togCloud` **checked**; after `loadAsync(cloud.ply)` set `cloudPoints.visible = true`. Façades / roofs / street unchecked + `*.visible = false` after load. BAG / zclean / offtile stay off. Entering **sculpt** auto-checks `#togFacades` and shows `facadeRoot` if currently off (cloud left as user left it).

No server delete; API paths unchanged; `planes.json` / `facades.*` / `roofs.*` / `street.*` remain on disk.

## Acceptance

- Cold open Studio: point cloud visible; no façade/roof/street slabs until toggled
- Toggle façades/roofs/street → overlays appear; files on disk unchanged
- Sculpt on → façades become visible for pick; BAG still default-off

## One-liner

*Studio default: raw MA cloud ON; façades/roofs/street OFF as toggle overlays; product planes stay on disk — stop treating cardboard shells as the hero.*
