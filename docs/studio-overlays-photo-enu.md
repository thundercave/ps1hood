# Studio overlays → photo ENU

**Date:** 2026-09-13 (Europe/Amsterdam)

## Rule

Photo-derived LocalFrame ENU (Street View cameras + `recon/cloud.ply` + façades) is the **source of truth**.

- Satellite and BAG (if shown) **join** that ENU.
- Do **not** invent OSM/BAG as the hero mesh.
- Do **not** re-solve product cameras into BAG/RD for Studio display.

## What Studio does

| Layer | Placement |
|-------|-----------|
| **Satellite ground** | Ortho ENU of `satellite.bbox` corners (`Ortho.enu_corners` → `scene.satellite.enu`). Size `(ee−sw)×(nn−sh)`, centre mid-corners at `u=0`. Remap `(e,u,-n)`. |
| **BAG shells** | ENU verts from capture; **default-hidden** in product viewer (`#togBag`). Live align pane may show them. |
| **Cloud / façades / cams** | Already photo ENU; toggles on by default. |

## Verify

Open Studio `viewer?run=smoke-dense` and **reload** the viewer page. Ortho roads/roofs should sit under SV camera arrows to ~metre. BAG stays off until you check the toggle.

- Sat/BAG viewer fixes: **viewer reload only** (JS computes ENU from `satellite.bbox` + `origin` when `satellite.enu` is absent).
- Optional: re-run reconstruct to persist `satellite.enu` into `scene.json`.

## Soft note

BAG Z often treats NAP / `h_maaiveld` as ellipsoid altitude while SV poses force `u≈2.5`. Enabling BAG for debug may show a vertical residual until a later NAP fix — expected; not required for product read.

## See also

- [`studio-layer-alignment-survey.md`](studio-layer-alignment-survey.md)
- [`studio-overlay-enu-align-rd.md`](studio-overlay-enu-align-rd.md)
- [`compare-and-pathforward.md`](compare-and-pathforward.md) §19
