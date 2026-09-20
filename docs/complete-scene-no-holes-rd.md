# Complete scene — no holes / deformations (tight R&D)

**Date:** 2026-09-20 · **Strategy flip:** completeness before PS1 look.  
**Product now:** MA ENU cloud ~579k + hybrid façades 10/8 (PS1 post optional) · sat absolute XY · BAG off.  
**Sacred:** sat = absolute XY · photo multi-view = relative 3D · **no BAG hero**.

---

## 1) Ranked hole / deformation causes (smoke-dense)

| Rank | Cause | Symptom | Why |
|------|--------|---------|-----|
| **1** | **SV-blind roofs / backyards** | Big topside holes; yards empty | Street View never sees them; façades-only Path α can’t invent roofs |
| **2** | **Façade gaps** | Missing wall strips between planes | 10 planes leave side walls / returns / occluded faces; 2 untextured |
| **3** | **Sparse / holey ground + street furniture** | Thin cloud, gaps in road/sidewalk | Flow 99k holey; MA denser but still SV-limited; no midframe density in hero path |
| **4** | **MA floaters / deformations** | Spikes above roofs, smear at depth edges | Feed-forward depth noise; clip was too hungry @2 m (user kept unclipped) |
| **5** | **Sat tile rim** | Cloud/façades cut or float vs ortho edge | Ortho bbox ≠ full block; SE(2) seat residual ~metre + weak NCC (~0.2) |
| **6** | **PS1 post (secondary)** | Chunky but not the *holes* | Manhattan/128²/RGB555 is look — defer until mesh is complete |

---

## 2) Top fix order (smoke-dense)

### PR-A — Sat-locked roof / yard shells ★
1. From sat ortho (already ENU): extract roof / yard polygons (Canny + flood / simple height proxy from MA z p50 over footprint).  
2. Extrude or flat **shell quads** in sat XY; Z from MA median / façade top.  
3. Texture from **sat crop** (nearest → later PS1).  
4. Gate: shells must sit on sat edges ≤1 m; fail-loud if footprint empty.

*Fills cause #1 without BAG.*

### PR-B — Denser drive appearance (ENU lerp only) ★ shipped glue
1. FILM (or equivalent) midframes between panos with **`lerp_pose`** — appearance only.  
2. Re-run MapAnything / densify with **locked poses** (never free-pose).  
3. Optional soft floater gate (opt-in, not 2 m hungry default).

*Addresses #3–4; sacred one-world-frame.*

**PC recipe (smoke-dense)** — ship from existing interpolate + MA path:

```bash
# 1) Midframes (poses ALWAYS lerp_pose ENU; film falls back to flow if weights absent)
uv run ps1hood interpolate smoke-dense

# 2) Pose-locked bundle = real panos + densify-stride mids (≥4 m clearance)
uv run ps1hood densify smoke-dense --backend mapanything --export-only --stride 2 \
  --prefer-interp-bundle
# → runs/smoke-dense/mapanything/bundle  (ignore_pose_inputs=False)

# 3) Infer on CUDA host (this box has no NVIDIA CUDA):
python scripts/run_mapanything_bundle.py \
  runs/smoke-dense/mapanything/bundle --apache \
  --import-recon runs/smoke-dense
# backs up recon/cloud.ply → cloud.ply.bak; façades/roofs untouched

# Or one-shot when CUDA is available:
uv run ps1hood densify smoke-dense --backend mapanything --apache --stride 2 \
  --prefer-interp-bundle --backup
```

Soft floater clip stays **off** unless you explicitly run  
`ps1hood align/run … --cloud-clip-sat` (not densify default).  
Docs: [`mapanything-densify.md`](mapanything-densify.md) §PR-B.

### PR-C — Façade gap fill (after A/B)
1. Seed planes from MA vertical peels **or** sat building outlines as *hypotheses only* → ZNCC gate (existing Path α).  
2. Do **not** promote un-scored shells.  
3. Cap peels; quality-keep vs 10/8 product.

*Addresses #2 without peel-as-hero spam.*

### PR-D — Sat seat tighten (parallel, cheap)
Edge-fused NCC (pack already shipped) when usage allows — lowers #5 residual.

---

## 3) What NOT to do
- **More Path α peels as hero** to “fill” the block (already exhausted for completeness)  
- **BAG / OSM extruded shells** as product geometry  
- **Free-pose** COLMAP/MASt3R/MapAnything (ghosts)  
- **Further PS1 polish** until roofs/yards/gaps closed  
- **Hungry default cloud clip** that deletes real yards  

---

## Acceptance (smoke-dense “complete”)
- Roofs + backyards present as sat-locked shells (or dense MA) — no sky holes over buildings  
- Street + façade ring continuous enough to walk in Studio  
- Cams + cloud + façades + roofs agree on sat street/rooflines ~metre  
- BAG still off / non-hero  

---

## One-liner
*Fill SV-blind tops with sat-locked roof/yard shells; densify with ENU-lerped midframes + locked poses; only then PS1.*
