# PS1 façade simplify — PR1 pack (tight)

**Date:** 2026-09-13 · **Product:** hybrid façades **10/8/0.418** (dual-source Path α)  
**Goal:** kill wavy walls/windows **without** re-recon / more peels / FILM / sat roofs.  
**Sacred:** photo-derived geometry · sat XY lock · no BAG hero.

---

## Out of scope (this PR)
Sat roofs · denser panos/FILM · more Path α peels · Instant Meshes · Poisson

---

## PR1 recipe (post-process on existing `planes.json` + textures)

### 1) Manhattan / axis-aligned rect snap (geometry)
Keep each accepted plane’s **center** + **n** from photo ZNCC. Rebuild the quad as a rectangle in ENU:

```text
n_xy = normalize(n[:2]);  n = (n_xy[0], n_xy[1], 0)
right = (−n[1], n[0], 0)          # horizontal tangent
up    = (0, 0, 1)
# optional: snap n_xy to nearest 90° of dominant street heading (median cam travel ±90°)
width  = clamp(hyp.width_m,  3…20);  height = clamp(hyp.height_m, 2.5…15)
# or from current corners: project to (right,up), take percentile 5–95, then
# force equal opposite edges (true rectangle)
corners = center ± (w/2)·right ± (h/2)·up   # BL,BR,TR,TL order
```

**Wire:** new `reconstruct/ps1_facades.py::manhattan_rectify_planes(planes) → planes`  
Call from `extract_facades` **after** accept / promote, **before** `_warp_facade_texture` / `_write_obj`.  
CLI: `ps1hood facades … --ps1-rectify` (default **on** for Studio product) or post: `ps1hood ps1-facades <run>`.

### 2) Window UV / grid regularize (texture space)
After warp (or on existing `textures/facade_XX.jpg`):

1. Warp with **axis-aligned** dst quad (already rectangular in ortho bake).  
2. Optional: detect vertical/horizontal edges (Sobel) → snap dominant periods → light affine to make mullions axis-aligned.  
3. **Minimum viable:** skip detector; just **nearest-neighbor** resize to fixed texel size so windows read as blocks (PS1).

```text
tex_w, tex_h = 128×128 or 128×256 (cap); INTER_NEAREST
```

### 3) PS1 palette + resolution cap
```text
RGB555: c15 = (r>>3<<3, g>>3<<3, b>>3<<3)   # or full 15-bit pack
optional 1-step Bayer / ordered dither
write JPEG quality ≤85 or PNG; Studio already nearest-filters if we set magFilter
```

**Wire:** `_warp_facade_texture` end → `ps1_quantize(img)` · ground tex optional same.

### 4) Studio (tiny)
Viewer: façade materials `magFilter=NearestFilter`, `minFilter=NearestFilter` (PS1 look). Vertex snap already ≈ Manhattan quads.

---

## Acceptance (smoke-dense)
- Walls look **planar rectangles** in Studio top + street view (no bow-tie / wavy sill)  
- Windows read as **blocky grid**, not warped curves  
- Still **10** planes (or same count); mean ZNCC gate **not** re-run required  
- No BAG; sat XY unchanged; no densify re-run  

---

## First PR files
| File | Change |
|------|--------|
| `reconstruct/ps1_facades.py` | `manhattan_rectify_planes`, `ps1_quantize`, `nearest_resize` |
| `reconstruct/facades.py` | hook after accept; flag `--ps1-rectify` / `--ps1-tex-size 128` |
| `studio/static/viewer.html` | nearest filtering on façade maps |
| `tests/test_ps1_facades.py` | rect opposite edges equal; RGB555 bits |

**Do not** change align / MapAnything / peel params in this PR.

---

## One-liner for implementers
*Rectify accepted façade quads to ENU Manhattan rectangles about photo centers → nearest 128² warp → RGB555 → Studio nearest filter.*
