# PS1 Hood — Research Pack (known-pose facade texturing)

**Scope:** (1) SV → vertical facade with known poses, (2) known-pose MVS without full SfM, (3) cheap PS1 look on textured quads.  
**Stack bias:** Python/OpenCV + Three.js; CPU-friendly; avoid heavy GPU/COLMAP mapper when possible.  
**Date:** 2026-09-06

---

## Digests

### 1) Known-pose Street View → vertical facade planes

**Core idea:** With ENU position + heading (+ pitch/FoV), treat each facade as a vertical rectangle. Project the 4 wall corners into the image (or into a rectilinear crop of a pano), then `getPerspectiveTransform` + `warpPerspective` to bake an ortho facade texture. Skip SfM for this path.

**Pitfalls (act on these):**
- **Planar assumption fails** when the wall has bay windows, recesses, or the RANSAC plane is wrong → seams, stretched windows. Mitigate: one texture per RANSAC segment; reject if corner reprojection / plane residual is high; don’t warp across corners.
- **`getPerspectiveTransform` needs 4 ordered corners** (same winding). Degenerate if almost collinear in image (grazing view) → skip or use multi-view blend.
- **Radial / pano distortion:** GSV/Mapillary panos are equirectangular. Always crop to a **rectilinear** view (heading/pitch/FoV) *before* homography, or sample the equirect via lon/lat remap. Homography on raw equirect = wrong.
- **Heading/pitch error:** Even a few degrees shifts UV. Prefer wall-center heading from geometry (`atan2` to wall centroid) over EXIF heading alone; add a small pitch from camera height vs wall mid-height.
- **Occluders / cars:** Warp still paints them. Optional later: mask with semantic seg or pick best view by facing angle × distance × sharpness.

**Best links (3–5):**
1. OpenCV homography tutorial — https://docs.opencv.org/5.0/tutorials/features/homography/homography.html  
2. Equirec2Perspec (MIT) — equirect → perspective crop — https://github.com/fuenwang/Equirec2Perspec  
3. Texture2LoD3 (CVPRW 2025) — pano + LoD building planes → ortho facade textures — https://wenzhaotang.github.io/Texture2LoD3/ · paper https://arxiv.org/abs/2504.05249 · code https://github.com/WenzhaoTang/Texture2LoD3  
4. OpenFACADES (MIT) — Mapillary pano → building-facing crops (geometry + heading) — https://github.com/seshing/OpenFACADES  
5. ISPRS 2026 GSV→wall crop recipe (project wall quad → image, crop band) — https://isprs-annals.copernicus.org/articles/XI-2-2026/127/2026/isprs-annals-XI-2-2026-127-2026.pdf  

**GPU?** No (OpenCV CPU). Texture2LoD3/OpenFACADES optional ML parts can be ignored.

---

### 2) Known-pose depth / MVS without full COLMAP mapper

**Goal:** Sparse SV baselines + known poses → a few 3D points or depth cues for smarter plane pairing / fail-loud checks — **not** city-scale NeRF.

**Options (CPU-friendly first):**
| Tool | Role | GPU | License |
|------|------|-----|---------|
| OpenCV `triangulatePoints` / stereo | Two-view points from matched features + known P1,P2 | No | Apache-2 |
| COLMAP `point_triangulator` | Features + matcher + **fixed** poses → sparse cloud (skip `mapper`) | Optional (CPU OK) | BSD |
| OpenMVS `TextureMesh` / densify with posed scene | Texture existing mesh/planes; densify if needed | Optional (TextureMesh CPU default) | AGPL-3.0 |
| Open3D RGBD/TSDF | Only if you invent depth (plane depth / mono depth); pose integrate | Optional | MIT |

**COLMAP known-pose recipe (no mapper):**
1. Write `cameras.txt` + `images.txt` (poses) + empty `points3D.txt`
2. `feature_extractor` → `exhaustive_matcher` (or sequential for drive)
3. `point_triangulator --input_path <posed_model> --output_path <triangulated>`
4. Dense `patch_match_stereo` needs CUDA — **skip for CPU path**; use sparse points only, or OpenCV triangulation.

**OpenMVS:** Fill `Interface.h` / scene with poses + images (+ optional mesh of facade quads); run `TextureMesh -i scene.mvs -m facades.ply`. AGPL — fine for hobby, note if shipping.

**Best links (3–5):**
1. COLMAP FAQ — reconstruct from known poses / `point_triangulator` — https://colmap.github.io/faq.html#reconstruct-sparse-dense-model-from-known-camera-poses  
2. OpenMVS — https://github.com/cdcseacave/openMVS · Interface wiki https://github.com/cdcseacave/openMVS/wiki/Interface · TextureMesh with custom mesh (issues #225, #1035)  
3. OpenCV `cv2.triangulatePoints` — https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html  
4. Open3D RGBD integration (if you have depth) — https://www.open3d.org/docs/latest/tutorial/pipelines/rgbd_integration.html  
5. MVS-Texturing (BSD-3) — posed images + mesh → atlas — https://github.com/nmoehrle/mvs-texturing  

**GPU?** Prefer **no**. COLMAP dense = GPU; triangulation + OpenCV + OpenMVS TextureMesh = CPU.

---

### 3) Cheap PS1 look on textured low-poly walls (Python + Three.js)

**Target look:** Affine UV warp, vertex snap, RGB555 (~15-bit) + optional Bayer dither. Apply on **already textured facade quads** (don’t need fancy materials).

**Three.js (runtime):**
- Vertex snap in clip space: `floor(pos.xy * res) / res` after divide-by-w / before remultiply.
- Affine mapping: pass `uv * w` and `w`; sample with `(uv*w)/w` in fragment (undo perspective-correct interp).
- Palette: quantize to 5 bits/channel (`floor(c*31+0.5)/31`) + 4×4 Bayer dither; render low-res (e.g. 320×240) then nearest upscale.

**Python/OpenCV (bake into textures for Studio):**
- Resize facade tex to ≤128–256, `INTER_NEAREST`
- RGB555: `img & ~7` per channel (or `>>3<<3`)
- Optional ordered dither before quantize
- Export PNG; Studio/Three just samples nearest / no mips

**Best links (3–5):**
1. Roman Liutikov — PS1 in Three.js (affine + snap + 15-bit) — https://romanliutikov.com/blog/ps1-style-graphics-in-threejs  
2. `threejs-psx-shader` — snap + affine + dither passes — https://github.com/lferreira457/threejs-psx-shader  
3. frytaz PS1Material gist — https://gist.github.com/frytaz/d849055d6838b17cb1bf6c58f646f07a  
4. three-retropass — pixelate + palette quantize — https://github.com/mesmotronic/three-retropass  
5. OpenCV color reduce / palette quantize patterns — https://stackoverflow.com/questions/5906693/how-to-reduce-the-number-of-colors-in-an-image-with-opencv  

**GPU?** Browser WebGL only (already required for Studio). Bake path = pure CPU.

---

## Implement FIRST (ordered)

### Python / OpenCV (pipeline)
1. **Facade ortho bake (week-1):** For each vertical plane + best posed SV frame: rectilinear crop (Equirec2Perspec or own lon/lat remap) → project 4 corners → `getPerspectiveTransform` + `warpPerspective` → PNG. Fail-loud if corners out of frame / grazing / residual high.
2. **PS1 bake:** Nearest downscale + RGB555 (± Bayer) on those PNGs before Studio import.
3. **Only if planes need 3D validation:** OpenCV feature match + `triangulatePoints` between 2–3 nearby posed views; use points to score/adjust plane distance — **not** full COLMAP mapper.
4. **Optional later:** COLMAP `point_triangulator` (CPU) for denser sparse cloud; OpenMVS `TextureMesh` if multi-view atlas beats single-view warp (watch AGPL).

### Three.js (Studio)
1. **Shader patch on wall materials:** vertex snap + affine UV (copy from threejs-psx-shader / Liutikov chunks).
2. **Low-res render target + RGB555 dither upscale** (or rely on pre-baked palette textures + `NearestFilter`, no mipmaps).
3. Keep geometry as **low-poly quads** (one quad per facade segment) so affine warp reads “PS1” instead of fighting dense meshes.

### Explicit non-goals for v1
- Full COLMAP mapper / PatchMatch dense (GPU)
- NeRF / Gaussian splats
- Perfect occlusion cleaning

---

## One-line 3DBAG note (NL only)
If the hood is in the Netherlands, 3DBAG LoD1.3/2.2 walls (CC BY 4.0, https://3dbag.nl / https://api.3dbag.nl) can **replace carved planes** as texture targets — same homography path, better footprints/heights. Skip if outside NL or OSM planes already good.
