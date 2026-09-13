# PS1 Hood — Compare + path forward (R&D)

**Dream:** walkable PS1 neighborhood from Street View multi-view — photo-derived geometry, not cadastral boxes.  
**Date:** 2026-09-06 (provisional compare; Studio notes from Chief TBD)

---

## 1) Compare — what we did right / wrong / better

### Right
| Move | Why it mattered |
|------|-----------------|
| **ENU poses sacred + `lerp_pose`** | Killed the “5 panos → 5 streetlights” failure mode |
| **Photo-consistency façades (Milestone A)** | Scored walls in *image* space (ZNCC≈0.48) instead of trusting street-biased flow RANSAC |
| **MapAnything with locked poses** | ~592k dense pts on ROCm 6900 — first cloud that can actually look like the photos |
| **Refuse OSM/BAG as hero mesh** | Stayed on the dream path |
| **ROCm torch/tv matrix discipline** | 2.14↔0.29 unblocked PC densify |

### Wrong
| Move | Cost |
|------|------|
| Flow cloud → RANSAC as buildings | Big wrong walls, uncorrelated with SV |
| Classic COLMAP `mapper` on SV orbits | “No good initial pair” / pure rotation |
| OpenMVS densify on ~2-view tracks | 0 dense pts (“not enough images in view”) |
| Treating FILM parallax as geometry | Phenomenal detail, ghost duplicates |
| Guided matching exhaustive | OOM; didn’t lengthen tracks |

### Better (next bar)
1. **Studio-compare** MapAnything vs A-planes vs old flow — façades, ghosts, ground, trees, holes.  
2. Turn 592k pts into **PS1-readable mesh + textures** (planar clusters / Instant Meshes + projective bake).  
3. Grow **multi-view tracks** (MASt3R matcher) so classical densify/texture also work.  
4. Validate **no duplicate landmarks** quantitatively (voxel occupancy / reproject).  
5. Thin structures (signs, poles, trees) — densify alone won’t PS1-ize them; need primitives or billboards.

---

## 2) Path to dream (ranked)

### Path α — MapAnything cloud → PS1 mesh ★ (primary)
1. Studio soak + write compare checklist  
2. Voxel downsample → iterative vertical `segment_plane` + ground → quads/hulls  
3. Projective / ortho texture bake from SV (reuse A warp) + RGB555  
4. Instant Meshes / Open3D quadric on leftover organic bits  
5. Studio: vertex snap + affine UV  

### Path β — MASt3R matcher → longer tracks → OpenMVS/TextureMesh
Unblocks classical densify/texture; complements α.

### Path γ — Known-pose 3DGS (cloud or ROCm if viable) → mesh extract → Instant Meshes  
Highest fidelity appearance; heavier; then force low-poly.

### Path δ — Hybrid primitives  
Planes for buildings + MapAnything points for street furniture → PS1 billboards for trees/signs.

---

## 3) Near-term milestones
1. **Compare doc** from Studio (Chief) + Scrapy research ← *now*  
2. **Mesh+texture MVP** from MapAnything PLY on smoke block  
3. **MASt3R matcher** for track length  
4. **PS1 polish** (palette, snap, affine) in Studio  

## Explicit non-goals
OSM/BAG hero · per-clip free-pose recon · more mapper retries · trusting flow-RANSAC buildings

---

*Research links / tool detail: filled by follow-up scrape into this file or `-extra.md`.*

---

## 4) Research toolkit (post–MapAnything densify)

### Mesh / PS1 retopo
| Tool | Role | Link |
|------|------|------|
| Open3D `segment_plane` / `detect_planar_patches` | Building walls from dense cloud | https://www.open3d.org/docs/latest/tutorial/geometry/pointcloud.html |
| Open3D BPA / Poisson | Organic leftover surfaces (trees caution) | same |
| Instant Meshes | Quad remesh from mesh or PC | https://github.com/wjakob/instant-meshes |
| PyNanoInstantMeshes | Python remesh API | https://github.com/vork/PyNanoInstantMeshes |
| Open3D `simplify_quadric_decimation` | Target face count for Studio | Open3D mesh docs |

### Texture
| Approach | Notes |
|----------|-------|
| Milestone A ortho warp on planar patches | Best for façades; reuse ZNCC accept |
| OpenMVS `TextureMesh` | After classical mesh; AGPL |
| MVS-Texturing | BSD; posed images + mesh |
| Vertex colors from MapAnything | Fast preview; not PS1 final |

### Tracks / densify backup
| Tool | Role |
|------|------|
| MASt3R matcher → posed COLMAP | Longer tracks; then OpenMVS |
| MapAnything (done) | Primary dense cloud |
| Known-pose 3DGS | Appearance spike; then remesh |

### PS1 look (already researched)
RGB555 bake + Three.js vertex snap + affine UV — prior packs.

---

## 5) Recommended sequence (make the dream real)

```
Studio-compare MapAnything
    → α: planarize buildings + projective texture + RGB555
    → Studio walkable smoke block
    → β: MASt3R matcher (optional quality)
    → γ: 3DGS only if planar path stalls on organic detail
    → δ: billboards for trees/signs
```

**Success criteria for “dream real” on one block**
- Walls line up with windows in SV  
- One lamp per real lamp (no ghosts)  
- Walkable ground in Studio  
- PS1 read at 320–640 upscale (snap/affine/palette)  
- No OSM/BAG required for the hero mesh  



---

## 6) Live update — MASt3R matcher (2026-09-06 evening)

- Result: **5773** posed pts, **mean track 2.00**, only **0.1% ≥3-view**
- MapAnything ~591k + flow 99k unchanged
- Conclusion: matcher increased *two-view* density, did **not** unlock OpenMVS neighbor graph
- Do **not** wait on β for dream path — proceed Path α (planarize MapAnything)
- β fix research: ensure match graph has 3-cycles; import must merge tracks across pairs sharing a keypoint/image (not isolated pair tracks); multi-forward partners per frame


---

## 7) STUDIO REVISION (2026-09-06) — CRITICAL

**KEEP FLOW ~99415 as product. Do NOT promote MapAnything until ENU frame fix.**

| Cloud | Verdict |
|-------|---------|
| Flow 99k | ENU-aligned, on-street, holey; hugs cameras + photo façades (7 planes / 5 textured / ZNCC 0.42) |
| MapAnything 591k | WRONG FRAME — sky smear / floating sheet; z p50~60m; ΔC~(27,-3,47)m; predicted-pose drift ~53–60m |

**Path α revised:** (1) fix MA ENU fuse/export → (2) planarize+texture. Checklist: `/workspace/mapanything-enu-frame-fix.md`. Hide 3DBAG cardboard shells from hero Studio view.


---

## 8) PR #14 MERGED (2026-09-13) — ENU frame fix landed

**Title:** Fix MapAnything ENU frame: pts3d_cam + locked cam2world  
**Merged:** 2026-09-13 ~17:26 CEST

| Item | Status |
|------|--------|
| Root cause | Export dumped raw MA `pts3d` (model world) as ENU |
| Fix | `WORLD_ENU = R_c2w @ X_cam + C_enu` with **our** poses; prefer `pts3d_cam` / `depth_z`; refuse raw `pts3d` |
| Gates | Fail-loud if centroid ≫25 m from cams or z p50 diverges from camera u |
| Tests | 157 pytest + 19 mapanything unit (incl. refuse-raw, cam2world vs w2c) |

**Still KEEP FLOW as product** until live PC re-run passes Studio gates (on-street, centroid/z near cameras). Then Path α planarize+texture.


---

## 9) GATE PASSED (2026-09-13) — MA ENU Product → Path α planarize

**Studio/PC:** MapAnything ENU GATE PASSED — **579613** pts, `|ΔC|centroid 7.78 m`, z p50 **2.71** vs cam u **2.98** (Street-level Studio).

| Item | Status |
|------|--------|
| MA | **Studio PRODUCT** (flow ~99k = backup only) |
| Next | **Path α:** planarize MA ENU PLY → PS1 quads + texture |
| Recipe | **`docs/path-alpha-planarize-recipe.md`** (implementation-ready) |
| Wire | Extend `photo_planes.py` + `facades.py` — MA PLY → Open3D vertical `segment_plane` → existing ZNCC/ortho bake. No OSM/BAG. No Poisson hero. |

First PR: smoke-block planarize only (no Instant Meshes / organic remesh). See recipe §8 acceptance tests.
