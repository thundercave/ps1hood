# PS1 Hood — Additive research (paths, recipes, licenses)

**Companion to:** `/workspace/ps1hood-compare-and-pathforward.md`  
**Date:** 2026-09-06 · **Do not overwrite the main compare doc.**  
**Scope:** concrete post–MapAnything (~592k locked-pose) recipes, tool licenses, ROCm vs CUDA, ranked milestones with risks.

---

## A) Meshing recipes after a dense locked-pose cloud

### A1 — Planar-first (best PS1 fit) ★

Buildings are mostly vertical planes. Prefer **cluster → hull → quad** over organic Poisson for the hero walls.

```text
PLY (MapAnything, ENU)
  → voxel_down_sample(0.05–0.15 m)
  → estimate_normals (consistent orientation toward cameras)
  → loop: segment_plane (distance≈0.05–0.12 m) while residual size ≥ N
       keep only near-vertical normals (|n·up| < ~0.15)
  → remaining: ground plane (near-horizontal) + organic leftover
  → per plane: convex hull / alpha shape → rectangular façade quads
  → ZNCC / Milestone-A photo-score before promoting to Studio
```

- Open3D plane / patch APIs: https://www.open3d.org/docs/latest/tutorial/geometry/pointcloud.html  
- Open3D surface recon (BPA / Poisson / alpha): https://www.open3d.org/docs/release/tutorial/geometry/surface_reconstruction.html  
- Geospatial Poisson-vs-BPA intuition (watertight bubbles vs honest holes): https://www.3d-geospatial.com/point-cloud-mesh-processing-pipelines/surface-reconstruction-algorithms/

**Why planar-first:** affine UV + few hundred quads *is* the PS1 read. Organic million-tri meshes fight the aesthetic and the warp tools you already have.

### A2 — Poisson vs BPA vs planar (when to use which)

| Algo | Watertight | Hallucinates? | Use on PS1 Hood |
|------|------------|---------------|-----------------|
| **Planar segment + hull** | No | Low if ZNCC-gated | **Façades / ground (default)** |
| **BPA** | No — honest holes | Low | Clean, uniform leftovers; tune radii from NN distance |
| **Screened Poisson** | Yes | Yes (bubbles in empty space) | Terrain envelopes / closed props; **trim by density** |
| **Alpha shapes** | No | Parameter-sensitive | Sharp façade breaks / footprints |

Open3D calls:
- `TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8..11)` → trim low `densities`
- `TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii=[r,2r,4r])`
- Always need oriented normals first.

### A3 — PS1 retopo / decimation

| Tool | Role | License | Link |
|------|------|---------|------|
| **Instant Meshes** | Field-aligned quad remesh from mesh *or* point cloud; target face count | BSD-style | https://github.com/wjakob/instant-meshes |
| **PyNanoInstantMeshes** | Scriptable Instant Meshes | (wraps above) | https://github.com/vork/PyNanoInstantMeshes |
| **Open3D `simplify_quadric_decimation`** | Fast target-face count; no nice edge flow | MIT | Open3D mesh docs |
| **MeshLab Quadric Edge Collapse** | GUI bake + texture-aware simplify | GPL | https://www.meshlab.net/ |

**Recipe:** planar façades stay as quads (don’t Instant-Mesh them). Instant Meshes / quadric only on organic leftovers (cars→omit, bushes→billboard later). After remesh, **UVs are lost** — re-bake textures (next section).

---

## B) Texturing with known poses

### B1 — Façade ortho / projective (primary, stack-native)

Reuse Milestone A tooling:
1. Best 1–3 SV crops facing the plane (cross-pano, 2–25 m).
2. Rectilinear crop from equirect *before* homography (never warp raw pano).
3. `getPerspectiveTransform` + `warpPerspective` → ortho PNG.
4. Fail-loud on grazing / out-of-frame / low ZNCC.
5. PS1 bake: nearest resize ≤128–256, RGB555 ± Bayer.

Links:
- OpenCV homography: https://docs.opencv.org/5.0/tutorials/features/homography/homography.html  
- Equirec2Perspec (MIT): https://github.com/fuenwang/Equirec2Perspec  
- Texture2LoD3 (pano→LoD walls): https://wenzhaotang.github.io/Texture2LoD3/ · https://arxiv.org/abs/2504.05249 · https://github.com/WenzhaoTang/Texture2LoD3  
- OpenFACADES: https://github.com/seshing/OpenFACADES  
- Prior pack: `/workspace/photo-consistent-facades.md`, `/workspace/ps1hood-research-pack.md`

### B2 — Multi-view mesh atlas (after classical mesh exists)

| Tool | License | Notes |
|------|---------|-------|
| **OpenMVS `TextureMesh`** | AGPL-3.0 | `TextureMesh scene.mvs -m mesh.ply`; seam leveling; hobby OK, note if shipping | https://github.com/cdcseacave/openMVS |
| **MVS-Texturing** | BSD-ish (check repo) | Posed images + mesh → atlas; lighter license than OpenMVS | https://github.com/nmoehrle/mvs-texturing |

OpenMVS known-pose texturing pitfalls: mesh must share ENU with cameras; Interface.h / `--mesh-file` wiring — https://github.com/cdcseacave/openMVS/issues/1035 · https://github.com/cdcseacave/openMVS/wiki/Interface

**MapAnything shortcut:** vertex colors / depth-colored GLB for Studio *preview* only — not final PS1 atlas.

### B3 — Façade photo-consistency on denser clouds

Do **not** re-run RANSAC on the whole MapAnything cloud as “buildings.” Instead:
1. Seed vertical plane hypotheses from dense inliers *or* Manhattan headings × distance grid.
2. Multi-view ZNCC under locked `P=K[R|t]` (Milestone A code).
3. Accept only planes that beat ZNCC threshold (A was ≈0.48 — treat as floor, retune on denser support).
4. Reject street-dominated / car-occluded patches via semantic mask (below).

---

## C) Tracks, densify backup, MapAnything→mesh

### C1 — Why OpenMVS failed (and when to retry)

Sparse mean views/point ≈ **2.0** → `SelectNeighborViews` starves → “not enough images in view.”  
Fix tracks **before** densify: sequential / custom pair list (≥3 neighbors, baseline ≥4–6 m), then MASt3R matcher. Target mean ≳ **2.5–3**.  
Docs: `/workspace/openmvs-denser-sparse.md`, `/workspace/longer-tracks-no-oom.md`  
Issues: https://github.com/cdcseacave/openMVS/issues/917 · #560 · #888

### C2 — MASt3R as matcher only (Path β)

```text
custom cross-pano pairs.txt
  → MASt3R pairwise matches (NOT free-pose global recon)
  → import to COLMAP DB
  → point_triangulator with FIXED ENU poses
  → optional OpenMVS Densify (--view-neighbors-file pairs.txt)
  → ReconstructMesh / TextureMesh
```

- MASt3R: https://github.com/naver/mast3r · arXiv https://arxiv.org/abs/2406.09756  
- MASt3R-SfM (retrieval + sparse global align — use only if you abandon locked poses, which you should **not** for hero): https://europe.naverlabs.com/research/publications/mast3r-sfm-a-fully-integrated-solution-for-unconstrained-structure-from-motion/  
- Ghost rule: never merge free-pose MASt3R/DUSt3R clouds into ENU — `/workspace/correct-geometry-ghost-duplicates.md`

### C3 — MapAnything → mesh recipes

Official stack: https://map-anything.github.io/ · https://github.com/facebookresearch/map-anything  
Apache weights: https://huggingface.co/facebook/map-anything-apache  

Already working (PR #9 + TheRock torch 2.14 / tv 0.29 on 6900 XT):
- Locked `camera_poses` cam2world + `ignore_pose_inputs=False`  
- Recipe: `/workspace/mapanything-enu-recipe.md` · ROCm gotchas: `/workspace/mapanything-rocm-gfx1030.md`

**Mesh export options:**
1. **Preferred:** PLY → planar segment → quads (A1) + ortho bake.  
2. MapAnything / demo GLB-OBJ if available — treat as preview, then retopo.  
3. Feed MapAnything PLY + posed images into OpenMVS Interface → `ReconstructMesh` (needs visibility / view IDs — issue #1035) — secondary.

---

## D) Known-pose 3DGS (Path γ)

**Idea:** init Gaussians from MapAnything cloud; **freeze cameras** to ENU; train appearance; later extract mesh / render reference views for bake.

| Piece | Link / note |
|-------|-------------|
| Original 3DGS | https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/ |
| gsplat (Apache-2.0) | https://github.com/nerfstudio-project/gsplat — **CUDA-first** |
| nerfstudio Splatfacto | https://docs.nerf.studio/nerfology/methods/splat.html — init from PLY via transforms.json / COLMAP |
| Init from non-COLMAP PLY | https://github.com/nerfstudio-project/nerfstudio/issues/2681 |

**Hardware:** Prefer **cloud CUDA**. Official AMD gsplat ROCm ports target Instinct (MI300), not consumer 6900 XT — don’t block the milestone on ROCm GS.  
**Risk:** mesh extract from GS is noisy; still need Instant Meshes / planar bake for PS1. Use γ as **appearance oracle** or hole-fill reference, not final game mesh.  
**Anti-ghost:** never enable pose refinement that invents new extrinsics (JOGS / COLMAP-free papers are for *unposed* cases — opposite of your strength).

---

## E) Semantic cleanup, holes, trees / thin objects

### E1 — Semantic masks before densify / texture

Mask **sky / road / cars / people**; keep **building / wall / window** for façade scoring and texture.

| Tool | Notes | Link |
|------|-------|------|
| Mask2Former (HF) | Panoptic/semantic; ADE20K/COCO weights | https://huggingface.co/docs/transformers/main/model_doc/mask2former |
| SegFormer | Lighter façade/opening detection | used in Texture2LoD3 comparisons |
| Semantic-SAM + CLIP filter | Texture2LoD3 façade isolation | https://filipbiljecki.com/publications/2025_cvprw_texture2lod3.pdf |

Project masks into ENU (vote across views) before deleting points — single-view masks flicker.

### E2 — Hole filling

| Cause | Fix |
|-------|-----|
| Sparse grazing façades | More cross-pano views; MASt3R matches; don’t Poisson-fill walls |
| Poisson bubbles | Trim by vertex density; or don’t Poisson façades |
| BPA holes | Honest — fill with planar patch if ZNCC OK, else leave / billboard |
| Car occlusion | Semantic mask + pick alternate view for texture |

### E3 — Trees / poles / signs (PS1 reality check)

Dense recon will shred thin structures. **Do not** Instant-Mesh trees into organic blobs for the dream walkable hood.

| Object | PS1 treatment |
|--------|----------------|
| Trees / hedges | Cardboard **billboards** or simple cones + leaf sprites; place from MapAnything cluster centroids |
| Poles / lamps | Vertical cylinders / 2-tri crosses; one instance per ENU cluster (anti-ghost voxel) |
| Signs | Textured quads facing road |
| Cars | Omit or low-poly props — not hero geometry |

Path δ in the main doc = planes for buildings + primitives/billboards for thin stuff.

---

## F) Ranked paths — milestones, risks, hardware, licenses

### Path α — MapAnything → planar PS1 mesh ★ PRIMARY

| # | Milestone (2–4) | HW | Risk |
|---|-----------------|----|------|
| α1 | Studio-compare checklist vs A-planes / flow (Chief) | Local | Subjective; wait for Chief notes |
| α2 | Voxel + vertical `segment_plane` → quads on smoke block | CPU / Open3D | Over-segment windows as separate planes |
| α3 | Ortho / projective bake + ZNCC gate + RGB555 | CPU OpenCV | Grazing SV; car paint on walls |
| α4 | Studio import + snap/affine walkable smoke | Browser | UV seams; scale mismatch |

**Licenses:** Open3D MIT · OpenCV Apache-2 · MapAnything Apache weights preferred (CC-BY-NC research weights exist — stick to `map-anything-apache`).  
**Success:** walls align with windows; one lamp; walkable ground; no OSM/BAG.

### Path β — MASt3R matcher → longer tracks → OpenMVS

| # | Milestone | HW | Risk |
|---|-----------|----|------|
| β1 | Custom pairs + MASt3R matches → COLMAP DB | CUDA preferred; ROCm experimental | VRAM; matcher≠recon |
| β2 | `point_triangulator` fixed poses; mean views/pt ≳2.5 | CPU OK | Still ~2.0 if pairs too local |
| β3 | OpenMVS densify + TextureMesh | CPU densify; CUDA PatchMatch optional | AGPL; neighbor starve if β2 weak |

**Licenses:** MASt3R (check NAVER academic terms) · COLMAP BSD · OpenMVS **AGPL-3.0**.  
**Role:** complements α (better classical texture/densify); not a replacement for MapAnything cloud.

### Path γ — Known-pose 3DGS → remesh

| # | Milestone | HW | Risk |
|---|-----------|----|------|
| γ1 | Export ENU cameras + MapAnything PLY → nerfstudio/gsplat | **Cloud CUDA** | ROCm GS not on 6900 |
| γ2 | Train with **frozen** poses; novel-view check | CUDA | Overfit / floaters |
| γ3 | Mesh extract or bake reference views → Instant Meshes / planar | CPU | Mesh from GS noisy |

**Licenses:** gsplat Apache-2 · nerfstudio Apache-2 · original INRIA 3DGS research license (read before ship).

### Path δ — Hybrid primitives

| # | Milestone | HW | Risk |
|---|-----------|----|------|
| δ1 | Cluster non-planar MapAnything leftovers (trees/poles) | CPU | Mis-classifying façade relief as “tree” |
| δ2 | Billboard / cylinder / sign quad library in Studio | Browser | Art direction time |
| δ3 | Semantic mask pipeline to strip cars/sky from α textures | GPU optional | Mask flicker |

---

## G) Hardware cheat-sheet (2026-09-06)

| Workload | RX 6900 XT ROCm | Cloud CUDA | CPU |
|----------|-----------------|------------|-----|
| MapAnything locked-pose densify | **Works** (PR #9) | Fallback | No |
| Open3D planar / BPA / Poisson / quadric | — | — | **Yes** |
| Milestone A ZNCC / ortho warp | — | — | **Yes** |
| OpenMVS densify / TextureMesh | — | optional PatchMatch | **Yes** |
| MASt3R matcher | Maybe / fragile | **Preferred** | Slow |
| 3DGS / gsplat train | Don’t bet | **Preferred** | No |
| Instant Meshes | — | — | **Yes** |
| Mask2Former | ROCm torch maybe | Easy | Slow |

ROCm MapAnything notes: bf16→fp16 fallback, disable flash-attn / use SDPA, `minibatch_size=1`, stride≥2 — `/workspace/mapanything-rocm-gfx1030.md`.

---

## H) License quick-ref

| Asset | License | Ship note |
|-------|---------|-----------|
| MapAnything code | Apache-2.0 (repo) | Prefer `facebook/map-anything-apache` weights |
| MapAnything NC weights | CC-BY-NC | Research only |
| Open3D | MIT | OK |
| Instant Meshes | BSD-style | OK |
| OpenMVS | AGPL-3.0 | Hobby OK; viral if you distribute binaries linked to it |
| MVS-Texturing | permissive (verify SPDX on GitHub) | Prefer over OpenMVS if shipping atlas tool |
| COLMAP | BSD | OK |
| gsplat / nerfstudio | Apache-2.0 | OK |
| MASt3R | Research / check NAVER | Matcher-only use; don’t redistribute weights blindly |
| Street View imagery | Google ToS | Personal/hobby recon ≠ redistribute raw panos |

---

## I) Suggested Studio-compare checklist (for Chief)

When Studio notes arrive, score **MapAnything vs Milestone-A planes vs old flow cloud** on:

1. Façade alignment (windows/doors vs SV)  
2. Ghost duplicates (lamps/signs count)  
3. Ground continuity / walkability  
4. Trees / poles (usable clusters vs spaghetti)  
5. Holes / missing walls  
6. Scale / ENU drift vs `align/`  
7. Texture readiness (planar area %, grazing views)

Append results into the main compare doc §1 “Better” — keep this `-extra.md` for recipes only.

---

## J) One-block “dream real” sequence (ops)

```
1. Freeze MapAnything PLY + poses (ignore_pose_inputs=False verified)
2. α2–α3 planar + bake on smoke bbox only
3. Studio walk + PS1 shader (snap/affine/RGB555)
4. δ billboards for trees/signs from leftover clusters
5. Parallel β: MASt3R matcher → track histogram
6. γ only if α stalls on appearance holes
```

**Non-goals (reaffirmed):** OSM/3DBAG hero mesh · free-pose per-clip recon · mapper retries · flow-RANSAC buildings · Poisson as façade hero.
