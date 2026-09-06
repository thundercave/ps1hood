# Photo-consistent façades (Milestone A)

**Implemented in-repo:** `ps1_hood.reconstruct.photo_planes` + `extract_facades`.

- Hypotheses: sparse posed points (optional) + Manhattan / heading×distance along the drive — **not** flow-cloud RANSAC as product geometry.
- Score: plane-induced homography `H = K_s (R_rel + t_rel n_refᵀ / c) K_r⁻¹` (n·X+d=0 → n_ref·x=c), ZNCC on ortho patches across **cross-pano** views (`zncc_accept` default 0.35).
- Fail-loud: `facades.obj` with ground only + error log if nothing passes; Studio must not show invented RANSAC blocks.
- Ortho bake: reuse frontal-camera warp into `recon/textures/facade_XX.jpg`.

Research digest follows (adapted from `/workspace/photo-consistent-facades.md`).

---

**Dream:** walkable PS1 hood from SV multi-view — walls that match the photos.  
**Not the hero:** OSM / 3DBAG extruded shells.  
**Now:** posed COLMAP ~200–1k pts; flow cloud ~99k (street OK, façades weak); RANSAC-on-cloud façades **geometrically uncorrelated** with imagery (expected: flow fails on near-fronto-parallel walls + cars/sky; RANSAC invents planes in noisy street points).

---

## Recommended next experiment (do this first)

**Replace cloud-RANSAC façade geometry with photo-consistency plane search using known poses.**

For each candidate vertical plane (seed from: sparse triangulated wall points, Manhattan VP directions, or a coarse grid of headings × distances along the road):

1. Project plane into 2–4 **cross-pano** SV crops (known `P = K[R|t]`).
2. Warp patches via homography (plane-induced) and score **NCC / ZNCC** across views.
3. Keep / refine plane only if multi-view score passes; fail-loud otherwise.
4. Bake ortho texture from best view(s) onto that plane (you already have warp tooling).

This directly answers “walls that align with what’s in the images.” Cloud RANSAC cannot.

**Parallel spike (same week):** OpenMVS `DensifyPointCloud` from known poses + current sparse seed (posed COLMAP or filtered flow). CPU densify exists; CUDA PatchMatch optional. AGPL — fine for hobby.

---

## 1) Approaches for photo-consistent building geometry

| Approach | What it does | Fits known poses? | Notes |
|----------|--------------|-------------------|-------|
| **Photo-consistency plane / slanted plane-sweep** | Hypothesize depth or plane; warp neighbors; pick max NCC | **Yes — designed for it** | Classical MVS; best CPU/AMD path for façades |
| **OpenMVS densify** | Depth maps + fuse from posed scene + sparse cloud | **Yes** | https://github.com/cdcseacave/openMVS — CPU densify; CUDA PatchMatch optional |
| **COLMAP PatchMatch** | Dense stereo | Yes after posed sparse | **CUDA only** — skip on AMD box |
| **PyTorch plane-sweep** (e.g. OpenReco-style) | Portable WTA depth on torch device | Yes | CUDA / MPS / **ROCm** / CPU — https://github.com/abhibagul/OpenReco |
| **OpenMVS / MVS-Texturing mesh** | Mesh + atlas from posed images | Yes | After densify |
| **DUSt3R / MASt3R** | Pairwise pointmaps + global align | **Weak / unofficial** for fixed poses | Issues: mast3r#6,#26; dust3r#30,#143 — `preset_pose` incomplete; scale↔focal entangled |
| **MASt3R matches → COLMAP** | Use MASt3R as matcher, keep your poses | Poses yours | `demo_glomap` / kapture toys in mast3r repo |
| **VGGT** | Feed-forward depth/points/poses | Predicts own poses | https://github.com/facebookresearch/vggt — can unproject depth with **your** extrinsics experimentally; needs GPU |
| **3DGS / NeRF → mesh** | Optimize radiance/splats then extract mesh | Yes (posed images) | Great consistency; heavy; mesh extract noisy; then simplify for PS1 |

**Why flow+RANSAC fails façades:** DIS tracks texture poorly on repetitive brick/windows and grazing walls; street/ground dominates inliers; RANSAC fits the densest wrong structure. Photo-consistency on **vertical** hypotheses ignores the street cloud.

### Links (geometry)
1. OpenMVS — https://github.com/cdcseacave/openMVS · wiki Usage  
2. COLMAP known poses FAQ — https://colmap.github.io/faq.html#reconstruct-sparse-dense-model-from-known-camera-poses  
3. Plane-sweep / slanted NCC — Calvet et al. multi-view normals + plane-sweep (Springer)  
4. Facade-warped correlation — Devernay CVPR’00 http://devernay.free.fr/publis/cvpr00.pdf  
5. MASt3R — https://github.com/naver/mast3r · arXiv 2406.09756 (known-pose support incomplete)  
6. DUSt3R — https://github.com/naver/dust3r · arXiv 2312.14132  
7. VGGT — https://github.com/facebookresearch/vggt · CVPR 2025  
8. Portable plane-sweep precedent — OpenReco `mvs_planesweep.py`  

---

## 2) Walls that align with imagery (checklist)

1. **Score geometry in image space**, not only in 3D point space.  
2. Use **cross-pano** baselines (2–25 m) — same as `select_stereo_pairs`.  
3. Constrain **vertical** planes (Manhattan); estimate yaw from vanishing points or from pose headings facing façades.  
4. **Validate** by reprojecting texture and measuring NCC / edge alignment; reject low scores (fail-loud).  
5. Optional: building/façade semantic mask (SegFormer / Mask2Former) to ignore sky/road/cars before MVS.  
6. Do **not** promote RANSAC planes from the flow cloud to “buildings.”

---

## 3) Path to PS1-poly after denser recon

1. Dense / plane mesh → **keep large planar patches** (already PS1-native).  
2. Else: Instant Meshes / Open3D `simplify_quadric_decimation` → few hundred–few thousand tris per block.  
3. Bake textures (OpenMVS TextureMesh or your ortho warp) → resize nearest → **RGB555**.  
4. Studio: vertex snap + affine UV (prior research pack).  
5. Prefer **textured quads** over organic Poisson meshes for the PS1 read.

Links: Instant Meshes https://github.com/wjakob/instant-meshes · Open3D mesh simplify docs · prior `/workspace/cpu-known-pose-mvs.md` / PS1 Three.js notes.

---

## 4) AMD RX 6900 XT vs CUDA / cloud

| Stack | 6900 XT (gfx1030) | Notes |
|-------|-------------------|--------|
| OpenCV / NumPy plane search, flow | **Yes (CPU)** | Milestone 1 |
| OpenMVS densify (CPU) | **Yes** | AGPL |
| PyTorch plane-sweep | **Maybe** | Needs ROCm PyTorch on RDNA2 — experimental on consumer cards |
| COLMAP PatchMatch | **No** | CUDA |
| Official MASt3R / DUSt3R / VGGT | **Expect CUDA** | Try cloud GPU or ROCm torch build; not the safe first milestone |
| AMD `gsplat` ROCm port | **Not for 6900** | Official target = Instinct MI300; don’t bet the milestone on it |
| 3DGS / nerfstudio | Prefer **cloud CUDA** | Then mesh→simplify locally |

**PC Grok role:** run OpenMVS CPU densify + photo-consistency plane search locally; reserve cloud CUDA for MASt3R/VGGT/3DGS spikes.

---

## 5) Ranked next 1–2 milestones (this repo)

### Milestone A (1–2 weeks) — **Photo-consistent façades** ★
- Implement vertical plane photo-consistency search + fail-loud + ortho bake.  
- Kill Studio path that treats flow-RANSAC planes as building truth.  
- Success: walls line up with windows/doors in SV when textured; wrong walls rejected.

### Milestone B (parallel / next) — **Posed densify**
- Export known poses + sparse seed → OpenMVS DensifyPointCloud (CPU) → mesh/texture.  
- Or minimal PyTorch plane-sweep depth → fuse → same plane fit.  
- Success: denser façade points that still photo-validate.

### Later spikes (not blocking A)
- Cloud CUDA: MASt3R as **matcher** into posed COLMAP / or VGGT depth unprojected with **your** poses.  
- 3DGS on cloud → mesh → Instant Meshes → PS1 bake (quality spike, heavier ops).

### Explicit non-goals
- OSM/BAG as hero mesh  
- More mapper retries  
- Trusting flow-cloud RANSAC for building outlines  

---

## One-line diagnosis for the user
Nothing’s impossible — the failure mode is **using a street-biased flow cloud as façade geometry**. Switch the source of walls to **multi-view photo-consistency under known poses** (then densify). PS1 look is a simplify+palette step after that.
