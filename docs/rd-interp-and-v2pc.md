# R&D: OSS frame interp + video→point cloud + toolchains (ps1hood)

**Hard rule:** one ENU world frame; densifiers must not invent extrinsics (ghost lamps).  
**Hardware:** AMD 6900 XT (ROCm experimental) + cloud CUDA when needed.  
**License note:** Hobby OK on NC licenses; flag before any product ship.

---

## A) Frame interpolators (large baseline / SV stills)

| Rank | Tool | License | Large motion | Speed / VRAM | AMD notes | SV fit |
|------|------|---------|--------------|--------------|-----------|--------|
| **1** | **FILM** (Google) | Apache-2.0 | ★★★★★ designed for it | Heavier TF; patch mode for hi-res | CUDA/TF; CPU slow | **Best OSS default for pano gaps** — already in README lore |
| **1b** | **FILM-PyTorch** (dajes) | Apache-2.0 | Same FILM strengths | ~6–12 GB @ 720–1080; recursive for big gaps | PyTorch → ROCm more realistic than TF | Drop-in vs TF FILM on 6900 |
| **2** | **GIMM-VFI** (NeurIPS’24) | **S-Lab 1.0 (NC)** | ★★★★ SNU-FILM-arb Hard SOTA-ish | ~8–11 GB @ 2K/4K w/ DS_SCALE | PyTorch CUDA first | Strong FILM challenger for hard motion; hobby OK, commercial needs S-Lab |
| **2b** | **EDEN** (CVPR’25) | Apache-2.0 | ★★★★★ diffusion large-motion | Slower / CUDA | Cloud | A/B vs GIMM if FILM ghosts/tears |
| **3** | **Practical-RIFE** v4.25+ | MIT | ★★★ (blends on huge motion) | Fast, practical | ncnn-vulkan exists | Good realtime / fallback; not best for 8–20 m SV gaps |
| 3b | EMA-VFI | Apache-2.0 | ★★★ | Mid | PyTorch | Secondary quality check |
| 4 | AMT | **CC BY-NC 4.0** | ★★★ | Lightweight | PyTorch | Fine for small gaps; not MIT |
| 5 | SPEED (MM’26) | check | ★★★★ diffusion one-step | Newer | CUDA | Watch — early for production |
| — | SoftSplat / XVFI / VFIformer | Academic / research / MIT | SoftSplat classic; XVFI 4K; VFIformer heavy | SoftSplat slow; XVFI mid | CUDA | Legacy; skip unless needed |
| — | DIS optical flow (current fallback) | Apache (OpenCV) | ★★ | CPU | Native | Ship path today; not AI-quality |

**Links:**  
- FILM (TF, archived) https://github.com/google-research/frame-interpolation  
- FILM-PyTorch https://github.com/dajes/frame-interpolation-pytorch  
- GIMM-VFI https://github.com/GSeanCDAT/GIMM-VFI  
- EDEN https://github.com/bbldCVer/EDEN  
- Practical-RIFE https://github.com/hzwer/Practical-RIFE  
- EMA-VFI https://github.com/MCG-NJU/EMA-VFI  
- AMT https://github.com/MCG-NKU/AMT  
- SPEED https://github.com/bbldCVer/SPEED  
- SoftSplat https://github.com/sniklaus/softmax-splatting · XVFI https://github.com/JihyongOh/XVFI · VFIformer https://github.com/dvlab-research/VFIformer  

**SV rule:** metres-apart stills ≠ video. Prefer FILM / FILM-PyTorch / GIMM / EDEN; RIFE only when baselines shrink via more SV samples. Always `lerp_pose` — never trust interp for geometry.

**R&D takeaway:** Keep FILM (prefer PyTorch port) as quality king for SV; A/B **GIMM-VFI** and **EDEN** on smoke legs; RIFE for speed. Interp = appearance only + `lerp_pose`.

---

## B) Video / multi-view → point cloud (OSS)

| Rank | Tool | Known poses? | License | Compute | Role for us |
|------|------|--------------|---------|---------|-------------|
| **1** | **MapAnything** (Meta’25) | **Yes** — cams+K+poses in | Apache-2.0 | CUDA | **Top feed-forward densify with ENU lock** https://github.com/facebookresearch/map-anything |
| **2** | **OpenMVS** Densify | Yes (via COLMAP/Interface) | AGPL-3.0 | CPU OK | Milestone B classic densify |
| **3** | **COLMAP** `point_triangulator` | Yes | BSD | CPU | Posed sparse / tracks |
| **4** | **MASt3R** / DUSt3R | Weak / unofficial; matcher-only | **CC BY-NC-SA 4.0** | CUDA | **Matcher** → posed COLMAP, not free-pose hero (weights/code still NC) |
| **5** | **VGGT** | Predicts poses; **discard cams**, unproject depth w/ yours | VGGT license | CUDA | Fast cloud densify spike |
| 6 | Fast3R / Speed3R / CUT3R / Spann3R | Mostly free-pose multi-view | CUT3R research; Spann3R CC BY-NC-SA | CUDA | Long sequences; lock/align to ENU after if used — low priority |
| 7 | MegaSaM | Pose+depth from video | Apache-2.0 | CUDA | Invents poses → ENU-align or skip |
| 8 | **gsplat / 3DGS** | Yes (fixed cams) | Apache | CUDA; **amd_gsplat = Instinct MI300 only, not 6900** | Cloud proof “no ghosts” |
| 9 | InstantSplat-class | Often pose-free | NVIDIA research / varies | CUDA | Risk of ghosting — low priority |

**MapAnything** is the headline find: feed-forward metric 3D with optional **camera_poses + intrinsics** (OpenCV cam2world). Ideal cloud CUDA experiment after posed sparse grows.

### B+) Pose-lock recipes (anti-ghost)

#### MASt3R = matcher only (TC3 concrete)
Official toys: `kapture_mast3r_mapping.py` / `demo_glomap.py` in https://github.com/naver/mast3r  
- Write matches into COLMAP DB.  
- Export **fixed ENU priors** as COLMAP `images.txt` / cameras.  
- Run **`point_triangulator`** (or pycolmap triangulator) — **do not** run free mapper / GLOMAP as hero.  
- Flags to mind: keep poses (`has_pose`); avoid `--ignore_pose`; skip pose-optimizing BA that drifts world frame (or BA with locked extrinsics if available).  
Community wrappers: https://github.com/jwd222/mast3r-sfm , https://github.com/hjh530/GIM-colmap-recon  

**In-repo:** [`mast3r-matcher.md`](mast3r-matcher.md) — `ps1hood reconstruct --backend mast3r`
(or `--backend colmap_posed --matcher mast3r`). Fail-loud without package/GPU; free-pose GA not CLI.

#### VGGT without inventing poses
https://github.com/facebookresearch/vggt  
- Run aggregator + **depth_head** (and optional point_head).  
- **Discard** `camera_head` extrinsics.  
- Unproject depths with **your ENU cam2world + K** (`unproject_depth_map_to_point_map` pattern).  
- Optional: Sim(3) align VGGT cloud → ENU for sanity, then throw away VGGT poses.  
- Cloud CUDA; ~1B weights; chunk long SV sets.

#### MapAnything (TC2) — locked ENU
https://github.com/facebookresearch/map-anything — feed `camera_poses` + intrinsics; **never** `ignore_pose_inputs`.
In-repo: [`mapanything-densify.md`](mapanything-densify.md) — Path A posed COLMAP demo
(`--apache --stride 2 --save_glb --save_colmap`); Path B
`ps1hood export mapanything-bundle` + `scripts/run_mapanything_bundle.py`.
Prefer `facebook/map-anything-apache`; `minibatch_size=1` if VRAM tight.

#### 3DGS mesh → PS1
- Train with **fixed cams** (gsplat / nerfstudio splatfacto; camera-optimizer **off**).  
- Mesh: **SuGaR** https://github.com/Anttwo/SuGaR or **2D-GS** surface, then **Instant Meshes** https://github.com/wjakob/instant-meshes for low-poly PS1.  
- TSDF-from-posed-depths (Open3D) is lighter if GS is overkill.

---

## C) Toolchains for THIS repo (ranked)

### TC1 — Ship path (now)
`align` → FILM/RIFE + `lerp_pose` → posed sparse (sequential/multi-pair) → OpenMVS densify **or** Milestone A photo-planes → PS1 simplify/RGB555.  
*Status: in flight with Chief.*

### TC2 — Cloud densify with pose lock ★ next R&D
Subsampled SV crops (+ optional FILM mids) + ENU poses → **MapAnything** (images+K+poses) → PLY/depth → plane cluster / Instant Meshes → Studio.  
*Validates “walking demo density” without per-clip ghosts.*

### TC3 — Learned matcher, classical geometry
Cross-pano pair list → **MASt3R matches** → COLMAP DB → `point_triangulator` (fixed) → OpenMVS.  
*Fixes mean track≈2 without guided-match OOM.*

### TC4 — Proof GS
Known-pose **3DGS** on cloud CUDA (smoke block) → mesh extract → Instant Meshes → PS1.  
*If ghosts vanish, diagnosis sealed.*

### Avoid
Dropping `drive.mp4` into Luma/Postshot/pose-free MASt3R and concatenating PLYs.

---

## Top experiments when Chief asks
1. **GIMM-VFI vs FILM** (prefer FILM-PyTorch) on one hard SV leg (visual + then posed densify quality).  
2. **MapAnything** smoke with locked ENU poses (cloud CUDA).  
3. **MASt3R-as-matcher** on multi-forward pair list → longer tracks → retry OpenMVS.  
4. **VGGT depth + locked ENU unproject** vs MapAnything on same 20-view block (ghost count + facade planarity).

## Hardware cheat sheet / reality (6900 XT)

| Piece | 6900 XT | Cloud CUDA |
|-------|---------|------------|
| DIS / OpenCV / COLMAP CPU / OpenMVS CPU | Yes | — |
| FILM-PyTorch / GIMM / RIFE / MapAnything / MASt3R / VGGT / 3DGS / EDEN | Risky/slow ROCm | Preferred |

| Claim | Reality |
|-------|---------|
| AMD `amd_gsplat` / ROCm gsplat docs | Target **Instinct MI300**, not RDNA2 consumer — don’t expect 6900 XT |
| ROCm on 6900 (gfx1030) | Experimental: `HSA_OVERRIDE_GFX_VERSION=10.3.0`; community PyTorch wheels fragile |
| MASt3R-SLAM-ROCm | https://github.com/EmmanuelMess/MASt3R-SLAM-ROCm — SLAM path; still prefer matcher→COLMAP for ENU lock |
| Local default | CPU COLMAP + OpenMVS + DIS/FILM-CPU; **cloud CUDA** for MapAnything / MASt3R / VGGT / 3DGS / EDEN |

---

**Related in-repo:** [correct-geometry-ghost-duplicates.md](correct-geometry-ghost-duplicates.md) (ENU midframes), [mapanything-densify.md](mapanything-densify.md) (TC2 MapAnything), [openmvs-densify.md](openmvs-densify.md) / [openmvs-denser-sparse.md](openmvs-denser-sparse.md) (Milestone B), [gpu-mast3r-ubuntu.md](gpu-mast3r-ubuntu.md) (CUDA/ROCm), [mast3r-matcher.md](mast3r-matcher.md) (TC3 matcher→posed).
