# R&D: OSS frame interp + video→point cloud + toolchains (ps1hood)

**Hard rule:** one ENU world frame; densifiers must not invent extrinsics (ghost lamps).  
**Hardware:** AMD 6900 XT (ROCm experimental) + cloud CUDA when needed.

---

## A) Frame interpolators (large baseline / SV stills)

| Rank | Tool | License | Large motion | Speed / VRAM | AMD notes | SV fit |
|------|------|---------|--------------|--------------|-----------|--------|
| **1** | **FILM** (Google) | Apache-2.0 | ★★★★★ designed for it | Heavier TF; patch mode for hi-res | CUDA/TF; CPU slow | **Best OSS default for pano gaps** — already in README lore |
| **2** | **GIMM-VFI** (NeurIPS’24) | check repo | ★★★★ SNU-FILM-arb Hard SOTA-ish | ~8–11 GB @ 2K/4K w/ DS_SCALE | PyTorch CUDA first | Strong FILM challenger for hard motion |
| **3** | **Practical-RIFE** v4.25+ | MIT | ★★★ (blends on huge motion) | Fast, practical | ncnn-vulkan exists | Good realtime / fallback; not best for 8–20 m SV gaps |
| 4 | AMT | MIT-ish | ★★★ | Lightweight | PyTorch | Fine for small gaps |
| 5 | SPEED (MM’26) | check | ★★★★ diffusion one-step | Newer | CUDA | Watch — early for production |
| — | DIS optical flow (current fallback) | Apache (OpenCV) | ★★ | CPU | Native | Ship path today; not AI-quality |

**Links:**  
- FILM https://github.com/google-research/frame-interpolation (archived but usable)  
- GIMM-VFI https://github.com/GSeanCDAT/GIMM-VFI  
- Practical-RIFE https://github.com/hzwer/Practical-RIFE  
- AMT https://github.com/MCG-NKU/AMT  
- SPEED https://github.com/bbldCVer/SPEED  

**R&D takeaway:** Keep FILM as quality king for SV; A/B **GIMM-VFI** on smoke legs; RIFE for speed. Interp = appearance only + `lerp_pose`.

---

## B) Video / multi-view → point cloud (OSS)

| Rank | Tool | Known poses? | License | Compute | Role for us |
|------|------|--------------|---------|---------|-------------|
| **1** | **MapAnything** (Meta’25) | **Yes** — cams+K+poses in | Apache-2.0 | CUDA | **Top feed-forward densify with ENU lock** https://github.com/facebookresearch/map-anything |
| **2** | **OpenMVS** Densify | Yes (via COLMAP/Interface) | AGPL-3.0 | CPU OK | Milestone B classic densify |
| **3** | **COLMAP** `point_triangulator` | Yes | BSD | CPU | Posed sparse / tracks |
| **4** | **MASt3R** | Weak / unofficial | CC BY-NC-SA | CUDA | **Matcher** → posed COLMAP, not free-pose hero |
| **5** | **VGGT** | Predicts poses; depth can unproject w/ yours | VGGT license | CUDA | Fast cloud densify spike |
| 6 | Fast3R / Speed3R / CUT3R / Spann3R | Mostly free-pose multi-view | varies | CUDA | Long sequences; lock/align to ENU after if used |
| 7 | **gsplat / 3DGS** | Yes (fixed cams) | Apache | CUDA; ROCm gsplat ≠ 6900 | Cloud proof “no ghosts” |
| 8 | InstantSplat-class | Often pose-free | varies | CUDA | Risk of ghosting — low priority |

**MapAnything** is the headline find: feed-forward metric 3D with optional **camera_poses + intrinsics** (OpenCV cam2world). Ideal cloud CUDA experiment after posed sparse grows.

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

## Top 3 experiments when Chief asks
1. **GIMM-VFI vs FILM** on one hard SV leg (visual + then posed densify quality).  
2. **MapAnything** smoke with locked ENU poses (cloud CUDA).  
3. **MASt3R-as-matcher** on multi-forward pair list → longer tracks → retry OpenMVS.

## Hardware cheat sheet
| Piece | 6900 XT | Cloud CUDA |
|-------|---------|------------|
| DIS / OpenCV / COLMAP CPU / OpenMVS CPU | Yes | — |
| FILM / GIMM / RIFE / MapAnything / MASt3R / VGGT / 3DGS | Risky/slow ROCm | Preferred |

---

**Related in-repo:** [correct-geometry-ghost-duplicates.md](correct-geometry-ghost-duplicates.md) (ENU midframes), [openmvs-densify.md](openmvs-densify.md) / [openmvs-denser-sparse.md](openmvs-denser-sparse.md) (Milestone B), [gpu-mast3r-ubuntu.md](gpu-mast3r-ubuntu.md) (CUDA/ROCm).
