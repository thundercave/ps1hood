# Keep interp detail without ghost duplicates (ps1hood)

**User near-win:** FILM between SV panos → phenomenal dense cloud (signs/trees/bumps) but **N panos → N copies** of the same lamp.
**Rule:** one ENU world frame; **metric poses are sacred**; dense AI may not invent its own extrinsics.

You already have the hard parts:
- `align/` → ENU poses on real SV crops
- `interpolate/sequence.lerp_pose` → midframe poses in metres
- `unproject.triangulate_frames` + `select_stereo_pairs` / cross-pano filter
- Milestone A photo-plane ZNCC; Milestone B OpenMVS densify (optional AGPL)

Ghosts happen when a **second** recon (Luma / Postshot / MASt3R free poses / per-clip SfM / monocular depth) rebuilds geometry in another frame and you merge PLYs.

---

## How to keep FILM density without duplicating objects

### Do
1. Build `interp/frames` with FILM/RIFE **and** attach `lerp_pose(a,b,t)` to every midframe (already the intent of the drive).
2. **Assert** every midframe (full FILM rate) has valid `e,n,u,heading` **and** `fov,width,height` (PINHOLE `K` needs size — missing → silent wrong-fx). Pipeline: `assert_interp_frame_poses`.
3. **Skip** FILM/lerp on legs with ‖ΔENU‖ ≪ 2 m (`MIN_LERP_BASELINE_M`) — same-pano heading mates / rotation-only; write the destination keyframe only.
4. Run densify/recon on **keyframes + every Nth midframe** (`select_densify_frames`), not the full FILM rate (near-dupe cams drown MVS). Still assert poses on the full set before subsample.
5. Run **one** densifier on that posed set with fixed extrinsics:
   - OpenMVS DensifyPointCloud from posed sparse (see `docs/openmvs-densify.md`), or
   - OpenCV multi-view under known `P`, or
   - 3DGS/NeRF with **fixed** cameras (cloud CUDA), or
   - Milestone A planes scored in image space.
6. **Fuse in ENU:** voxel merge (~5–15 cm); drop points with high multi-view reproj; never `cat` per-clip clouds.

### Don’t
- Independent recon per pano→pano clip then concatenate.
- Drop `drive.mp4` into Luma/Postshot/MASt3R and let it estimate poses.
- Monocular depth per frame without scaling to ENU (`align` metres).
- Treat FILM motion as ground-truth parallax for free SfM.
- Invent midframe extrinsics; they must come from `lerp_pose`.

### Fusion checklist (anti-ghost)
| Check | Pass |
|-------|------|
| All frames share one LocalFrame / ENU origin | yes |
| Midframe `e,n,u,heading` from `lerp_pose`, not from AI | yes |
| `fov` / `width` / `height` survive lerp (PINHOLE K) | yes |
| Legs with ‖ΔENU‖ < 2 m skip FILM (no fake baseline) | yes |
| Densifier gets keyframes + every Nth midframe (not full FILM rate) | yes |
| Densifier receives cameras.txt / poses.csv / fixed R\|t | yes |
| Same lamp visible in views i and j → one 3D cluster after voxel+reproj | yes |
| Fail-loud if interp pose / size missing before densify/recon | yes |

---

## Ranked next steps (stack-native)
1. **Verify** every `interp` frame dict has ENU + PINHOLE size before recon (`assert_interp_frame_poses`) — landed in-repo.
2. **Subsample** midframes for densify (`select_densify_frames`); keep full rate only for video/preview.
3. **Grow posed sparse** with the same strided midframes (`select_posed_sparse_frames` → `colmap_posed`) so OpenMVS neighbors see longer tracks — see [`openmvs-denser-sparse.md`](openmvs-denser-sparse.md).
4. **Milestone B** densify on full posed set (not per-leg). Do **not** concat per-clip AI recon; do **not** use OSM/BAG as hero.
5. Optional cloud **3DGS with locked extrinsics** as ghost-proof A/B.
6. MASt3R only as **matcher → posed COLMAP**, never as free-pose scene builder for the hero cloud.

## Links
- DUSt3R global align (why free pairwise clouds need fusion): https://arxiv.org/abs/2312.14132
- OpenMVS known poses / densify: https://github.com/cdcseacave/openMVS/wiki/Usage
- Monocular scale drift vs GS: https://arxiv.org/abs/2507.03737
- Related: [`openmvs-densify.md`](openmvs-densify.md), [`openmvs-denser-sparse.md`](openmvs-denser-sparse.md), [`photo-consistent-facades.md`](photo-consistent-facades.md)
