# OpenMVS “not enough images in view” with thin posed sparse → 0 dense

**In-repo wiring:** `colmap_posed` grows the match + triangulator graph with
orbit keyframes **plus** subsampled FILM midframes (`select_posed_sparse_frames`
— asserts `lerp_pose` ENU + fov/width/height on the full FILM rate, keeps
midframes with ≥`POSED_MATCH_MIDFRAME_BASELINE_M` / 4 m ENU clearance, stride
`DENSIFY_MIDFRAME_STRIDE`). Matching uses **forward drive** cross-pano pairs
(each frame ↔ 3 later track mates via `cross_pano_forward_pairs`); guided
matching stays **off** by default (optional with max_image_size / max_num_matches
caps). Two-view tracks stay allowed (`tri_ignore_two_view_tracks=0`). Logs mean
track length **and** fraction of points with ≥3 views — if mean ≲ 2.2, densify
is still risky. See [`longer-tracks-no-oom.md`](longer-tracks-no-oom.md).

**Do not** treat lowering `--number-views-fuse` as a fix for neighbor selection —
fuse only runs **after** depth maps exist. OpenMVS densify auto-builds
`--view-neighbors-file` from `recon/colmap/cross_pano_pairs.txt` when present
(see §2).

Related: [`openmvs-densify.md`](openmvs-densify.md),
[`correct-geometry-ghost-duplicates.md`](correct-geometry-ghost-duplicates.md).

---

## What the error means
`SelectNeighborViews` picks stereo neighbors using **sparse-point visibility + baseline/angle**.  
If the covisibility graph is too thin (few shared points between images, bad K, tiny baseline, or grazing SV), **every** reference image fails → 0 depth maps → 0 dense pts.

205 points is not “too few” in absolute terms; it’s too **weakly shared** across views for neighbor selection (classic SfM-too-sparse for MVS). See openMVS#917, #560, #888.

Also verify **K** in `scene.mvs` (wrong fx/cx from InterfaceCOLMAP has caused the same error — #917).

---

## Ranked fixes (stay in one ENU frame)

### 1) Denser posed sparse with longer tracks ★
Goal: more points **and** higher views/point across **cross-pano** images.

| Lever | Action |
|-------|--------|
| FILM midframes + `lerp_pose` | Add subsampled midframes (every Nth, min 2 m baseline) into posed COLMAP / match graph so landmarks see more cameras |
| Cross-pano matches | Exhaustive or custom match list from `select_stereo_pairs`; skip same-center orbits |
| COLMAP triangulator | `--Mapper.tri_ignore_two_view_tracks 0`, `filter_min_tri_angle` ~0.5–1°; guided matching |
| SIFT/DSP-SIFT | More features on façades; mask sky/road if easy |
| MASt3R-as-matcher | Matches → database → `point_triangulator` with **fixed** poses (not MASt3R free poses) |
| Target | Prefer sparse with **mean ≥3 views/point** and dozens of images that share points — log COLMAP/OpenMVS visibility stats before densify |

### 2) Bypass neighbor heuristic (when sparse is OK but picky)
```bash
# view-neighbors-file: one line per ref image:
# IMAGE_ID neighbor_id neighbor_id ...
DensifyPointCloud scene.mvs --view-neighbors-file pairs.txt --resolution-level 2
```
Build `pairs.txt` from `select_stereo_pairs` / cross-pano ENU baselines (2–25 m). Forces neighbors even when sparse score is low (#942).

### 3) OpenMVS knobs (after neighbors exist)
- `--number-views 3` or `4` (estimation neighbors)
- `--number-views-fuse 2` (don’t go to 1 until you have depth maps — fuse≠neighbor select)
- Try newer OpenMVS if empty-cloud neighbor path improved (#888)
- SGM path only if you have stereo-ish pairs: `--fusion-mode -1` then `-2`

### 4) Risky / last
- Lower internal neighbor score in source (#942) — avoid unless forks  
- Blindly lowering fuse thresholds **won’t** fix “not enough images in view”

---

## Suggested smoke sequence
1. Log posed sparse: `#points`, mean views/point, #images with ≥2 shared points to some neighbor.  
2. If mean views/point ≲ 2.2 → **don’t densify yet** — grow sparse (midframes + MASt3R matcher + two-view tracks).  
3. Re-export InterfaceCOLMAP; sanity-check K in Viewer.  
4. Densify with `--view-neighbors-file` from pose pairs if auto neighbors still fail.  
5. Only then tune resolution-level / fuse.

## Parallel (not OpenMVS)
Milestone A photo-planes + OpenCV SIFT triangulate under known P already add façade geometry without OpenMVS neighbor graph.

## Links
- https://github.com/cdcseacave/openMVS/issues/917  
- https://github.com/cdcseacave/openMVS/issues/560  
- https://github.com/cdcseacave/openMVS/issues/888  
- https://github.com/cdcseacave/openMVS/issues/942 (`--view-neighbors-file`)  
