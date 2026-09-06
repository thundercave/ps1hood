# Longer tracks without guided-matching OOM (mean views/point ~2.01)

**Status (in-repo):** `colmap_posed` now builds a **forward drive** match list
(each frame ↔ up to 3 later track mates, cross-pano only), keeps
`SiftMatching.guided_matching` **off** by default, and subsamples FILM mids with
≥`POSED_MATCH_MIDFRAME_BASELINE_M` (4 m) ENU clearance so near-dupes do not steal
into two-view tracks. After triangulator, logs mean track length **and** fraction
of points with ≥3 views; warns if mean ≲ 2.2. OpenMVS densify can pass
`--view-neighbors-file` built from `cross_pano_pairs.txt` when present.

**Symptom this fixes:** FILM mids → many posed pts but mean track ≈ **2.01**
(almost pure two-view). OpenMVS neighbor select still starves. Guided matching OOM’d.

Mean ~2.0 means matches never **chain** across a third image. More two-view points ≠ longer tracks.

Related: [`openmvs-denser-sparse.md`](openmvs-denser-sparse.md),
[`openmvs-densify.md`](openmvs-densify.md).

---

## Ranked ideas (fixed ENU poses)

### 1) Sequential matcher along the drive (not exhaustive+guided) ★
Order images by `order_track` / interp sequence. Then:
```bash
colmap sequential_matcher \
  --database_path database.db \
  --SequentialMatching.overlap 15 \
  --SequentialMatching.quadratic_overlap 1 \
  --SiftMatching.guided_matching 0 \
  --SiftMatching.max_num_matches 8192
```
Quadratic overlap creates i↔i+2, i+4, … so the same keypoint can enter 3+ pair edges → tracks lengthen after `point_triangulator`.

Custom pair list from `select_stereo_pairs` is even lighter than exhaustive:
```bash
# pairs.txt: two image names per line (cross-pano, 2–25 m)
colmap matches_importer --database_path database.db --match_list_path pairs.txt --match_type pairs
```
Build pairs so each image links to **several** others (not just one neighbor) — e.g. each frame ↔ next 3 track neighbors with baseline ≥ 2 m.

### 2) Kill near-duplicate midframes in the match graph
Too-dense FILM samples → matches prefer **almost-identical** neighbors (easy 2-view tracks) and skip real multi-view façades.
- Subsample harder for **matching** (e.g. ≥ 4–6 m along track) while keeping denser frames only for texture later.
- Same min-baseline rule as PR #5, maybe raise to 4 m for the triangulator set.

### 3) Guided matching without OOM (if you still want it)
- Downscale: `--SiftExtraction.max_image_size 1600` (or 1200)
- `--SiftMatching.max_num_matches 4096` or `2048`
- Match **only** the custom pair list (never exhaustive+guided)
- CPU guided on fewer pairs beats GPU OOM on all-pairs

### 4) MASt3R / RoMa as matcher (cloud or PC if VRAM allows) ★ wired
Pairwise matches on the **same** custom cross-pano pair list → import to COLMAP DB → `point_triangulator` with fixed poses. Learned matchers often repeat across 3+ views better than SIFT on façades. Still **no free poses**.

In-repo: `ps1hood reconstruct <run> --backend mast3r` (or `--backend colmap_posed --matcher mast3r`). See [`mast3r-matcher.md`](mast3r-matcher.md).

### 5) Bypass OpenMVS neighbor starve **now**
Even with 2-view sparse, try:
```bash
DensifyPointCloud scene.mvs --view-neighbors-file pairs.txt --resolution-level 2
```
`pairs.txt` from `select_stereo_pairs`. May get *some* depth maps; quality limited until tracks grow. Worth a smoke while (1)–(2) run.

### 6) Don’t bother
- More two-view triangulation angle relax alone → more pts, still mean≈2  
- Thrashing `--number-views-fuse`  
- Exhaustive guided on full FILM rate  

---

## Success metric before densify
Aim **mean views/point ≳ 2.5–3** (or at least a fat tail of 3+ view tracks). Log histogram after triangulator; if 95%+ are exactly 2-view, matching graph is still a matching of edges, not a multi-view track graph.

## Smoke recipe (wired in `colmap_posed` / `densify`)
1. Image set = real panos + midframes every ≥4 m (`select_posed_sparse_frames`)
2. `cross_pano_pairs.txt` = each image ↔ 3 forward track mates (`cross_pano_forward_pairs`)
3. `matches_importer` on that list, guided **off** (optional guided: max_image_size 1600, max_num_matches 4096)
4. `point_triangulator` (two-view tracks OK as seed, but pairs must create 3-cycles)
5. Log mean track + % ≥3 views; target mean ≳ 2.5–3
6. If mean still ~2.0 → MASt3R matches on same pairs
7. Parallel: OpenMVS densify auto-builds `--view-neighbors-file` from the match list

