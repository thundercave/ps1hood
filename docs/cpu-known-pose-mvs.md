# PS1 Hood — CPU known-pose MVS digest

**Scope:** photo-derived sparse 3D from Street View multi-view that can read as PS1.  
**Hard skips:** GPU PatchMatch / COLMAP dense stereo, NeRF / Gaussian splat training, cadastral/OSM extruded boxes.  
**Smoke context (2026-09-06):** 29 rectilinear crops (`*_hNNN_p+0.jpg`), auto `SIMPLE_RADIAL` per image, `exhaustive_matcher` OK, **`mapper` → “No good initial image pair”** (classic same-pano multi-heading / weak baseline). Next step = posed `point_triangulator`, not more mapper retries.

---

## 1) COLMAP `point_triangulator` (known poses)

### Digest
Feed fixed poses; only triangulate. Do **not** run `mapper`.

**Manual model layout**
```
manual_sparse/
  cameras.txt      # CAMERA_ID MODEL W H params…
  images.txt       # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
                   # (blank line after every pose line — POINTS2D empty)
  points3D.txt     # empty file
```

**Pose convention (COLMAP):** `(QW…TZ)` is **world→camera**. Camera center  
`C = -R^T * t` — if you dump pano ENU centers into `TX TY TZ` without converting, triangulation is garbage / empty.

**Exact CLI (CPU-friendly)**
```bash
# 1) Features — pin PINHOLE for rectilinear crops; share one camera if same FoV
colmap feature_extractor \
  --database_path $P/database.db \
  --image_path $P/images \
  --ImageReader.camera_model PINHOLE \
  --ImageReader.single_camera 1 \
  --ImageReader.camera_params "$FX,$FY,$CX,$CY" \
  --SiftExtraction.use_gpu 0

# 2) Optional: overwrite DB intrinsics from cameras.txt via pycolmap if distortion known

# 3) Matcher — prefer pairs with REAL baseline (different pano IDs)
colmap exhaustive_matcher \
  --database_path $P/database.db \
  --SiftMatching.use_gpu 0 \
  --SiftMatching.guided_matching 1
# Better for drive strips: sequential_matcher / spatial_matcher with GPS priors
# Write a custom match list that SKIPS same-pano heading pairs

# 4) Triangulate (clear_points remaps image_id by filename — use it)
colmap point_triangulator \
  --database_path $P/database.db \
  --image_path $P/images \
  --input_path $P/manual_sparse \
  --output_path $P/triangulated \
  --clear_points 1 \
  --Mapper.tri_ignore_two_view_tracks 0 \
  --Mapper.filter_min_tri_angle 0.5
```

### Street View–specific pitfalls
| Failure | Cause | Fix |
|---------|--------|-----|
| Empty / tiny `points3D` | Many headings share **one pano center** → ~pure rotation; matches verify as H, tri-angle ≈ 0 | Only match crops from **different** panos; drop same-center pairs from match list |
| `mapper`: no initial pair (smoke) | Same as above | Stop using mapper; use `point_triangulator` with known poses |
| image_id / name crash | DB ids ≠ `images.txt` | `--clear_points 1` (default on recent COLMAP) |
| Wrong camera model | `SIMPLE_RADIAL` auto + guessed f on already-rectilinear crops | `PINHOLE`, `fx = (W/2)/tan(hfov/2)` from crop FoV; `cx=W/2`, `cy=H/2` (COLMAP pixel origin) |
| Scale / up-axis | Mixing ECEF vs ENU; Y-up vs Z-up; putting **C** into `t` | ENU meters; `t = -R * C`; Z-up ENU → optional `model_orientation_aligner` later |
| Near-duplicates | Consecutive panos ~1–5 m + overlapping headings | Keep ≥1 heading/pano facing facade; cull near-identical crops; lower `filter_min_tri_angle` carefully (0.5–1.0°) |
| Still empty after poses | No cross-pano matches; `ignore_two_view_tracks` | Guided matching; allow two-view tracks; check pair inlier counts in DB |

### Links
1. https://colmap.github.io/faq.html#reconstruct-sparse-dense-model-from-known-camera-poses  
2. https://colmap.github.io/format.html (images.txt quaternion + `C=-R^T t`)  
3. https://colmap.readthedocs.io/en/latest/faq.html (same FAQ; min tri-angle / two-view tracks)  
4. https://github.com/colmap/colmap/pull/2252 (`clear_points` default)  
5. https://github.com/colmap/colmap/blob/main/python/examples/panorama_sfm.py (pano→perspective rig pattern)

---

## 2) OpenCV multi-view densify / triangulation (known P)

### Digest
OpenCV has **no** lightweight patch-MVS. Use sparse multi-view DLT (+ optional pairwise SGBM on **translated** pairs only).

**Sparse (primary for SV)**
```python
# P = K @ [R|t]  world→cam, 3x4
# pts1, pts2: 2xN float64, undistorted if needed
X_h = cv2.triangulatePoints(P1, P2, pts1, pts2)  # 4xN
X = (X_h[:3] / X_h[3]).T
# Filter: cheirality (z>0 both cams), reproj err < 2–4 px, tri-angle > ~1°, finite depth
```
N-view: `cv2.sfm.triangulatePoints(points2d_list, P_list)` (opencv-contrib) — DLT over tracks.

**Sequential stereo (only if baseline exists)**
1. Pick pano pairs with ‖C_i − C_j‖ ≳ 2–5 m, similar heading toward facade.  
2. `cv2.stereoRectify` from known `R,t` between cams → `StereoSGBM` → `reprojectImageTo3D`.  
3. Same-pano heading pairs: **skip** (rectify degenerates / disparity noise).

**“Patch-based CPU densify”**
- Core OpenCV: none worth wiring for SV smoke.  
- Heavier CPU options (only if sparse is too thin): OpenMVS densify (AGPL), classic PMVS — not first try.  
- Do **not** call COLMAP `patch_match_stereo` (CUDA).

**What works on sparse SV baselines**
- Match **across panos** along the drive (ORB/SIFT + ratio + essential/fundamental check with known K; or force known relative pose and RANSAC on reprojection).  
- Build tracks ≥2–3 views with different centers → triangulate → statistical outlier filter.  
- Expect facade/ground clusters, holes in sky/cars — enough for plane RANSAC / PS1 quads.

### Links
1. https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html (`triangulatePoints`, `stereoRectify`, `reprojectImageTo3D`)  
2. https://docs.opencv.org/4.x/d0/dbd/group__triangulation.html (`cv::sfm::triangulatePoints` N-view)  
3. https://docs.opencv.org/4.x/d2/d85/classcv_1_1StereoSGBM.html  
4. https://nghiaho.com/?p=2379 (OpenCV SfM → P matrices → densify handoff)  
5. https://opencv.org/structure-from-motion-in-opencv/ (CPU SfM overview; no GPU)

---

## 3) Meshing sparse SV clouds → PS1 low-poly (no cadastral)

### Digest
Sparse SfM clouds ≠ watertight city meshes. Prefer **plane clustering → quads**, then light simplify — that *is* the PS1 look. Poisson is a fallback, not the hero.

**Pipeline**
1. **Clean:** statistical outlier removal; voxel downsample (~0.15–0.5 m street scale).  
2. **Plane RANSAC (Open3D `segment_plane`)** repeatedly → ground + facade slabs. Fit rectangles from inlier AABB/PCA; extrude height from point extents. This is photo-derived geometry, not OSM footprints.  
3. **If you insist on freeform mesh:**  
   - **Ball pivoting / alpha shapes** (Open3D): keep measured points, honest holes — better for sparse SV.  
   - **Poisson** (`create_from_point_cloud_poisson`, depth **6–8**): needs oriented normals; **trim** low `densities` quantile or it blobs sky/holes. Over-smooths window reveals.  
   - PCL **greedy projection** triangulation: equivalent niche to BPA; Open3D doesn’t ship GPT — use BPA.  
4. **Decimate to PS1:** `simplify_quadric_decimation`  
   - Single facade slab: **~12–80 tris** (or 1–2 quads).  
   - One building shell: **~200–800 tris**.  
   - Smoke block: **~2k–8k tris** total.  
   - Vertex clustering (`simplify_vertex_clustering`) first if Poisson exploded.  
5. Texture with known-pose facade warps (existing research pack) — not dense UV bake.

**Poisson vs BPA for this project:** BPA / plane-quads win. Poisson only if you need a continuous ground skirt and then immediately quadric-crush it.

### Links
1. https://www.open3d.org/docs/latest/tutorial/geometry/surface_reconstruction.html (Poisson, BPA, alpha)  
2. https://www.open3d.org/docs/latest/python_api/open3d.geometry.TriangleMesh.html (`simplify_quadric_decimation`, `create_from_point_cloud_poisson`)  
3. https://www.open3d.org/docs/latest/tutorial/geometry/pointcloud.html (`segment_plane`)  
4. https://www.3d-geospatial.com/point-cloud-mesh-processing-pipelines/surface-reconstruction-algorithms/ (Poisson vs BPA for facades)  
5. https://pointclouds.org/documentation/classpcl_1_1_greedy_projection_triangulation.html (PCL GPT reference)

---

## Try FIRST on the smoke (ordered)

1. **Stop `mapper`.** Confirm every crop pose: same pano id ⇒ **same C**, different R from heading.  
2. Write `manual_sparse/{cameras,images,points3D}.txt` with `PINHOLE` + `t=-R@C` (ENU). Blank POINTS2D lines.  
3. Rebuild DB: `feature_extractor` with `--ImageReader.single_camera 1` + FoV-derived fx/fy (don’t trust auto SIMPLE_RADIAL f=2304 blindly).  
4. Match with a **pair list excluding same-pano** pairs (or spatial matcher on distinct panos only) + guided matching.  
5. `point_triangulator --clear_points 1 --Mapper.tri_ignore_two_view_tracks 0 --Mapper.filter_min_tri_angle 0.5`.  
6. If still empty: dump match inlier counts for cross-pano pairs; fix scale/pose; try OpenCV `triangulatePoints` on top-N cross-pano matches as a sanity oracle.  
7. On the resulting PLY: iterative `segment_plane` → facade/ground quads → optional BPA → `simplify_quadric_decimation` to ~2–8k tris.  
8. **Do not:** `patch_match_stereo`, NeRF, OSM/cadastral boxes.

---

## Explicit exclusions
- ❌ GPU PatchMatch / COLMAP dense  
- ❌ NeRF / 3DGS as geometry source  
- ❌ Cadastral / OSM extruded footprints as primary mesh  
