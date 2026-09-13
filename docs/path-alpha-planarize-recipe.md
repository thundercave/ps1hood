# Path α — Planarize MapAnything ENU PLY (implementation recipe)

**Audience:** Chief / PC implementers  
**Date:** 2026-09-13  
**Status:** GATE PASSED on PC — MA ENU Product (579613 pts, |ΔC|centroid 7.78 m, z p50 2.71 vs cam u 2.98). Flow ~99k remains backup only.  
**Product rule:** photo-derived geometry. **No OSM/3DBAG hero shells. No Poisson hero walls.**

**Wire into existing stack — do not invent a parallel façade pipeline:**

| Existing | Role |
|----------|------|
| `src/ps1_hood/reconstruct/photo_planes.py` | Hypotheses → ZNCC score → accept; `plane_dict_for_obj` |
| `src/ps1_hood/reconstruct/facades.py` | PLY read, ground fit, ortho warp, OBJ/MTL write (`extract_facades`) |
| `tests/test_photo_planes.py` | Homography / ZNCC / fail-loud tests |
| `pipeline._facade_pass` | Already calls `extract_facades(cloud_ply, …)` after densify |

**Path α change:** MA ENU PLY → **Open3D vertical plane seeds** → **same** `score_vertical_plane` / texture path. Heading×distance hyps stay as fallback / complement, not the hero for dense MA.

Sacred: one ENU world frame; locked cam2world from `align/` / `lerp_pose`. Plane form in `photo_planes`: `n·X + d = 0`, `n_u ≈ 0`.

---

## 1) One-page pipeline

```text
MapAnything ENU PLY (~580k)
        │
        ▼
  load XYZ (+ optional RGB)          facades._read_ply_xyz  OR  o3d.io.read_point_cloud
        │
        ▼
  voxel_down_sample(0.08 m)          ~30–80k pts on smoke block
        │
        ▼
  estimate_normals(knn=30)
  orient_normals_towards_camera_location(cam_centroid)
        │
        ├──────────────────────────────────────┐
        ▼                                      ▼
  iterative segment_plane                 detect_planar_patches  (optional alt)
  (prefer for α1)                         needs normals; returns OBBs
        │                                      │
        ▼                                      ▼
  keep |n·up| < 0.15 (vertical)           same vertical filter on OBB R[:,2]
  peel inliers; loop while residual ≥ N
        │
        ▼
  ground: near-horizontal plane           |n·up| > 0.85  (or reuse _fit_ground_z)
        │
        ▼
  per vertical plane:
    inliers → OBB / extent in (right, up)
    → 4 ENU corners (BL,BR,TR,TL)
    → hyp {n, d, center, width_m, height_m, source="ma_segment"}
        │
        ▼
  photo_planes.score_vertical_plane       ZNCC gate (reuse Milestone A)
  (pick 2–4 cross-pano frontal SV views)
        │
        ├─ accept → facades._warp_facade_texture → RGB555 bake
        └─ reject → drop (fail-loud log)
        │
        ▼
  residual organic pts                    OMIT / Path δ billboard later
  (trees, cars, poles)                    DO NOT Poisson / Instant Meshes in α1
        │
        ▼
  planes.json + facades.obj + textures/facade_XX.jpg|png
        │
        ▼
  Studio (existing facades.obj path)
```

---

## 2) Step-by-step with numbers (~580k smoke block)

### 2.1 Load

- Input: product MA ENU PLY (post–PR #14 fuse).  
- Prefer `open3d.io.read_point_cloud` if Open3D present; else reuse `facades._read_ply_xyz`.  
- Gate (already done on PC, re-check in CLI):  
  - `|centroid_cloud − centroid_cams| < 25 m` (smoke passed 7.78 m)  
  - `|z_p50 − cam_u_p50| < 5 m` (passed 2.71 vs 2.98)

### 2.2 Voxel downsample

| Param | Recommend | Notes |
|-------|-----------|--------|
| `voxel_size` | **0.08 m** default | Range **0.05–0.15 m** |
| Target after DS | ~20k–100k | 0.05 → denser / slower RANSAC; 0.15 → may merge thin walls |

```python
pcd = pcd.voxel_down_sample(voxel_size=0.08)
```

Also fine: keep `facades._voxel_downsample_xyz(xyz, 0.20)` only for ground seed; for plane peel use **0.08**.

### 2.3 Normals (camera-facing / toward street)

```python
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30)
)
# cam_centroid = mean of locked ENU camera centers (e,n,u)
pcd.orient_normals_towards_camera_location(cam_centroid)
```

**Gotcha:** `detect_planar_patches` **requires** normals. `segment_plane` does not, but normals help the vertical filter and later residual cleanup.

### 2.4 Iterative `segment_plane` vs `detect_planar_patches`

| API | Prefer when | Skip when |
|-----|-------------|-----------|
| **`segment_plane`** (α1 default) | Want controllable peel + hard vertical gate after each fit; smoke block; fail-loud count | — |
| **`detect_planar_patches`** | One-shot multi-plane; already have good normals; want OBB extents for free | No normals; noisy organic-heavy cloud without DS |

**α1: iterative `segment_plane` only.**

```text
distance_threshold = 0.08   # m (match voxel; try 0.05–0.12)
ransac_n           = 3
num_iterations     = 1000   # bump to 2000 if under-segmenting
min_inliers        = 400    # after 0.08 m DS on ~580k; tune 250–800
max_planes         = 24     # hard cap before ZNCC; Studio usually keeps ≤12
residual_stop      = 1500   # stop peeling when leftover pts < this
```

Loop:

1. `model, inliers = pcd.segment_plane(distance_threshold, ransac_n, num_iterations)`  
   - `model` is length-4: `a,b,c,d` with `aX+bY+cZ+d = 0` → map to photo_planes: `n=[a,b,c]/‖·‖`, `d_plane=d/‖·‖` (same form `n·X + d = 0`).
2. Vertical filter: `|n · up| < 0.15` with `up=(0,0,1)`. If fail → treat as **ground candidate** if `|n·up| > 0.85`, else discard inliers into `organic` bucket and continue (or skip removing non-vertical once to avoid eating façades — prefer: if non-vertical and not ground, **do not remove** those inliers from residual on first pass; only remove vertical accepts + one ground).  
   **Practical α1 rule:**  
   - Vertical accept → remove inliers, keep plane.  
   - First strong horizontal → save as ground, remove.  
   - Other (roof/car) → remove from residual into `organic` so RANSAC can find next wall (else street clutter dominates).
3. Stop when `len(points) < residual_stop` or `planes >= max_planes` or inlier count `< min_inliers`.

### 2.5 Ground plane

- Either first horizontal from loop, or reuse `facades._fit_ground_z` (percentile + RANSAC z=const).  
- Ground mesh: AABB of cloud XY at `ground_z` (already in `_write_obj`).  
- Texture: existing satellite ortho path in `facades._ground_satellite_texture` — keep.

### 2.6 Per-plane → 4 ENU corners

For inlier points `P` and unit normal `n` (flip so `n` points toward `cam_centroid`):

```text
right = normalize(cross(up, n))
up_t  = cross(n, right)          # may differ slightly from +U if n not perfectly vertical
# Force: up_t = (0,0,1) projected / orthonormalize for PS1 quads
up_t  = normalize(up - n*(up·n))
right = cross(up_t, n)           # re-orthonormalize
```

Extents in plane frame:

```text
c  = mean(P) projected onto plane
s  = (P - c) · right   →  s_min, s_max
t  = (P - c) · up_t    →  t_min, t_max
# Optional pad 0.05 m; clamp height: t_min ≥ ground_z - 0.2
corners = [
  c + s_min*right + t_min*up_t,  # BL
  c + s_max*right + t_min*up_t,  # BR
  c + s_max*right + t_max*up_t,  # TR
  c + s_min*right + t_max*up_t,  # TL
]
width_m  = s_max - s_min
height_m = t_max - t_min
```

Reject quads with `width_m < 1.5` or `height_m < 1.5` or `width_m*height_m < 4` (m²).

**OBB shortcut:** `pcd_inliers.get_oriented_bounding_box()` — then take the two long edges; if using `detect_planar_patches`, OBB `R[:,2]` is the patch normal (Open3D docs). Still convert to the 4-corner convention above for `facades._warp_facade_texture` / `sample_plane_quad_world` (BL,BR,TR,TL).

**Do not** use convex hull triangles as hero mesh — hull → rectangle only.

### 2.7 Residual organic

- Collect leftover pts after peel.  
- **α1:** write optional `residual.ply` for Studio debug; **omit from mesh**.  
- Path δ later: billboards for trees/signs.  
- **Forbidden in α1:** Poisson, BPA, Instant Meshes on façades or residual.

---

## 3) Photo-consistency gate (reuse Milestone A)

Call existing `photo_planes.score_vertical_plane` / wire hyps into `search_photo_consistent_planes` path.

### Thresholds (denser MA support)

| Gate | Value | Rationale |
|------|-------|-----------|
| `zncc_accept` | **0.40** default for MA seeds | A was 0.35–0.48 on flow; denser support → slightly stricter |
| Soft warn | mean ZNCC of kept `< 0.42` | Log; don't hard-fail if count OK |
| Hard fail | 0 planes after ZNCC | Existing fail-loud in `extract_facades` |
| `patch` | 64 (A) or **96** if wall >12 m | Larger walls → bigger patch |
| `min_views` | 2 | Cross-pano required |

### Which SV views (`_pick_scoring_views` already close)

Reuse / tighten existing `_pick_scoring_views` in `photo_planes.py`:

| Constraint | Value |
|------------|-------|
| Distance cam→plane center | **2–25 m** (existing 2–35; prefer ≤25) |
| Frontal `(-fwd)·n` | **≥ 0.35** for texture bake; **≥ 0.25** for ZNCC score |
| Cross-pano | Distinct `pano_id` only |
| Max views | 4 (1 ref + ≤3 src) |
| Grazing reject | frontal `< 0.20` → skip (facades `_pick_frontal_camera` already) |

Flip `n` toward camera before scoring (A already does).

**Integration shape:**

```python
hyps = planes_from_mapanything_ply(ply, cam_centers, ...)  # NEW
# each hyp: n, d, center, width_m, height_m, source="ma_segment"
# Then score exactly like search_photo_consistent_planes loop
# OR add source branch inside search_photo_consistent_planes
```

Do **not** re-RANSAC the whole MA cloud in image space as a second geometry source — segment once in 3D, **validate** in image space.

---

## 4) Texture bake

Reuse `facades._warp_facade_texture` + `_pick_frontal_camera`:

1. Rectilinear SV crop already on disk (`frame["path"]`) — **never** warp raw equirect.  
2. `getPerspectiveTransform(quad_uv → ortho)` + `warpPerspective`.  
3. Naming (Studio today): `textures/facade_{i:02d}.jpg`, `textures/ground.jpg`.  
4. **α1 add:** optional PS1 bake sibling:

```text
textures/facade_{i:02d}.jpg          # full ortho (debug / Studio)
textures/facade_{i:02d}_rgb555.png   # nearest resize ≤128–256, RGB555
```

RGB555 sketch (keep in facades or small util):

```python
def bake_rgb555(bgr, max_side=256):
    h, w = bgr.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        bgr = cv2.resize(bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_NEAREST)
    # 5-bit per channel
    out = (bgr.astype(np.uint16) >> 3) << 3
    return out.astype(np.uint8)
```

`planes.json` suggestion (alongside OBJ):

```json
{
  "frame": "ENU",
  "source": "mapanything_planarize",
  "ground_z": 1.2,
  "planes": [
    {
      "id": "facade_00",
      "n": [0.91, 0.42, 0.0],
      "d": -12.3,
      "quad": [[...],[...],[...],[...]],
      "width_m": 8.2,
      "height_m": 9.1,
      "zncc": 0.47,
      "texture": "textures/facade_00.jpg",
      "inliers": 2200
    }
  ],
  "residual_points": 12000,
  "gates": {"mean_zncc": 0.45, "plane_count": 7}
}
```

---

## 5) Minimal pasteable Python (Open3D + numpy)

Self-contained seeds → quads. Wire output into `photo_planes` hyp dicts. Verified against Open3D latest `PointCloud` API (`segment_plane`, `voxel_down_sample`, `estimate_normals`, `orient_normals_towards_camera_location`).

```python
"""Path α sketch: MA ENU PLY → vertical façade quads (Open3D).

Not a full app — paste into photo_planes.py / call from facades.extract_facades.
Requires: open3d, numpy. Plane form matches photo_planes: n·X + d = 0.
"""
from __future__ import annotations

import numpy as np

try:
    import open3d as o3d
except ImportError as e:  # pragma: no cover
    raise ImportError("Path α planarize needs open3d (optional dep)") from e


UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)


def load_pcd_enu(ply_path: str):
    pcd = o3d.io.read_point_cloud(ply_path)
    if pcd.is_empty():
        raise RuntimeError(f"empty PLY: {ply_path}")
    return pcd


def prepare_cloud(
    pcd,
    cam_centroid: np.ndarray,
    *,
    voxel: float = 0.08,
    knn: int = 30,
):
    pcd = pcd.voxel_down_sample(voxel_size=voxel)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    pcd.orient_normals_towards_camera_location(np.asarray(cam_centroid, dtype=np.float64))
    return pcd


def _unit_plane(model: np.ndarray) -> tuple[np.ndarray, float]:
    """Open3D [a,b,c,d] → unit n, d with n·X + d = 0."""
    m = np.asarray(model, dtype=np.float64).reshape(4)
    n = m[:3]
    nrm = np.linalg.norm(n) + 1e-12
    return n / nrm, float(m[3] / nrm)


def _flip_n_toward(n: np.ndarray, d: float, target: np.ndarray) -> tuple[np.ndarray, float]:
    # Point on plane near origin projection of target
    p = target - (n @ target + d) * n
    if n @ (target - p) < 0:
        return -n, -d
    return n, d


def inliers_to_quad(
    pts: np.ndarray,
    n: np.ndarray,
    d: float,
    *,
    ground_z: float | None = None,
    min_width: float = 1.5,
    min_height: float = 1.5,
) -> dict | None:
    """pts (N,3) inliers → rectangular façade corners BL,BR,TR,TL in ENU."""
    n = np.asarray(n, dtype=np.float64)
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 20:
        return None
    c = pts.mean(axis=0)
    c = c - (n @ c + d) * n
    up_t = UP - n * float(n @ UP)
    un = np.linalg.norm(up_t)
    if un < 1e-6:
        return None
    up_t = up_t / un
    right = np.cross(up_t, n)
    right /= np.linalg.norm(right) + 1e-12
    # re-orthonormalize up
    up_t = np.cross(n, right)
    up_t /= np.linalg.norm(up_t) + 1e-12

    s = (pts - c) @ right
    t = (pts - c) @ up_t
    s0, s1 = float(s.min()), float(s.max())
    t0, t1 = float(t.min()), float(t.max())
    if ground_z is not None:
        # lift bottom toward ground if floating noise
        t0 = max(t0, float(ground_z) - float(c @ up_t) - 0.2)
    width_m = s1 - s0
    height_m = t1 - t0
    if width_m < min_width or height_m < min_height or width_m * height_m < 4.0:
        return None
    corners = np.stack(
        [
            c + s0 * right + t0 * up_t,
            c + s1 * right + t0 * up_t,
            c + s1 * right + t1 * up_t,
            c + s0 * right + t1 * up_t,
        ],
        axis=0,
    )
    return {
        "n": n,
        "d": float(d),
        "center": corners.mean(axis=0),
        "width_m": width_m,
        "height_m": height_m,
        "corners": corners,
        "source": "ma_segment",
    }


def segment_vertical_planes(
    pcd,
    cam_centroid: np.ndarray,
    *,
    distance_threshold: float = 0.08,
    ransac_n: int = 3,
    num_iterations: int = 1000,
    min_inliers: int = 400,
    max_planes: int = 24,
    residual_stop: int = 1500,
    vertical_dot_max: float = 0.15,
    ground_dot_min: float = 0.85,
) -> tuple[list[dict], dict | None, o3d.geometry.PointCloud]:
    """Iterative RANSAC peel → vertical façade hyps + optional ground + residual cloud.

    Returns (facade_hyps, ground_dict_or_None, residual_pcd).
    Each hyp is ready for photo_planes.score_vertical_plane / plane_dict_for_obj.
    """
    cam_centroid = np.asarray(cam_centroid, dtype=np.float64)
    work = pcd
    facades: list[dict] = []
    ground = None

    while len(work.points) >= residual_stop and len(facades) < max_planes:
        model, inliers = work.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
        )
        if len(inliers) < min_inliers:
            break
        n, d = _unit_plane(model)
        n, d = _flip_n_toward(n, d, cam_centroid)
        inl_pcd = work.select_by_index(inliers)
        pts = np.asarray(inl_pcd.points)
        abs_up = abs(float(n @ UP))

        if abs_up < vertical_dot_max:
            # Force vertical n for photo_planes convention
            n_xy = n.copy()
            n_xy[2] = 0.0
            nn = np.linalg.norm(n_xy)
            if nn < 1e-6:
                work = work.select_by_index(inliers, invert=True)
                continue
            n_xy /= nn
            # recompute d from inlier centroid
            c = pts.mean(axis=0)
            d_v = float(-n_xy @ c)
            n_xy, d_v = _flip_n_toward(n_xy, d_v, cam_centroid)
            quad = inliers_to_quad(pts, n_xy, d_v)
            if quad is not None:
                quad["inliers"] = len(inliers)
                facades.append(quad)
            work = work.select_by_index(inliers, invert=True)
        elif abs_up > ground_dot_min and ground is None:
            ground = {
                "n": n,
                "d": d,
                "z": float(np.median(pts[:, 2])),
                "inliers": len(inliers),
                "source": "ma_segment_ground",
            }
            work = work.select_by_index(inliers, invert=True)
        else:
            # roof / clutter — strip so next RANSAC can find walls
            work = work.select_by_index(inliers, invert=True)

    return facades, ground, work


# Optional one-shot alternative (not α1 default):
def detect_patches_vertical(pcd, *, min_num_points: int = 400, vertical_dot_max: float = 0.15):
    """Uses detect_planar_patches → OBBs; filter vertical; convert via OBB axes."""
    obbs = pcd.detect_planar_patches(
        normal_variance_threshold_deg=60,
        coplanarity_deg=75,
        outlier_ratio=0.75,
        min_plane_edge_length=1.5,
        min_num_points=min_num_points,
    )
    out = []
    for obb in obbs:
        R = np.asarray(obb.R)
        n = R[:, 2]  # Open3D: third column = patch normal
        if abs(float(n @ UP)) > vertical_dot_max:
            continue
        # sample OBB corners / extent — prefer inlier crop via obb for quads
        # (implementer: crop pcd by obb, then inliers_to_quad)
        out.append(obb)
    return out
```

**API gotchas (verified):**

1. `segment_plane` → `(model[4], inlier_indices)`; model is `ax+by+cz+d=0`.  
2. `detect_planar_patches` → `list[OrientedBoundingBox]`; **`R[:,2]` = normal**; needs normals; `extent[2]` is thickness (nonzero).  
3. `orient_normals_towards_camera_location` takes a **single** camera point — use cam centroid (good enough for street block).  
4. Homography in repo uses **`+ outer(t_rel, n_ref)/c`** (fixed vs old Scrapy minus-sign sketch) — do not “fix” it back.  
5. `open3d` is **not** in core `pyproject.toml` deps yet — add optional extra `planarize = ["open3d>=0.18"]` (or pin what PC has).  
6. Corner order for textures must stay **BL, BR, TR, TL** (`sample_plane_quad_world` / `_warp_facade_texture`).

---

## 6) Repo wiring (extend, don't fork)

### Where

| Piece | Path |
|-------|------|
| NEW segment helpers | `src/ps1_hood/reconstruct/photo_planes.py` (bottom) — `segment_vertical_planes`, `inliers_to_quad`, `planes_from_mapanything_ply` |
| OR thin module | `src/ps1_hood/reconstruct/planarize.py` imported by `photo_planes` / `facades` — only if file size hurts; **prefer photo_planes** to keep hyp→score colocated |
| Texture / OBJ | `facades.extract_facades` — add branch when PLY point count ≫ sparse (e.g. `≥ 50_000`) **or** explicit flag |
| Tests | `tests/test_photo_planes.py` + new `tests/test_planarize.py` (synthetic wall cloud) |
| Dep | `pyproject.toml` optional-extra `planarize = ["open3d>=0.18"]` |

### `extract_facades` change (sketch)

```python
# In extract_facades(...):
#   ..., planarize: bool | None = None, zncc_accept: float = 0.40

use_planarize = planarize if planarize is not None else (len(xyz) >= 50_000)
if use_planarize:
    from ps1_hood.reconstruct.photo_planes import planes_from_mapanything_ply
    cam_c = np.mean([[f["e"], f["n"], f["u"]] for f in frames], axis=0)
    hyps = planes_from_mapanything_ply(Path(ply_path), cam_c, ground_z=ground_z)
    accepted = score_hyps(hyps, frames, zncc_accept=zncc_accept)  # factor from search_*
    # meta["source"] = "mapanything_planarize"
else:
    accepted = search_photo_consistent_planes(...)  # existing Milestone A
```

### CLI flag sketch

```text
ps1hood planarize <name> \
  --ply runs/<name>/recon/cloud_mapanything.ply \   # or densify product path
  --voxel 0.08 \
  --zncc-accept 0.40 \
  --max-planes 12 \
  --out runs/<name>/recon/facades.obj

# or extend densify / reconstruct:
ps1hood densify <name> --backend mapanything
ps1hood facades <name> --planarize --zncc-accept 0.40
```

Minimal: **no new command required** — gate inside `_facade_pass` when MA product PLY detected (`meta["backend"]=="mapanything"` or point count). Explicit `ps1hood facades --planarize` is clearer for Chief smoke.

### Inputs / outputs

| In | Out |
|----|-----|
| MA ENU PLY | `recon/planes.json` |
| Posed frames (same as today: `e,n,u,heading,pitch,fov,path,pano_id`) | `recon/facades.obj` + `.mtl` |
| Locked poses from align (already on frames) | `recon/textures/facade_XX.jpg` (+ optional `_rgb555.png`) |
| | `recon/residual_organic.ply` (debug, optional) |

COLMAP sparse optional — **not required** for α1 if frames JSON has poses + paths.

---

## 7) Fail-loud gates

Centroid/z already enforced at MA export. Add at planarize end:

| Gate | Fail if | Action |
|------|---------|--------|
| Plane count floor | `< 3` vertical accepts after ZNCC on smoke block | `log.error` + empty walls / non-zero exit on CLI |
| Mean ZNCC | `< 0.38` across kept | Fail-loud (tunable; A saw 0.42–0.48) |
| Max plane tilt | any kept `|n·up| > 0.20` | Reject that plane pre-texture |
| Max plane distance | plane center XY farther than **40 m** from nearest camera | Reject (floaters) |
| Min inliers / plane | `< 200` post-DS | Reject |
| Empty residual warning | residual `< 5%` of DS cloud | Warn (over-peeled?) |
| Huge residual | residual `> 70%` | Warn (under-segmented; loosen distance_threshold or min_inliers) |
| Texture coverage | textured / planes `< 0.5` | Warn; still ship geometry |

Never invent OSM/BAG or flow-RANSAC walls to “fill” a failed gate.

---

## 8) First PR scope (smallest shippable)

**In**

1. Optional `open3d` extra.  
2. `planes_from_mapanything_ply` / `segment_vertical_planes` + `inliers_to_quad` in `photo_planes.py` (or `planarize.py`).  
3. `extract_facades(..., planarize=True)` branch → score with existing ZNCC → existing OBJ/texture writers.  
4. `planes.json` write.  
5. Fail-loud gates above.  
6. Unit tests: synthetic vertical wall + ground cloud → ≥1 quad; tilt reject; empty PLY fail.  
7. Docs one-liner in `docs/photo-consistent-facades.md` or ROADMAP: Path α planarize.

**Out (explicit non-goals for PR)**

- Instant Meshes / Poisson / BPA  
- Organic remesh / billboards (Path δ)  
- RGB555 mandatory (optional helper OK)  
- MASt3R / OpenMVS texture atlas  
- OSM/BAG  
- Changing ENU fuse (already merged)

### Acceptance tests

1. Synthetic: 2 vertical walls + ground points → `segment_vertical_planes` returns 2 hyps, `|n·up|<0.15`, quad areas sane.  
2. ZNCC: accepted hyp with consistent warped views → `ok` (reuse existing tests).  
3. Wrong depth / grazing → reject.  
4. Smoke block (PC): MA PLY ~580k → `planes ≥ 3`, `mean_zncc ≥ 0.38`, facades.obj loads in Studio, windows roughly align on ≥1 wall.  
5. `planarize=False` / small PLY → legacy heading×distance path unchanged (regession).  
6. No BAG/OSM import in new code paths (`rg` check in review).

---

## 9) Best links (short)

| Resource | URL |
|----------|-----|
| Open3D point cloud (segment_plane, normals, voxel) | https://www.open3d.org/docs/latest/tutorial/geometry/pointcloud.html |
| Open3D `PointCloud` API | https://www.open3d.org/docs/latest/python_api/open3d.geometry.PointCloud.html |
| Instant Meshes (later residual only) | https://github.com/wjakob/instant-meshes |
| Texture2LoD3 (pano→façade texture ideas) | https://github.com/WenzhaoTang/Texture2LoD3 |
| MVS-Texturing (later atlas; BSD-ish) | https://github.com/nmoehrle/mvs-texturing |
| In-repo A | `photo_planes.py`, `facades.py`, `docs/photo-consistent-facades.md` |
| Prior R&D | `/workspace/ps1hood-compare-and-pathforward-extra.md` §A–B |

---

## 10) Implementer checklist (copy into PR)

- [ ] `uv sync --extra planarize` (or equiv) on PC  
- [ ] `planes_from_mapanything_ply` returns photo_planes-compatible hyps  
- [ ] ZNCC via existing `score_vertical_plane` (homography **plus** sign)  
- [ ] `extract_facades` source=`mapanything_planarize` in meta  
- [ ] Studio opens `facades.obj` without BAG shells  
- [ ] Gates logged; 0 planes → ground only, error log  
- [ ] Residual not meshed  
- [ ] Flow cloud path still works as backup when MA absent  

**Done when:** Chief can MessageSubagent this file and PC ships smoke-block planar façades without further research.
