"""Path α — MapAnything ENU PLY → vertical plane seeds (Open3D peel / numpy fallback).

Segment once in 3D, validate with Milestone A ZNCC (`photo_planes`). Residual
organic points are omitted (no Poisson / Instant Meshes in α1). OSM/BAG never
used as hero geometry.

Plane form matches photo_planes: n·X + d = 0, n_u ≈ 0.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)

# Recipe §2 defaults (α1).
DEFAULT_VOXEL_M = 0.08
DEFAULT_PLANE_DIST_M = 0.08
DEFAULT_MIN_INLIERS = 400
DEFAULT_MAX_PLANES = 24
DEFAULT_RESIDUAL_STOP = 1500
DEFAULT_VERTICAL_DOT = 0.15
DEFAULT_GROUND_DOT = 0.85
DEFAULT_ZNCC_ACCEPT_MA = 0.35
PLANARIZE_AUTO_MIN_POINTS = 50_000
DEFAULT_AABB_PERCENTILE = (5.0, 95.0)
DEFAULT_MAX_WIDTH_M = 25.0
DEFAULT_MAX_HEIGHT_M = 15.0
DEFAULT_MIN_HEIGHT_M = 2.5
DEFAULT_MIN_CAM_DEPTH_M = 2.0
DEFAULT_MAX_CAM_DEPTH_M = 35.0
# Hard reject only outside this; signed |n·C+d| alone used to zero scored=0
# on street-parallel peels (PR#16 4–25). Gate uses center→cam Euclidean.
DEFAULT_REFINE_DELTAS_M = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)
# Hard depth reject only outside this (pre-refine). Soft band uses min/max_cam_depth_m.
DEFAULT_HARD_MIN_CAM_DIST_M = 1.0
DEFAULT_HARD_MAX_CAM_DIST_M = 50.0
# Split long street-slab peels into overlapping façade windows before ZNCC.
# PR-1 denser defaults (was 12 / 10 / 2) — NL row-house scale ~6–10 m.
DEFAULT_SPLIT_TRIGGER_WIDTH_M = 8.0
DEFAULT_SPLIT_WINDOW_M = 8.0
DEFAULT_SPLIT_OVERLAP_M = 2.0
DEFAULT_SCORE_CLAMP_M = 12.0
DEFAULT_MAX_HEADING_SEEDS = 24

try:
    import open3d as o3d

    HAS_OPEN3D = True
except ImportError:  # pragma: no cover
    o3d = None  # type: ignore[assignment]
    HAS_OPEN3D = False



def resolve_dense_ply(project_root: Path, source: str = "mapanything") -> Path:
    """Locate dense ENU PLY for Path α.

    ``mapanything``: mapanything/cloud.ply → recon/cloud_mapanything.ply
    ``recon``: recon/cloud.ply
    ``auto``: MapAnything first, then recon/cloud.ply
    """
    root = Path(project_root)
    ma_candidates = [
        root / "mapanything" / "cloud.ply",
        root / "recon" / "cloud_mapanything.ply",
    ]
    recon = root / "recon" / "cloud.ply"
    src = (source or "mapanything").lower().strip()
    if src == "mapanything":
        for p in ma_candidates:
            if p.is_file():
                return p
        raise FileNotFoundError(
            "Path α needs a MapAnything ENU cloud "
            f"(tried {[str(p) for p in ma_candidates]}). "
            "Run densify --backend mapanything first."
        )
    if src == "recon":
        if recon.is_file():
            return recon
        raise FileNotFoundError(f"no recon cloud at {recon}")
    if src == "auto":
        for p in ma_candidates:
            if p.is_file():
                return p
        if recon.is_file():
            return recon
        raise FileNotFoundError(f"Path α: no MapAnything or recon/cloud.ply under {root}")
    raise ValueError(f"unknown façade source {source!r} (mapanything|recon|auto)")


def _unit_plane(model: np.ndarray) -> tuple[np.ndarray, float]:
    """Open3D [a,b,c,d] → unit n, d with n·X + d = 0."""
    m = np.asarray(model, dtype=np.float64).reshape(4)
    n = m[:3]
    nrm = float(np.linalg.norm(n) + 1e-12)
    return n / nrm, float(m[3] / nrm)


def _flip_n_toward(
    n: np.ndarray, d: float, target: np.ndarray
) -> tuple[np.ndarray, float]:
    p = target - (n @ target + d) * n
    if float(n @ (target - p)) < 0:
        return -n, -d
    return n, d


def inliers_to_quad(
    pts: np.ndarray,
    n: np.ndarray,
    d: float,
    *,
    ground_z: float | None = None,
    min_width: float = 1.5,
    min_height: float = DEFAULT_MIN_HEIGHT_M,
    aabb_percentile: tuple[float, float] = DEFAULT_AABB_PERCENTILE,
    max_width: float = DEFAULT_MAX_WIDTH_M,
    max_height: float = DEFAULT_MAX_HEIGHT_M,
) -> dict[str, Any] | None:
    """Inlier pts → rectangular façade corners BL,BR,TR,TL in ENU.

    Uses percentile AABB (default 5–95) to reject outlier inliers on street-slab
    peels, then clamps extent so scoring is not dominated by multi-building
    slabs (see path-alpha-zncc-fail-rd).
    """
    n = np.asarray(n, dtype=np.float64)
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 20:
        return None
    c = pts.mean(axis=0)
    c = c - (n @ c + d) * n
    up_t = UP - n * float(n @ UP)
    un = float(np.linalg.norm(up_t))
    if un < 1e-6:
        return None
    up_t = up_t / un
    right = np.cross(up_t, n)
    right /= np.linalg.norm(right) + 1e-12
    up_t = np.cross(n, right)
    up_t /= np.linalg.norm(up_t) + 1e-12

    s = (pts - c) @ right
    t = (pts - c) @ up_t
    lo, hi = float(aabb_percentile[0]), float(aabb_percentile[1])
    s0, s1 = (float(x) for x in np.percentile(s, [lo, hi]))
    t0, t1 = (float(x) for x in np.percentile(t, [lo, hi]))
    if ground_z is not None:
        t0 = max(t0, float(ground_z) - float(c @ up_t) - 0.2)
    raw_width_m = float(s1 - s0)
    raw_height_m = float(t1 - t0)
    # Clamp huge street-slab peels about AABB center
    mid_s = 0.5 * (s0 + s1)
    mid_t = 0.5 * (t0 + t1)
    half_w = min(0.5 * (s1 - s0), 0.5 * max_width)
    half_h = min(0.5 * (t1 - t0), 0.5 * max_height)
    s0, s1 = mid_s - half_w, mid_s + half_w
    t0, t1 = mid_t - half_h, mid_t + half_h
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
        "n": n.astype(np.float64),
        "d": float(d),
        "center": corners.mean(axis=0),
        "width_m": float(width_m),
        "height_m": float(height_m),
        "raw_width_m": float(raw_width_m),
        "raw_height_m": float(raw_height_m),
        "corners": corners,
        "source": "ma_segment",
    }


def split_long_hyp(
    hyp: dict[str, Any],
    *,
    trigger_width_m: float = DEFAULT_SPLIT_TRIGGER_WIDTH_M,
    window_m: float = DEFAULT_SPLIT_WINDOW_M,
    overlap_m: float = DEFAULT_SPLIT_OVERLAP_M,
) -> list[dict[str, Any]]:
    """Split a wide peel into overlapping façade-sized windows (same n, d).

    Street-slab AABB centers often sit mid-block far from cams → depth/view
    skips (sentinel scored=0) or wrong crops. Overlapping ~8 m windows give
    ZNCC a chance on real façades (path-alpha-zncc-fail-rd Fix C; PR-1 denser
    defaults). When ``raw_width_m`` (pre-clamp percentile span) exceeds the
    clamped ``width_m``, expand the split span about the center so long peels
    still yield façade-local windows.
    """
    w = float(hyp.get("width_m") or 0.0)
    raw_w = float(hyp.get("raw_width_m") or 0.0)
    span = max(w, raw_w)
    if span <= trigger_width_m + 1e-6:
        return [hyp]
    corners = hyp.get("corners")
    n = np.asarray(hyp["n"], dtype=np.float64)
    center = np.asarray(hyp["center"], dtype=np.float64)
    if corners is not None:
        corners = np.asarray(corners, dtype=np.float64)
        # BL,BR,TR,TL — right axis from BL→BR
        right = corners[1] - corners[0]
        rn = float(np.linalg.norm(right))
        if rn < 1e-6:
            return [hyp]
        right = right / rn
        up = corners[3] - corners[0]
        un = float(np.linalg.norm(up))
        up = up / (un + 1e-12)
        origin = corners.mean(axis=0)
        # Project center onto right for span
        s_vals = (corners - origin) @ right
        s0, s1 = float(s_vals.min()), float(s_vals.max())
        t_vals = (corners - origin) @ up
        t0, t1 = float(t_vals.min()), float(t_vals.max())
        height_m = float(hyp.get("height_m") or (t1 - t0))
        # Expand to pre-clamp raw span when AABB clamp hid true length
        if raw_w > (s1 - s0) + 1e-6:
            mid = 0.5 * (s0 + s1)
            half = 0.5 * raw_w
            s0, s1 = mid - half, mid + half
    else:
        # Reconstruct local frame like inliers_to_quad
        up_t = UP - n * float(n @ UP)
        un = float(np.linalg.norm(up_t))
        if un < 1e-6:
            return [hyp]
        up_t = up_t / un
        right = np.cross(up_t, n)
        right /= np.linalg.norm(right) + 1e-12
        up = np.cross(n, right)
        up /= np.linalg.norm(up) + 1e-12
        origin = center.copy()
        half = 0.5 * span
        s0, s1 = -half, half
        height_m = float(hyp.get("height_m") or 6.0)
        t0, t1 = -0.5 * height_m, 0.5 * height_m

    step = max(window_m - overlap_m, window_m * 0.5)
    out: list[dict[str, Any]] = []
    s = s0
    while s < s1 - 1e-6:
        e = min(s + window_m, s1)
        if e - s < 1.5:
            break
        mid_s = 0.5 * (s + e)
        mid_t = 0.5 * (t0 + t1)
        c_i = origin + mid_s * right + mid_t * up
        # Re-project onto plane
        c_i = c_i - (n @ c_i + float(hyp["d"])) * n
        hw = 0.5 * (e - s)
        hh = 0.5 * (t1 - t0)
        corners_i = np.stack(
            [
                c_i - hw * right - hh * up,
                c_i + hw * right - hh * up,
                c_i + hw * right + hh * up,
                c_i - hw * right + hh * up,
            ],
            axis=0,
        )
        piece = {
            **hyp,
            "center": c_i,
            "width_m": float(e - s),
            "height_m": float(height_m),
            "corners": corners_i,
            "source": hyp.get("source") or "ma_segment",
            "split_parent": True,
        }
        out.append(piece)
        if e >= s1 - 1e-6:
            break
        s += step
    return out if out else [hyp]


def expand_hyps_for_scoring(
    hyps: list[dict[str, Any]],
    *,
    trigger_width_m: float = DEFAULT_SPLIT_TRIGGER_WIDTH_M,
    window_m: float = DEFAULT_SPLIT_WINDOW_M,
    overlap_m: float = DEFAULT_SPLIT_OVERLAP_M,
) -> list[dict[str, Any]]:
    """Expand peels into scoreable façade windows."""
    out: list[dict[str, Any]] = []
    for hyp in hyps:
        out.extend(
            split_long_hyp(
                hyp,
                trigger_width_m=trigger_width_m,
                window_m=window_m,
                overlap_m=overlap_m,
            )
        )
    return out


def _flip_n_toward_cams(
    n: np.ndarray, d: float, cams: np.ndarray, center: np.ndarray
) -> tuple[np.ndarray, float]:
    """Flip so the nearest camera has positive signed distance (in front of wall)."""
    if len(cams) == 0:
        return n, d
    nearest = cams[int(np.linalg.norm(cams - center, axis=1).argmin())]
    return _flip_n_toward(n, d, nearest)


def _ransac_plane_numpy(
    xyz: np.ndarray,
    *,
    distance: float,
    iterations: int,
    rng: np.random.Generator,
    min_points: int,
) -> tuple[np.ndarray, float, np.ndarray] | None:
    n_pts = len(xyz)
    if n_pts < max(3, min_points):
        return None
    best_inl: np.ndarray | None = None
    best_n: np.ndarray | None = None
    best_d = 0.0
    best_count = 0
    for _ in range(iterations):
        idx = rng.choice(n_pts, size=3, replace=False)
        p0, p1, p2 = xyz[idx]
        n = np.cross(p1 - p0, p2 - p0)
        nn = float(np.linalg.norm(n))
        if nn < 1e-9:
            continue
        n = n / nn
        d = float(-n @ p0)
        inl = np.abs(xyz @ n + d) < distance
        count = int(inl.sum())
        if count > best_count:
            best_count = count
            best_inl = inl
            best_n = n
            best_d = d
    if best_inl is None or best_count < min_points or best_n is None:
        return None
    pts = xyz[best_inl]
    c = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - c, full_matrices=False)
    n = vh[-1].astype(np.float64)
    nn = float(np.linalg.norm(n))
    if nn < 1e-9:
        return None
    n = n / nn
    d = float(-n @ c)
    inl = np.abs(xyz @ n + d) < distance
    if int(inl.sum()) < min_points:
        return None
    return n, d, inl


def segment_vertical_planes_numpy(
    xyz: np.ndarray,
    cam_centroid: np.ndarray,
    *,
    distance_threshold: float = DEFAULT_PLANE_DIST_M,
    num_iterations: int = 1000,
    min_inliers: int = DEFAULT_MIN_INLIERS,
    max_planes: int = DEFAULT_MAX_PLANES,
    residual_stop: int = DEFAULT_RESIDUAL_STOP,
    vertical_dot_max: float = DEFAULT_VERTICAL_DOT,
    ground_dot_min: float = DEFAULT_GROUND_DOT,
    ground_z: float | None = None,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, np.ndarray]:
    """Numpy RANSAC peel (tests / no Open3D). Same hyp shape as Open3D path."""
    remaining = np.asarray(xyz, dtype=np.float64).copy()
    cam_centroid = np.asarray(cam_centroid, dtype=np.float64)
    rng = np.random.default_rng(seed)
    facades: list[dict[str, Any]] = []
    ground: dict[str, Any] | None = None

    while len(remaining) >= residual_stop and len(facades) < max_planes:
        fit = _ransac_plane_numpy(
            remaining,
            distance=distance_threshold,
            iterations=num_iterations,
            rng=rng,
            min_points=min_inliers,
        )
        if fit is None:
            break
        n, d, inl = fit
        n, d = _flip_n_toward(n, d, cam_centroid)
        pts = remaining[inl]
        abs_up = abs(float(n @ UP))

        if abs_up < vertical_dot_max:
            n_xy = n.copy()
            n_xy[2] = 0.0
            nn = float(np.linalg.norm(n_xy))
            if nn < 1e-6:
                remaining = remaining[~inl]
                continue
            n_xy /= nn
            c = pts.mean(axis=0)
            d_v = float(-n_xy @ c)
            n_xy, d_v = _flip_n_toward(n_xy, d_v, cam_centroid)
            quad = inliers_to_quad(pts, n_xy, d_v, ground_z=ground_z)
            if quad is not None:
                quad["inliers"] = int(inl.sum())
                facades.append(quad)
            remaining = remaining[~inl]
        elif abs_up > ground_dot_min and ground is None:
            ground = {
                "n": n,
                "d": d,
                "z": float(np.median(pts[:, 2])),
                "inliers": int(inl.sum()),
                "source": "ma_segment_ground",
            }
            remaining = remaining[~inl]
        else:
            remaining = remaining[~inl]

    return facades, ground, remaining


def segment_vertical_planes_open3d(
    pcd: Any,
    cam_centroid: np.ndarray,
    *,
    distance_threshold: float = DEFAULT_PLANE_DIST_M,
    ransac_n: int = 3,
    num_iterations: int = 1000,
    min_inliers: int = DEFAULT_MIN_INLIERS,
    max_planes: int = DEFAULT_MAX_PLANES,
    residual_stop: int = DEFAULT_RESIDUAL_STOP,
    vertical_dot_max: float = DEFAULT_VERTICAL_DOT,
    ground_dot_min: float = DEFAULT_GROUND_DOT,
    ground_z: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, Any]:
    """Iterative Open3D segment_plane peel → façade hyps + ground + residual."""
    if not HAS_OPEN3D:
        raise ImportError("open3d required for segment_vertical_planes_open3d")
    cam_centroid = np.asarray(cam_centroid, dtype=np.float64)
    work = pcd
    facades: list[dict[str, Any]] = []
    ground: dict[str, Any] | None = None

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
            n_xy = n.copy()
            n_xy[2] = 0.0
            nn = float(np.linalg.norm(n_xy))
            if nn < 1e-6:
                work = work.select_by_index(inliers, invert=True)
                continue
            n_xy /= nn
            c = pts.mean(axis=0)
            d_v = float(-n_xy @ c)
            n_xy, d_v = _flip_n_toward(n_xy, d_v, cam_centroid)
            quad = inliers_to_quad(pts, n_xy, d_v, ground_z=ground_z)
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
            work = work.select_by_index(inliers, invert=True)

    return facades, ground, work


def prepare_cloud_open3d(
    ply_path: Path,
    cam_centroid: np.ndarray,
    *,
    voxel: float = DEFAULT_VOXEL_M,
    knn: int = 30,
) -> Any:
    if not HAS_OPEN3D:
        raise ImportError("open3d required")
    pcd = o3d.io.read_point_cloud(str(ply_path))
    if pcd.is_empty():
        raise RuntimeError(f"empty PLY: {ply_path}")
    pcd = pcd.voxel_down_sample(voxel_size=voxel)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    pcd.orient_normals_towards_camera_location(
        np.asarray(cam_centroid, dtype=np.float64)
    )
    return pcd


def planes_from_mapanything_ply(
    ply_path: Path,
    cam_centroid: np.ndarray,
    *,
    xyz: np.ndarray | None = None,
    ground_z: float | None = None,
    voxel: float = DEFAULT_VOXEL_M,
    distance_threshold: float = DEFAULT_PLANE_DIST_M,
    min_inliers: int = DEFAULT_MIN_INLIERS,
    max_planes: int = DEFAULT_MAX_PLANES,
    residual_stop: int = DEFAULT_RESIDUAL_STOP,
    prefer_open3d: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, int]:
    """MA ENU PLY → photo_planes-compatible vertical façade hypotheses.

    Returns (hyps, ground_or_None, residual_point_count).
    """
    ply_path = Path(ply_path)
    cam_centroid = np.asarray(cam_centroid, dtype=np.float64)

    if prefer_open3d and HAS_OPEN3D:
        pcd = prepare_cloud_open3d(ply_path, cam_centroid, voxel=voxel)
        hyps, ground, residual = segment_vertical_planes_open3d(
            pcd,
            cam_centroid,
            distance_threshold=distance_threshold,
            min_inliers=min_inliers,
            max_planes=max_planes,
            residual_stop=residual_stop,
            ground_z=ground_z,
        )
        n_residual = len(residual.points)
        backend = "open3d"
    else:
        if xyz is None:
            from ps1_hood.reconstruct.facades import (
                _read_ply_xyz,
                _voxel_downsample_xyz,
            )

            xyz = _read_ply_xyz(ply_path)
            xyz = _voxel_downsample_xyz(xyz, voxel)
        else:
            from ps1_hood.reconstruct.facades import _voxel_downsample_xyz

            xyz = _voxel_downsample_xyz(np.asarray(xyz, dtype=np.float64), voxel)
        # Scale min_inliers down for tiny synthetic clouds in tests
        min_inl = min_inliers
        if len(xyz) < min_inliers * 3:
            min_inl = max(40, len(xyz) // 20)
            residual_stop = min(residual_stop, max(30, len(xyz) // 10))
        hyps, ground, residual_xyz = segment_vertical_planes_numpy(
            xyz,
            cam_centroid,
            distance_threshold=distance_threshold,
            min_inliers=min_inl,
            max_planes=max_planes,
            residual_stop=residual_stop,
            ground_z=ground_z,
        )
        n_residual = len(residual_xyz)
        backend = "numpy"

    log.info(
        "planarize[%s]: %s vertical hyps, ground=%s, residual=%s from %s",
        backend,
        len(hyps),
        ground is not None,
        n_residual,
        ply_path.name,
    )
    return hyps, ground, n_residual


def score_planar_hyps(
    hyps: list[dict[str, Any]],
    frames: list[dict[str, Any]],
    *,
    zncc_accept: float = DEFAULT_ZNCC_ACCEPT_MA,
    max_keep: int = 16,
    patch: int = 64,
    refine: bool = True,
    refine_deltas_m: tuple[float, ...] = DEFAULT_REFINE_DELTAS_M,
    min_cam_depth_m: float = DEFAULT_MIN_CAM_DEPTH_M,
    max_cam_depth_m: float = DEFAULT_MAX_CAM_DEPTH_M,
    hard_min_cam_dist_m: float = DEFAULT_HARD_MIN_CAM_DIST_M,
    hard_max_cam_dist_m: float = DEFAULT_HARD_MAX_CAM_DIST_M,
    split_long: bool = True,
    seed_hyps: list[dict[str, Any]] | None = None,
    split_trigger_width_m: float = DEFAULT_SPLIT_TRIGGER_WIDTH_M,
    split_window_m: float = DEFAULT_SPLIT_WINDOW_M,
    split_overlap_m: float = DEFAULT_SPLIT_OVERLAP_M,
) -> list[dict[str, Any]]:
    """ZNCC-gate MA segment hyps via Milestone A scoring + ±n depth refine.

    Path α previously scored each hyp once; Milestone A searches deltas along
    ``n`` before accept. Missing refine left plane depth ~0.5–2 m off → ZNCC≈0.

    Post–PR#16/#17: Euclidean depth gate + SENTINEL telemetry landed, but a hard
    pre-refine gate still zeroed ``scored`` on street-parallel peels, and long
    AABB centers sat mid-block away from frontal cams. This path:

    1. Splits wide peels into overlapping façade windows.
    2. Hard-rejects only extreme cam distances; soft band applied per candidate
       *after* ±n refine (so refine can rescue depth).
    3. Re-picks views per candidate (esp. after normal flip).
    4. Flips ``n`` toward the nearest cam before scoring so the ref sees the front.
    5. Optional ``seed_hyps`` (Milestone A heading×distance) scored in the same
       pass + NMS → union(A, MA) promote (PR-1 hybrid).
    """
    from ps1_hood.reconstruct import photo_planes as pp
    from ps1_hood.reconstruct.photo_planes import load_view

    ma_hyps = list(hyps or [])
    extra = list(seed_hyps or [])
    for h in ma_hyps:
        if not h.get("source"):
            h["source"] = "ma_segment"
    all_hyps = ma_hyps + extra
    if not all_hyps:
        return []
    if len(frames) < 2:
        log.error(
            "planarize: need ≥2 posed frames for ZNCC; got %s — keeping none",
            len(frames),
        )
        return []

    view_cache: dict[int, Any] = {}

    def get_view(i: int):
        if i not in view_cache:
            view_cache[i] = load_view(frames[i], i)
        return view_cache[i]

    cams = np.array(
        [[float(f["e"]), float(f["n"]), float(f["u"])] for f in frames],
        dtype=np.float64,
    )

    if split_long:
        score_hyps = expand_hyps_for_scoring(
            all_hyps,
            trigger_width_m=split_trigger_width_m,
            window_m=split_window_m,
            overlap_m=split_overlap_m,
        )
    else:
        score_hyps = list(all_hyps)
    if split_long and len(score_hyps) != len(all_hyps):
        log.info(
            "planarize: expanded %s peels → %s score windows (long-wall split)",
            len(all_hyps),
            len(score_hyps),
        )
    if extra:
        log.info(
            "planarize hybrid: ma=%s a=%s → windows=%s",
            len(ma_hyps),
            len(extra),
            len(score_hyps),
        )

    accepted: list[dict[str, Any]] = []
    best_reject = -1.0  # SENTINEL — not a measured ZNCC until a finite score lands
    n_scored = 0
    n_finite = 0
    skip = {"tilt": 0, "xy40": 0, "depth": 0, "views": 0, "load": 0, "soft_depth": 0}

    for hyp in score_hyps:
        n0 = np.asarray(hyp["n"], dtype=np.float64)
        d0 = float(hyp["d"])
        center0 = np.asarray(hyp["center"], dtype=np.float64)
        if abs(float(n0 @ UP)) > 0.20:
            skip["tilt"] += 1
            continue
        if float(np.linalg.norm(cams[:, :2] - center0[:2], axis=1).min()) > 40.0:
            skip["xy40"] += 1
            continue
        # Hard cam-distance only (extreme). Soft [min,max]_cam_depth applied per
        # refine candidate so ±n can rescue peels that were just outside band.
        nearest_eucl0 = float(np.linalg.norm(cams - center0, axis=1).min())
        if nearest_eucl0 < hard_min_cam_dist_m or nearest_eucl0 > hard_max_cam_dist_m:
            skip["depth"] += 1
            continue

        # Seed normal facing nearest cam (ref should see the front of the wall).
        n0, d0 = _flip_n_toward_cams(n0, d0, cams, center0)

        w = min(float(hyp["width_m"]), DEFAULT_SCORE_CLAMP_M)
        h = min(float(hyp["height_m"]), DEFAULT_SCORE_CLAMP_M)
        patch_i = patch if max(w, h) <= DEFAULT_SCORE_CLAMP_M else 96

        candidates: list[tuple[np.ndarray, float, np.ndarray]] = [(n0, d0, center0)]
        if refine:
            for delta in refine_deltas_m:
                c2 = center0 + n0 * float(delta)
                d2 = float(-n0 @ c2)
                candidates.append((n0, d2, c2))
            # Opposite facing — re-pick views below; H invariant for same views
            # but picker / frontal set can differ after flip.
            n_opp, d_opp = -n0, -d0
            candidates.append((n_opp, d_opp, center0))
            for delta in refine_deltas_m:
                c2 = center0 + n_opp * float(delta)
                d2 = float(-n_opp @ c2)
                candidates.append((n_opp, d2, c2))

        best_local: dict[str, Any] | None = None
        best_delta = 0.0
        hyp_had_views = False
        hyp_had_load = False
        hyp_scored_any = False
        for n_c, d_c, c_c in candidates:
            # Soft depth on *this* candidate (post-refine center).
            nearest_eucl = float(np.linalg.norm(cams - c_c, axis=1).min())
            if nearest_eucl < min_cam_depth_m or nearest_eucl > max_cam_depth_m:
                continue

            idxs = pp._pick_scoring_views(
                frames, n_c, c_c, max_views=4, min_frontal=0.25
            )
            if len(idxs) < 2:
                continue
            hyp_had_views = True
            views = []
            for i in idxs:
                v = get_view(i)
                if v is None:
                    break
                views.append(v)
            if len(views) < 2:
                continue
            hyp_had_load = True

            # Ensure normal faces the reference camera (views[0]).
            ref = views[0]
            C_ref = -ref.Rcw.T @ ref.t
            n_use, d_use = _flip_n_toward(n_c, d_c, C_ref)

            corners = hyp.get("corners")
            # Only pass corners when scoring the un-refined seed (same center
            # footprint); refined centers rebuild via sample_plane_quad_world.
            use_corners = None
            if (
                corners is not None
                and float(np.linalg.norm(c_c - center0)) < 1e-6
                and abs(float(n_use @ n0)) > 0.99
            ):
                use_corners = np.asarray(corners, dtype=np.float64)

            result = pp.score_vertical_plane(
                views,
                n_use,
                d_use,
                c_c,
                width_m=w,
                height_m=h,
                patch=patch_i,
                zncc_accept=zncc_accept,
                corners=use_corners,
            )
            hyp_scored_any = True
            n_scored += 1
            z = result.get("zncc")
            if isinstance(z, float) and not math.isnan(z):
                n_finite += 1
                best_reject = max(best_reject, z)
            if not result.get("ok"):
                continue
            if best_local is None or float(result["zncc"]) > float(best_local["zncc"]):
                best_delta = float(n_use @ (c_c - center0))
                best_local = {
                    **hyp,
                    **result,
                    "n": n_use,
                    "d": d_use,
                    "center": c_c,
                    "width_m": float(hyp["width_m"]),
                    "height_m": float(hyp["height_m"]),
                    "source": hyp.get("source") or "ma_segment",
                    "view_indices": idxs[: len(views)],
                    "refine_delta_m": best_delta,
                }

        if not hyp_scored_any:
            # Classify why this hyp never reached score_vertical_plane.
            if not hyp_had_views:
                # Either soft-depth filtered all candidates or picker empty.
                # Prefer soft_depth when seed was near band edge.
                if nearest_eucl0 < min_cam_depth_m or nearest_eucl0 > max_cam_depth_m:
                    skip["soft_depth"] += 1
                else:
                    skip["views"] += 1
            elif not hyp_had_load:
                skip["load"] += 1
            else:
                skip["soft_depth"] += 1
            continue

        if best_local is not None:
            accepted.append(best_local)

    accepted.sort(key=lambda p: -float(p.get("zncc") or 0.0))
    kept: list[dict[str, Any]] = []
    for pl in accepted:
        n = pl["n"]
        d = pl["d"]
        dup = False
        for k in kept:
            if abs(float(n @ k["n"])) < 0.85:
                continue
            if abs(float(d - k["d"])) < 2.5 or abs(float(d + k["d"])) < 2.5:
                if float(np.linalg.norm(pl["center"][:2] - k["center"][:2])) < 6.0:
                    dup = True
                    break
        if not dup:
            kept.append(pl)
        if len(kept) >= max_keep:
            break

    def _is_ma_source(src: str | None) -> bool:
        s = (src or "ma_segment").lower()
        return s.startswith("ma_") or s in {"ma_segment", "mapanything_planarize"}

    def _is_a_source(src: str | None) -> bool:
        s = (src or "").lower()
        return s in {"heading_distance", "manhattan", "sparse"} or s.startswith(
            "photo_"
        )

    n_pre_nms = len(accepted)
    ma_kept = sum(1 for p in kept if _is_ma_source(p.get("source")))
    a_kept = sum(1 for p in kept if _is_a_source(p.get("source")))

    if not kept:
        sentinel = n_scored == 0 or n_finite == 0
        tag = "SENTINEL" if sentinel else "measured"
        log.error(
            "planarize: FAIL-LOUD — 0 / %s hyps passed ZNCC≥%.2f "
            "(best rejected≈%.3f %s; scored=%s finite=%s; "
            "skips tilt=%s xy40=%s depth=%s soft_depth=%s views=%s load=%s; "
            "windows=%s; pre_nms=%s; ma=%s a=%s). Not inventing RANSAC/OSM walls.",
            len(all_hyps),
            zncc_accept,
            best_reject,
            tag,
            n_scored,
            n_finite,
            skip["tilt"],
            skip["xy40"],
            skip["depth"],
            skip["soft_depth"],
            skip["views"],
            skip["load"],
            len(score_hyps),
            n_pre_nms,
            len(ma_hyps),
            len(extra),
        )
    else:
        mean_z = float(np.mean([float(p["zncc"]) for p in kept]))
        if mean_z < 0.42:
            log.warning("planarize: mean ZNCC of kept=%.3f < 0.42 (soft warn)", mean_z)
        log.info(
            "planarize: kept %s / %s hyps (ZNCC≥%.2f, mean=%.3f, scored=%s, "
            "windows=%s, pre_nms=%s, ma_kept=%s a_kept=%s)",
            len(kept),
            len(all_hyps),
            zncc_accept,
            mean_z,
            n_scored,
            len(score_hyps),
            n_pre_nms,
            ma_kept,
            a_kept,
        )
    return kept



def write_planes_json(
    path: Path,
    *,
    planes: list[dict[str, Any]],
    ground_z: float,
    residual_points: int,
    source: str = "mapanything_planarize",
    textured_maps: list[str | None] | None = None,
) -> None:
    """Write recon/planes.json for Studio / debug."""
    import json

    entries = []
    for i, pl in enumerate(planes):
        n = np.asarray(pl.get("n") if "n" in pl else [pl["nx"], pl["ny"], 0.0])
        if "d" in pl and "nx" not in pl:
            d_plane = float(pl["d"])
        else:
            # facades dict: nx*e+ny*n = d_xy → n·X + d = 0 with d = -d_xy
            d_plane = float(-pl["d"]) if "nx" in pl else float(pl.get("d", 0.0))
        quad = pl.get("quad") or pl.get("corners")
        tex = None
        if textured_maps and i < len(textured_maps):
            tex = textured_maps[i]
        entries.append(
            {
                "id": f"facade_{i:02d}",
                "n": [float(n[0]), float(n[1]), float(n[2]) if len(n) > 2 else 0.0],
                "d": d_plane,
                "quad": [list(map(float, c)) for c in quad] if quad is not None else None,
                "width_m": float(pl.get("width_m") or 0.0),
                "height_m": float(pl.get("height_m") or 0.0),
                "zncc": float(pl["zncc"]) if pl.get("zncc") is not None else None,
                "texture": tex,
                "inliers": int(pl.get("count") or pl.get("inliers") or 0),
                "source": pl.get("source") or "ma_segment",
            }
        )
    znccs = [e["zncc"] for e in entries if e["zncc"] is not None]
    payload = {
        "frame": "ENU",
        "source": source,
        "ground_z": float(ground_z),
        "planes": entries,
        "residual_points": int(residual_points),
        "gates": {
            "mean_zncc": float(np.mean(znccs)) if znccs else None,
            "plane_count": len(entries),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
