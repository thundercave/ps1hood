"""Vertical façade planes from multi-view photo-consistency (ZNCC).

Hypothesize vertical planes (sparse seeds + Manhattan heading×distance),
score with plane-induced homography + ZNCC across cross-pano views, keep
only acceptances. Do not invent flow-RANSAC walls as product geometry.

Plane form: n·X + d = 0 with n_u ≈ 0 (vertical), ||n||=1.
Poses: OpenCV world→camera x = Rcw @ X + t (same as unproject._Rt).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ps1_hood.geo import camera_rotation_cv, heading_diff, wrap_heading

log = logging.getLogger(__name__)

# Default accept threshold from research sketch / Milestone A.
DEFAULT_ZNCC_ACCEPT = 0.35


@dataclass
class View:
    image_bgr: np.ndarray  # HxWx3 uint8
    K: np.ndarray  # 3x3
    Rcw: np.ndarray  # 3x3 world→camera
    t: np.ndarray  # (3,)
    frame_index: int = -1
    pano_id: str | None = None


def K_from_frame(width: int, height: int, fov_deg: float) -> np.ndarray:
    f = 0.5 * width / math.tan(math.radians(fov_deg) / 2.0)
    return np.array(
        [[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def Rt_from_frame(frame: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    R_wc = np.array(
        camera_rotation_cv(float(frame["heading"]), float(frame.get("pitch") or 0.0)),
        dtype=np.float64,
    )
    C = np.array(
        [float(frame["e"]), float(frame["n"]), float(frame["u"])],
        dtype=np.float64,
    )
    Rcw = R_wc.T
    t = -Rcw @ C
    return Rcw, t


def P_from_Rt(K: np.ndarray, Rcw: np.ndarray, t: np.ndarray) -> np.ndarray:
    return K @ np.hstack([Rcw, t.reshape(3, 1)])


def vertical_plane_from_point_heading(
    p0: np.ndarray, heading_deg: float
) -> tuple[np.ndarray, float]:
    """Vertical plane with outward normal = SV heading (0=+N, 90=+E).

    Plane: n·X + d = 0, n_u = 0.
    """
    h = math.radians(heading_deg)
    # camera_rotation_cv forward xy = (sin h, cos h)
    n = np.array([math.sin(h), math.cos(h), 0.0], dtype=np.float64)
    n /= np.linalg.norm(n) + 1e-12
    d = float(-n @ p0.astype(np.float64))
    return n, d


def plane_homography(
    K_ref: np.ndarray,
    R_ref: np.ndarray,
    t_ref: np.ndarray,
    K_src: np.ndarray,
    R_src: np.ndarray,
    t_src: np.ndarray,
    n: np.ndarray,
    d: float,
) -> np.ndarray:
    """H mapping ref pixels → src pixels for plane n·X + d = 0 (world).

    With n_ref·x_ref = c, Euclidean transfer is
    X_src = (R_rel + t_rel n_refᵀ / c) X_ref, so
    H = K_src @ (R_rel + outer(t_rel, n_ref) / c) @ inv(K_ref).

    (Scrapy sketch used a minus sign; verified against projected plane points.)
    """
    R_rel = R_src @ R_ref.T
    t_rel = t_src - R_rel @ t_ref
    n_ref = R_ref @ n
    c = float(n @ (R_ref.T @ t_ref) - d)
    if abs(c) < 1e-8:
        raise ValueError("plane through/near reference camera")
    return K_src @ (R_rel + np.outer(t_rel, n_ref) / c) @ np.linalg.inv(K_ref)


def zncc(a: np.ndarray, b: np.ndarray) -> float:
    """ZNCC on two same-shape float grayscale patches. Returns [-1,1] or nan."""
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    if a.size < 16:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-6:
        return float("nan")
    return float(np.dot(a, b) / denom)


def sample_plane_quad_world(
    n: np.ndarray,
    d: float,
    center: np.ndarray,
    width_m: float,
    height_m: float,
) -> np.ndarray:
    """4 corners of a vertical rectangle on the plane (BL, BR, TR, TL)."""
    c = center.astype(np.float64)
    c = c - (n @ c + d) * n
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(up, n)
    rn = np.linalg.norm(right)
    if rn < 1e-6:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        right /= rn
    up = np.cross(n, right)
    up /= np.linalg.norm(up) + 1e-12
    hw, hh = width_m / 2.0, height_m / 2.0
    return np.stack(
        [
            c - hw * right - hh * up,
            c + hw * right - hh * up,
            c + hw * right + hh * up,
            c - hw * right + hh * up,
        ],
        axis=0,
    )


def project_points(P: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hom = np.hstack([X, np.ones((len(X), 1))])
    x = (P @ hom.T).T
    front = x[:, 2] > 1e-6
    uv = x[:, :2] / np.clip(x[:, 2:3], 1e-6, None)
    return uv, front


def _pano_key(frame: dict[str, Any]) -> str:
    if frame.get("pano_id"):
        return str(frame["pano_id"])
    return f"{round(float(frame.get('e', 0)), 2)}_{round(float(frame.get('n', 0)), 2)}"


def load_view(frame: dict[str, Any], frame_index: int = -1) -> View | None:
    path = frame.get("path") or frame.get("shot_path")
    if not path:
        return None
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    K = K_from_frame(w, h, float(frame.get("fov") or 90.0))
    Rcw, t = Rt_from_frame(frame)
    return View(
        image_bgr=img,
        K=K,
        Rcw=Rcw,
        t=t,
        frame_index=frame_index,
        pano_id=str(frame["pano_id"]) if frame.get("pano_id") else None,
    )


def score_vertical_plane(
    views: list[View],
    n: np.ndarray,
    d: float,
    center: np.ndarray,
    *,
    width_m: float = 8.0,
    height_m: float = 9.0,
    patch: int = 64,
    zncc_accept: float = DEFAULT_ZNCC_ACCEPT,
    min_views: int = 2,
) -> dict[str, Any]:
    """Mean ZNCC of sources warped into ref ortho; accept if mean >= threshold.

    ``views[0]`` is the reference; remaining are cross-pano sources.
    """
    if len(views) < 2:
        return {"ok": False, "reason": "need >=2 views", "zncc": float("nan")}

    ref = views[0]
    corners = sample_plane_quad_world(n, d, center, width_m, height_m)
    Pref = P_from_Rt(ref.K, ref.Rcw, ref.t)
    uv_ref, front = project_points(Pref, corners)
    if not bool(front.all()):
        return {"ok": False, "reason": "quad behind ref camera", "zncc": float("nan")}

    h, w = ref.image_bgr.shape[:2]
    partial = bool(
        (uv_ref[:, 0] < 0).any()
        or (uv_ref[:, 0] > w - 1).any()
        or (uv_ref[:, 1] < 0).any()
        or (uv_ref[:, 1] > h - 1).any()
    )

    src = uv_ref.astype(np.float32)
    dst = np.array(
        [[0, patch - 1], [patch - 1, patch - 1], [patch - 1, 0], [0, 0]],
        dtype=np.float32,
    )
    H_ref = cv2.getPerspectiveTransform(src, dst)
    ref_gray = cv2.cvtColor(ref.image_bgr, cv2.COLOR_BGR2GRAY)
    tmpl = cv2.warpPerspective(ref_gray, H_ref, (patch, patch), flags=cv2.INTER_LINEAR)

    scores: list[float] = []
    for src_view in views[1:]:
        try:
            H = plane_homography(
                ref.K, ref.Rcw, ref.t, src_view.K, src_view.Rcw, src_view.t, n, d
            )
            H_src_to_ortho = H_ref @ np.linalg.inv(H)
        except (ValueError, np.linalg.LinAlgError) as exc:
            continue
        src_gray = cv2.cvtColor(src_view.image_bgr, cv2.COLOR_BGR2GRAY)
        warped = cv2.warpPerspective(
            src_gray, H_src_to_ortho, (patch, patch), flags=cv2.INTER_LINEAR
        )
        if float((warped > 0).mean()) < 0.4:
            continue
        s = zncc(tmpl, warped)
        if not math.isnan(s):
            scores.append(s)

    if len(scores) < max(1, min_views - 1):
        return {
            "ok": False,
            "reason": "too few valid source views",
            "zncc": float("nan"),
            "scores": scores,
        }

    mean = float(np.mean(scores))
    return {
        "ok": mean >= zncc_accept,
        "zncc": mean,
        "scores": scores,
        "partial_ref_quad": partial,
        "threshold": zncc_accept,
        "reason": "accept" if mean >= zncc_accept else "zncc below threshold",
        "corners": corners,
        "quad_center": corners.mean(axis=0),
    }


def hypothesize_vertical_planes(
    frames: list[dict[str, Any]],
    xyz: np.ndarray | None = None,
    *,
    distances_m: tuple[float, ...] = (6.0, 10.0, 14.0, 18.0, 22.0),
    width_m: float = 8.0,
    height_m: float = 9.0,
    ground_z: float = 0.0,
    max_heading_seeds: int = 24,
    max_sparse_seeds: int = 16,
) -> list[dict[str, Any]]:
    """Seed vertical planes from camera heading×distance and optional sparse points.

    Does **not** use flow-cloud RANSAC as product geometry — only hypotheses.
    """
    hyps: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int]] = set()

    def _add(n: np.ndarray, d: float, center: np.ndarray, source: str) -> None:
        key = (
            int(round(math.degrees(math.atan2(float(n[0]), float(n[1]))) / 5.0)),
            int(round(d * 2.0)),
            int(round(float(center[0]) + float(center[1]))),
        )
        if key in seen:
            return
        seen.add(key)
        hyps.append(
            {
                "n": n.astype(np.float64),
                "d": float(d),
                "center": center.astype(np.float64),
                "width_m": float(width_m),
                "height_m": float(height_m),
                "source": source,
            }
        )

    # --- heading × distance along drive (prefer façade-facing views) ---
    # Subsample frames: one per pano, prefer headings ~±90° from travel if known.
    by_pano: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for i, fr in enumerate(frames):
        by_pano.setdefault(_pano_key(fr), []).append((i, fr))

    seed_frames: list[dict[str, Any]] = []
    for group in by_pano.values():
        # Prefer views whose heading is not the travel direction when travel known.
        preferred = []
        for _, fr in group:
            travel = fr.get("travel_heading")
            if travel is None:
                preferred.append(fr)
                continue
            sep = abs(heading_diff(float(fr["heading"]), float(travel)))
            # Sideways: ~70–110° from travel
            if 70.0 <= sep <= 110.0:
                preferred.append(fr)
        chosen = preferred if preferred else [group[0][1]]
        # At most two sideways views per pano
        seed_frames.extend(chosen[:2])

    if len(seed_frames) > max_heading_seeds:
        step = max(1, len(seed_frames) // max_heading_seeds)
        seed_frames = seed_frames[::step][:max_heading_seeds]

    wall_u = ground_z + height_m * 0.45
    for fr in seed_frames:
        C = np.array(
            [float(fr["e"]), float(fr["n"]), float(fr.get("u") or wall_u)],
            dtype=np.float64,
        )
        heading = float(fr["heading"])
        fwd = np.array(
            [math.sin(math.radians(heading)), math.cos(math.radians(heading)), 0.0],
            dtype=np.float64,
        )
        for dist in distances_m:
            p0 = np.array(
                [C[0] + fwd[0] * dist, C[1] + fwd[1] * dist, wall_u],
                dtype=np.float64,
            )
            n, d = vertical_plane_from_point_heading(p0, heading)
            _add(n, d, p0, "heading_distance")

    # Manhattan partners: same centers with heading+90 (street-parallel walls)
    for fr in seed_frames[:: max(1, len(seed_frames) // 8 or 1)][:8]:
        C = np.array([float(fr["e"]), float(fr["n"]), wall_u], dtype=np.float64)
        for yaw_off in (90.0, -90.0):
            heading = wrap_heading(float(fr["heading"]) + yaw_off)
            fwd = np.array(
                [math.sin(math.radians(heading)), math.cos(math.radians(heading)), 0.0],
                dtype=np.float64,
            )
            for dist in (10.0, 16.0):
                p0 = C + fwd * dist
                p0[2] = wall_u
                n, d = vertical_plane_from_point_heading(p0, heading)
                _add(n, d, p0, "manhattan")

    # --- sparse posed points as seeds (optional) ---
    if xyz is not None and len(xyz) >= 30:
        above = xyz[xyz[:, 2] > (ground_z + 0.5)]
        if len(above) >= 20:
            rng = np.random.default_rng(0)
            # Lightweight vertical-line seeds → plane through point with PCA tangent
            n_try = min(max_sparse_seeds * 8, len(above))
            idx = rng.choice(len(above), size=n_try, replace=False)
            for i in idx:
                p = above[int(i)]
                # Neighbor PCA in XY for tangent → normal
                dif = above[:, :2] - p[:2]
                near = np.linalg.norm(dif, axis=1) < 3.0
                if int(near.sum()) < 8:
                    continue
                pts = above[near, :2]
                cov = np.cov((pts - pts.mean(axis=0)).T)
                evals, evecs = np.linalg.eigh(cov)
                tangent = evecs[:, int(np.argmax(evals))]
                n_xy = np.array([-tangent[1], tangent[0]], dtype=np.float64)
                nn = np.linalg.norm(n_xy)
                if nn < 1e-6:
                    continue
                n_xy /= nn
                n = np.array([n_xy[0], n_xy[1], 0.0], dtype=np.float64)
                d = float(-n @ p)
                center = np.array([p[0], p[1], wall_u], dtype=np.float64)
                _add(n, d, center, "sparse")
                if sum(1 for h in hyps if h["source"] == "sparse") >= max_sparse_seeds:
                    break

    log.info(
        "photo-planes: %s hypotheses (heading/manhattan/sparse)",
        len(hyps),
    )
    return hyps


def _pick_scoring_views(
    frames: list[dict[str, Any]],
    n: np.ndarray,
    center: np.ndarray,
    *,
    max_views: int = 4,
    min_frontal: float = 0.25,
) -> list[int]:
    """Pick up to max_views frame indices (cross-pano) that see the plane frontally."""
    n_xy = n[:2] / (np.linalg.norm(n[:2]) + 1e-12)
    scored: list[tuple[float, int]] = []
    for i, fr in enumerate(frames):
        C = np.array([float(fr["e"]), float(fr["n"]), float(fr["u"])], dtype=np.float64)
        to_cam = C[:2] - center[:2]
        # Orient normal toward camera
        n_use = n_xy.copy()
        if float(n_use @ to_cam) < 0:
            n_use = -n_use
        R_wc = np.array(
            camera_rotation_cv(float(fr["heading"]), float(fr.get("pitch") or 0.0)),
            dtype=np.float64,
        )
        fwd = R_wc[:2, 2]
        fn = float(np.linalg.norm(fwd))
        if fn < 1e-6:
            continue
        fwd = fwd / fn
        frontal = float((-fwd) @ n_use)
        if frontal < min_frontal:
            continue
        dist = float(np.linalg.norm(C - center))
        if dist < 2.0 or dist > 35.0:
            continue
        scored.append((frontal / (1.0 + 0.03 * dist), i))

    scored.sort(key=lambda t: -t[0])
    # Cross-pano only among selected
    picked: list[int] = []
    panos: set[str] = set()
    for _, i in scored:
        pk = _pano_key(frames[i])
        if pk in panos:
            continue
        panos.add(pk)
        picked.append(i)
        if len(picked) >= max_views:
            break
    return picked


def search_photo_consistent_planes(
    frames: list[dict[str, Any]],
    xyz: np.ndarray | None = None,
    *,
    zncc_accept: float = DEFAULT_ZNCC_ACCEPT,
    ground_z: float = 0.0,
    width_m: float = 8.0,
    height_m: float = 9.0,
    patch: int = 64,
    max_keep: int = 12,
    refine: bool = True,
) -> list[dict[str, Any]]:
    """Hypothesize → score → keep high-ZNCC planes only (fail-loud → empty list)."""
    if len(frames) < 2:
        log.error(
            "photo-consistency façades: need ≥2 posed frames; got %s — "
            "writing empty façades (not inventing RANSAC walls)",
            len(frames),
        )
        return []

    hyps = hypothesize_vertical_planes(
        frames,
        xyz,
        width_m=width_m,
        height_m=height_m,
        ground_z=ground_z,
    )
    if not hyps:
        log.error(
            "photo-consistency façades: no plane hypotheses — "
            "writing empty façades (not inventing RANSAC walls)"
        )
        return []

    # Cache loaded views by frame index
    view_cache: dict[int, View | None] = {}

    def get_view(i: int) -> View | None:
        if i not in view_cache:
            view_cache[i] = load_view(frames[i], i)
        return view_cache[i]

    accepted: list[dict[str, Any]] = []
    best_reject = -1.0
    n_scored = 0

    for hyp in hyps:
        n = hyp["n"]
        d = hyp["d"]
        center = hyp["center"]
        idxs = _pick_scoring_views(frames, n, center, max_views=4)
        if len(idxs) < 2:
            continue
        views: list[View] = []
        for i in idxs:
            v = get_view(i)
            if v is None:
                break
            views.append(v)
        if len(views) < 2:
            continue

        # Optional 1D distance refine around seed
        candidates = [(n, d, center)]
        if refine and hyp["source"] in {"heading_distance", "manhattan"}:
            # n points from camera toward wall; move plane along normal
            for delta in (-2.0, -1.0, 1.0, 2.0):
                c2 = center + n * delta
                d2 = float(-n @ c2)
                candidates.append((n, d2, c2))

        best_local: dict[str, Any] | None = None
        for n_c, d_c, c_c in candidates:
            result = score_vertical_plane(
                views,
                n_c,
                d_c,
                c_c,
                width_m=hyp["width_m"],
                height_m=hyp["height_m"],
                patch=patch,
                zncc_accept=zncc_accept,
            )
            n_scored += 1
            z = result.get("zncc")
            if isinstance(z, float) and not math.isnan(z):
                best_reject = max(best_reject, z)
            if not result.get("ok"):
                continue
            if best_local is None or float(result["zncc"]) > float(best_local["zncc"]):
                best_local = {
                    **result,
                    "n": n_c,
                    "d": d_c,
                    "center": c_c,
                    "width_m": hyp["width_m"],
                    "height_m": hyp["height_m"],
                    "source": hyp["source"],
                    "view_indices": idxs[: len(views)],
                }
        if best_local is not None:
            accepted.append(best_local)

    # Non-maximum suppression: similar normal + d
    accepted.sort(key=lambda p: -float(p["zncc"]))
    kept: list[dict[str, Any]] = []
    for pl in accepted:
        n = pl["n"]
        d = pl["d"]
        dup = False
        for k in kept:
            if abs(float(n @ k["n"])) < 0.85:
                continue
            # Same orientation family: compare plane offset
            if abs(float(d - k["d"])) < 2.5 or abs(float(d + k["d"])) < 2.5:
                # also require centers close in plane span
                if float(np.linalg.norm(pl["center"][:2] - k["center"][:2])) < 6.0:
                    dup = True
                    break
        if not dup:
            kept.append(pl)
        if len(kept) >= max_keep:
            break

    if not kept:
        log.error(
            "photo-consistency façades: FAIL-LOUD — 0 / %s hypotheses passed "
            "ZNCC≥%.2f (best rejected≈%.3f; scored=%s). "
            "Writing empty façades; not inventing flow-RANSAC walls as product geometry.",
            len(hyps),
            zncc_accept,
            best_reject,
            n_scored,
        )
    else:
        log.info(
            "photo-consistency façades: kept %s planes (scored %s / %s hyps, min ZNCC≥%.2f)",
            len(kept),
            n_scored,
            len(hyps),
            zncc_accept,
        )
    return kept


def plane_dict_for_obj(pl: dict[str, Any], ground_z: float) -> dict[str, Any]:
    """Convert accepted plane to facades.py-style dict (nx,ny,d_xy,min,max)."""
    n = np.asarray(pl["n"], dtype=np.float64)
    # facades._plane_quad uses nx*e + ny*n = d_xy  (n·xy = d)
    # our form: n·X + d = 0 → n_xy·xy = -d
    d_xy = float(-pl["d"])
    # Ensure normal points consistently
    nx, ny = float(n[0]), float(n[1])
    corners = pl.get("corners")
    if corners is None:
        corners = sample_plane_quad_world(
            n, float(pl["d"]), pl["center"], float(pl["width_m"]), float(pl["height_m"])
        )
    corners = np.asarray(corners, dtype=np.float64)
    return {
        "nx": nx,
        "ny": ny,
        "d": d_xy,
        "min": corners.min(axis=0).tolist(),
        "max": corners.max(axis=0).tolist(),
        "count": 0,
        "ground_z": float(ground_z),
        "zncc": float(pl.get("zncc") or 0.0),
        "source": pl.get("source") or "photo_consistency",
        "quad": [tuple(map(float, c)) for c in corners],
    }
