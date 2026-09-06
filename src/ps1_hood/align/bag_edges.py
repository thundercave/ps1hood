"""Snap Street View cameras to 3DBAG with edges after the GPS prior.

Metadata (GPS/heading) is the coarse pose. 3DBAG wall/roof edges are
projected into the camera and matched to Canny edges in the photo via a
distance transform. Search is metres + degrees around the prior so the
result can land around 10 cm when the façade is visible.

BAG itself is not 10 cm accurate on balconies, so the residual is reported.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from ps1_hood.align.seat import is_inside_footprint
from ps1_hood.geo import camera_rotation_cv, wrap_heading


def _K(width: int, height: int, fov_deg: float) -> np.ndarray:
    f = 0.5 * width / math.tan(math.radians(fov_deg) / 2.0)
    return np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1.0]], dtype=np.float64)


def photo_edge_dt(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 60, 160)
    inv = np.where(edges > 0, 0, 255).astype(np.uint8)
    return cv2.distanceTransform(inv, cv2.DIST_L2, 3)


def collect_bag_edges(
    buildings: list[dict[str, Any]],
    center: tuple[float, float],
    radius_m: float = 45.0,
    max_segs: int = 160,
) -> np.ndarray:
    """Nx6 array of 3D edge segments (x0,y0,z0,x1,y1,z1) near the camera."""
    segs: list[list[float]] = []
    ce, cn = center
    for bld in buildings:
        verts = bld["vertices"]
        be, bn = 0.5 * (bld["bbox"][0] + bld["bbox"][3]), 0.5 * (bld["bbox"][1] + bld["bbox"][4])
        if math.hypot(be - ce, bn - cn) > radius_m + 20:
            continue
        for i0, i1 in bld["edges"]:
            a, b = verts[i0], verts[i1]
            length = math.dist(a, b)
            if length < 0.4 or length > 40:
                continue
            dz = abs(a[2] - b[2])
            zmid = 0.5 * (a[2] + b[2])
            # skip kerb/ground rings — they match road paint, not façades
            if dz < 0.4 and zmid < 1.2:
                continue
            segs.append([a[0], a[1], a[2], b[0], b[1], b[2]])
    if not segs:
        return np.zeros((0, 6), dtype=np.float64)
    arr = np.asarray(segs, dtype=np.float64)
    if len(arr) > max_segs:
        mid = 0.5 * (arr[:, :2] + arr[:, 3:5])
        dist = np.hypot(mid[:, 0] - ce, mid[:, 1] - cn)
        dz = np.abs(arr[:, 5] - arr[:, 2])
        # closer, and slightly prefer façade-ish verticals
        order = dist - 0.35 * dz
        arr = arr[np.argpartition(order, max_segs)[:max_segs]]
    return arr


def _project_segments(
    segs: np.ndarray,
    *,
    e: float,
    n: float,
    u: float,
    heading: float,
    pitch: float,
    fov: float,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Project 3D segments to pixel lines. Returns (N,4) xyxy and keep mask."""
    if len(segs) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=bool)
    R = np.array(camera_rotation_cv(heading, pitch), dtype=np.float64)
    Rcw = R.T
    C = np.array([e, n, u], dtype=np.float64)
    K = _K(width, height, fov)
    pts = segs.reshape(-1, 3)
    cam = (Rcw @ (pts - C).T).T
    z = cam[:, 2]
    valid = z > 0.35
    uv = np.zeros((len(cam), 2), dtype=np.float64)
    uv[valid, 0] = K[0, 0] * cam[valid, 0] / z[valid] + K[0, 2]
    uv[valid, 1] = K[1, 1] * cam[valid, 1] / z[valid] + K[1, 2]
    p0, p1 = uv[0::2], uv[1::2]
    keep = valid[0::2] & valid[1::2]
    keep &= (
        ((p0[:, 0] >= -40) | (p1[:, 0] >= -40))
        & ((p0[:, 0] < width + 40) | (p1[:, 0] < width + 40))
        & ((p0[:, 1] >= -40) | (p1[:, 1] >= -40))
        & ((p0[:, 1] < height + 40) | (p1[:, 1] < height + 40))
    )
    lines = np.column_stack([p0, p1]).astype(np.float32)
    return lines, keep


def score_pose(
    dt: np.ndarray,
    segs: np.ndarray,
    *,
    e: float,
    n: float,
    u: float,
    heading: float,
    pitch: float,
    fov: float,
) -> float:
    h, w = dt.shape
    lines, keep = _project_segments(
        segs, e=e, n=n, u=u, heading=heading, pitch=pitch, fov=fov, width=w, height=h
    )
    kept = lines[keep]
    if len(kept) < 4:
        return 1e9
    t = np.linspace(0.0, 1.0, 8, dtype=np.float32)[None, :, None]
    pts = kept[:, None, :2] * (1.0 - t) + kept[:, None, 2:] * t
    xs = np.rint(pts[:, :, 0]).astype(np.int32)
    ys = np.rint(pts[:, :, 1]).astype(np.int32)
    inside = (xs >= 0) & (xs < w - 1) & (ys >= 0) & (ys < h - 1)
    n_inside = inside.sum(axis=1)
    if int((n_inside > 0).sum()) < 4:
        return 1e9
    vals = np.zeros(pts.shape[:2], dtype=np.float32)
    vals[inside] = dt[ys[inside], xs[inside]]
    line_sum = vals.sum(axis=1)
    good = n_inside > 0
    means = line_sum[good] / n_inside[good]
    return float(means.mean())


def snap_camera_to_bag(
    photo_bgr: np.ndarray,
    buildings: list[dict[str, Any]],
    *,
    e: float,
    n: float,
    u: float,
    heading: float,
    pitch: float,
    fov: float,
    travel_heading: float | None = None,
    footprints: list[Any] | None = None,
) -> dict[str, Any]:
    """Coarse-to-fine edge match. Stays on the street (along-track search)."""
    dt = photo_edge_dt(photo_bgr)
    # work at reduced width for speed
    scale = 480.0 / max(dt.shape[1], 1)
    if scale < 0.99:
        dt = cv2.resize(dt, (int(dt.shape[1] * scale), int(dt.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        dt *= scale
    segs = collect_bag_edges(buildings, (e, n), radius_m=45.0)
    if len(segs) < 6:
        return {
            "e": e,
            "n": n,
            "u": u,
            "heading": heading,
            "score": None,
            "residual_m": None,
            "snapped": False,
        }

    th = float(travel_heading if travel_heading is not None else heading)
    along = (math.sin(math.radians(th)), math.cos(math.radians(th)))
    across = (math.cos(math.radians(th)), -math.sin(math.radians(th)))

    best = {"score": score_pose(dt, segs, e=e, n=n, u=u, heading=heading, pitch=pitch, fov=fov),
            "e": e, "n": n, "u": u, "heading": heading}

    def consider(e2: float, n2: float, u2: float, h2: float) -> None:
        if footprints is not None and is_inside_footprint(e2, n2, footprints, margin_m=1.3):
            return
        score = score_pose(dt, segs, e=e2, n=n2, u=u2, heading=h2, pitch=pitch, fov=fov)
        if score < best["score"]:
            best["score"] = score
            best["e"] = e2
            best["n"] = n2
            best["u"] = u2
            best["heading"] = h2

    def search(span_a: float, step_a: float, span_c: float, step_c: float, span_z: float, step_z: float, span_h: float, step_h: float) -> None:
        ce, cn, cu, ch = best["e"], best["n"], best["u"], best["heading"]
        for da in np.arange(-span_a, span_a + 1e-9, step_a):
            for dc in np.arange(-span_c, span_c + 1e-9, step_c):
                e2 = ce + along[0] * float(da) + across[0] * float(dc)
                n2 = cn + along[1] * float(da) + across[1] * float(dc)
                for du in np.arange(-span_z, span_z + 1e-9, step_z):
                    for dh in np.arange(-span_h, span_h + 1e-9, step_h):
                        consider(e2, n2, cu + float(du), wrap_heading(ch + float(dh)))

    # GPS already has along-street order. Only nudge a little along-track
    # or neighbouring panos walk into each other. Lateral (kerb) is the
    # uncertain axis.
    search(1.2, 0.4, 2.2, 0.55, 0.4, 0.4, 4.0, 4.0)
    search(0.4, 0.2, 0.8, 0.4, 0.2, 0.2, 2.0, 2.0)
    search(0.15, 0.05, 0.25, 0.08, 0.08, 0.08, 0.6, 0.3)

    shift = math.hypot(best["e"] - e, best["n"] - n)
    return {
        "e": best["e"],
        "n": best["n"],
        "u": best["u"],
        "heading": best["heading"],
        "score": float(best["score"]),
        "residual_m": round(shift, 3),
        "snapped": best["score"] < 12.0,
    }
