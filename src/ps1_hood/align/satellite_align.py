"""Nudge a camera in SE(2) so its ground plane agrees with the satellite.

Fuses photometric NCC with Ortho road/roof Canny edge agreement (Chamfer).
Sacred: Ortho = absolute XY; BAG edges are not used here.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from ps1_hood.align.sat_edges import edge_agree, ortho_edge_bgr
from ps1_hood.capture.satellite import Ortho
from ps1_hood.geo import camera_rotation_cv, wrap_heading


def render_satellite_into_camera_fast(
    ortho: Ortho,
    *,
    e: float,
    n: float,
    heading: float,
    pitch: float,
    height_m: float,
    width: int,
    height: int,
    fov_deg: float,
    image: np.ndarray | None = None,
) -> np.ndarray:
    """Vectorised ground-plane projection of Ortho (or a substitute ``image``)."""
    src = ortho.image if image is None else image
    fx = 0.5 * width / math.tan(math.radians(fov_deg) / 2.0)
    fy = fx
    cx, cy = width / 2.0, height / 2.0
    R = np.array(camera_rotation_cv(heading, pitch), dtype=np.float64)
    y0 = int(height * 0.45)
    vv, uu = np.mgrid[y0:height, 0:width]
    x = (uu - cx) / fx
    y = (vv - cy) / fy
    ones = np.ones_like(x)
    rays = np.stack([x, y, ones], axis=-1) @ R.T
    dz = rays[..., 2]
    valid = dz < -1e-5
    t = np.zeros_like(dz)
    t[valid] = -height_m / dz[valid]
    valid &= (t > 0) & (t < 80.0)
    ge = e + t * rays[..., 0]
    gn = n + t * rays[..., 1]
    pu = (ge - ortho.sw) / max(ortho.ee - ortho.sw, 1e-6) * (ortho.w - 1)
    pv = (1.0 - (gn - ortho.sh) / max(ortho.nn - ortho.sh, 1e-6)) * (ortho.h - 1)
    map_x = pu.astype(np.float32)
    map_y = pv.astype(np.float32)
    sampled = cv2.remap(
        src,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    mask = valid.astype(np.uint8) * 255
    out = np.zeros((height, width, 3), dtype=np.uint8)
    out[y0:height] = cv2.bitwise_and(sampled, sampled, mask=mask)
    return out


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    mask = (b.mean(axis=2) > 5) & (a.mean(axis=2) > 5)
    if int(mask.sum()) < 200:
        return -1.0
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    va = ga[mask]
    vb = gb[mask]
    va = va - va.mean()
    vb = vb - vb.mean()
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom < 1e-6:
        return -1.0
    return float(np.dot(va, vb) / denom)


def fuse_sat_score(
    ncc_v: float,
    edge_v: float,
    *,
    w_ncc: float = 0.35,
    w_edge: float = 0.65,
) -> float:
    """Weighted fuse; missing terms (``-1``) drop their weight and renormalise."""
    parts: list[tuple[float, float]] = []
    if ncc_v > -0.5:
        parts.append((w_ncc, ncc_v))
    if edge_v > -0.5:
        parts.append((w_edge, edge_v))
    if not parts:
        return -1.0
    wsum = sum(w for w, _ in parts)
    if wsum < 1e-9:
        return -1.0
    return float(sum(w * v for w, v in parts) / wsum)


def align_camera_to_satellite(
    photo_bgr: np.ndarray,
    ortho: Ortho,
    *,
    e: float,
    n: float,
    heading: float,
    pitch: float,
    height_m: float,
    fov_deg: float,
    max_shift_m: float = 8.0,
    max_heading_deg: float = 15.0,
    w_ncc: float = 0.35,
    w_edge: float = 0.65,
    ortho_edges_bgr: np.ndarray | None = None,
) -> dict:
    """Coarse-to-fine search over east/north/heading. Metadata is the prior.

    Score = ``w_ncc * ncc + w_edge * edge_agree`` (renormalised if a term fails).
    Returns ``score`` (fused), ``ncc``, ``edge``, plus pose fields.
    """
    h, w = photo_bgr.shape[:2]
    # work at reduced res — this search is the slow part
    scale = 320.0 / max(w, 1)
    small = cv2.resize(photo_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]

    edge_src = ortho_edges_bgr if ortho_edges_bgr is not None else ortho_edge_bgr(ortho)

    best = {
        "score": -1.0,
        "ncc": -1.0,
        "edge": -1.0,
        "e": e,
        "n": n,
        "heading": heading,
    }
    # Coarse → medium → fine (edges survive downsample; fine pins kerb/roofline)
    stages = (
        (2.5, 5.0, max_shift_m, max_heading_deg),
        (0.8, 2.0, 3.0, 6.0),
        (0.4, 1.0, 2.0, 4.0),
    )
    for step_m, step_h, span_m, span_h in stages:
        ce, cn, ch = best["e"], best["n"], best["heading"]
        de = np.arange(-span_m, span_m + 1e-6, step_m)
        dn = np.arange(-span_m, span_m + 1e-6, step_m)
        dh = np.arange(-span_h, span_h + 1e-6, step_h)
        for ee in de:
            for nn in dn:
                for hh in dh:
                    cand_e = ce + float(ee)
                    cand_n = cn + float(nn)
                    cand_h = wrap_heading(ch + float(hh))
                    synth = render_satellite_into_camera_fast(
                        ortho,
                        e=cand_e,
                        n=cand_n,
                        heading=cand_h,
                        pitch=pitch,
                        height_m=height_m,
                        width=sw,
                        height=sh,
                        fov_deg=fov_deg,
                    )
                    synth_e = render_satellite_into_camera_fast(
                        ortho,
                        e=cand_e,
                        n=cand_n,
                        heading=cand_h,
                        pitch=pitch,
                        height_m=height_m,
                        width=sw,
                        height=sh,
                        fov_deg=fov_deg,
                        image=edge_src,
                    )
                    ncc_v = ncc(small, synth)
                    edge_v = edge_agree(small, synth_e)
                    score = fuse_sat_score(ncc_v, edge_v, w_ncc=w_ncc, w_edge=w_edge)
                    if score > best["score"]:
                        best = {
                            "score": score,
                            "ncc": ncc_v,
                            "edge": edge_v,
                            "e": cand_e,
                            "n": cand_n,
                            "heading": cand_h,
                        }
    return best
