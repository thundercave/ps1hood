"""Ortho road/roof Canny edges for sat absolute XY alignment.

Sacred: edges come from the Esri ortho (authority), not 3DBAG.
Reuse ``photo_edge_dt`` from bag_edges for the photo side only.
"""

from __future__ import annotations

import numpy as np
import cv2

from ps1_hood.align.bag_edges import photo_edge_dt
from ps1_hood.capture.satellite import Ortho


def ortho_canny(
    image_bgr: np.ndarray,
    *,
    low: int = 40,
    high: int = 120,
    close_ksize: int = 3,
) -> np.ndarray:
    """Canny edge map (uint8 0/255) of an Ortho BGR mosaic."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, low, high)
    if close_ksize and close_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (close_ksize, close_ksize))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k)
    return edges


def ortho_edge_bgr(ortho: Ortho, *, low: int = 40, high: int = 120) -> np.ndarray:
    """3-channel edge image suitable for the same remap as sat synth RGB."""
    edges = ortho_canny(ortho.image, low=low, high=high)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def edge_agree(
    photo_bgr: np.ndarray,
    synth_edge_bgr: np.ndarray,
    *,
    y0_frac: float = 0.45,
    sigma_px: float = 4.0,
) -> float:
    """Chamfer-style agreement in [0, 1]: photo Canny vs remapped Ortho edges.

    Builds a distance transform of the synthesised sat edges, samples at photo
    edge pixels in the lower FOV (ground band), returns ``exp(-mean_dt/sigma)``.
    Returns ``-1.0`` when either side has too few edges.
    """
    if photo_bgr.shape[:2] != synth_edge_bgr.shape[:2]:
        raise ValueError("photo and synth_edge must share HxW")
    h, w = photo_bgr.shape[:2]
    y0 = int(h * y0_frac)

    synth_gray = cv2.cvtColor(synth_edge_bgr, cv2.COLOR_BGR2GRAY)
    # Remap leaves black outside valid ground rays; treat bright as edge.
    sat_edge = (synth_gray > 20).astype(np.uint8) * 255
    # Restrict to lower band where warp is defined.
    sat_edge[:y0, :] = 0
    if int((sat_edge > 0).sum()) < 40:
        return -1.0

    inv = np.where(sat_edge > 0, 0, 255).astype(np.uint8)
    dt = cv2.distanceTransform(inv, cv2.DIST_L2, 3)

    photo_gray = cv2.cvtColor(photo_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(photo_gray, (3, 3), 0)
    photo_e = cv2.Canny(blur, 60, 160)
    photo_e[:y0, :] = 0
    ys, xs = np.where(photo_e > 0)
    if len(ys) < 40:
        return -1.0

    mean_d = float(dt[ys, xs].mean())
    if not np.isfinite(mean_d):
        return -1.0
    return float(np.exp(-mean_d / max(sigma_px, 1e-3)))


def photo_dt_for_sat(photo_bgr: np.ndarray) -> np.ndarray:
    """Photo edge DT (same as BAG snap) — exposed for callers that cache it."""
    return photo_edge_dt(photo_bgr)
