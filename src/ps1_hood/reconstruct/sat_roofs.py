"""Sat-locked roof / yard shells (PR-A).

Fill SV-blind tops with flat shells in Ortho ENU XY, textured from sat crop.
Roofs: Z from high percentile of MA z in footprint (or façade top) — never
street/cam height. Yards: low ground band, thin near ground. Footprints are
inset and rejected if they spill onto road / camera XY corridor. No BAG hero.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ps1_hood.align.sat_edges import ortho_canny
from ps1_hood.capture.satellite import Ortho
from ps1_hood.geo import BBox, LocalFrame

log = logging.getLogger(__name__)

# Pragmatic defaults (NL residential / smoke-dense scale)
MIN_AREA_M2 = 8.0
MAX_AREA_FRAC = 0.35  # drop near-full-tile blobs (failed street mask)
EDGE_GATE_M = 1.0
DEFAULT_ROOF_HEIGHT_M = 8.0
YARD_Z_OFFSET_M = 0.15
MIN_CLOUD_IN_POLY = 8
# Z: roofs use high band; yards use low ground band
ROOF_Z_PERCENTILE = 85.0
YARD_Z_PERCENTILE = 20.0
MIN_ROOF_ABOVE_GROUND_M = 4.0
YARD_MAX_ABOVE_GROUND_M = 1.5
# Footprint shrink / street reject
FOOTPRINT_INSET_M = 0.75  # AABB inset (m); mask erode uses half of this
CAM_CORRIDOR_MARGIN_M = 3.0
STREET_OVERLAP_FRAC = 0.35  # AABB rim always clips some asphalt — only severe spills
CAM_OVERLAP_FRAC = 0.10  # reject if camera corridor covers this fraction of AABB
MAX_ROOF_AABB_SIDE_M = 32.0  # over-wide slabs that swallow the street
MAX_ROOF_AREA_M2 = 180.0
STREET_SLAB_HALF_M = 2.0  # skip roof if z within ±this of cam_u (street view slab)


class SatRoofError(RuntimeError):
    """Fail-loud: missing ortho, empty sat, or zero footprints."""


def _metres_per_px(ortho: Ortho) -> tuple[float, float]:
    m_e = (ortho.ee - ortho.sw) / max(ortho.w - 1, 1)
    m_n = (ortho.nn - ortho.sh) / max(ortho.h - 1, 1)
    return float(m_e), float(m_n)


def px_to_enu(ortho: Ortho, u: float, v: float) -> tuple[float, float]:
    """Pixel (u,v) → ENU (e,n). Inverse of Ortho.enu_to_px."""
    e = ortho.sw + (u / max(ortho.w - 1, 1)) * (ortho.ee - ortho.sw)
    n = ortho.sh + (1.0 - v / max(ortho.h - 1, 1)) * (ortho.nn - ortho.sh)
    return float(e), float(n)


def enu_corners_quad(ortho: Ortho) -> list[tuple[float, float, float]]:
    """Four Ortho ENU corners at z=0 (SW, SE, NE, NW) — for tests / gates."""
    return [
        (ortho.sw, ortho.sh, 0.0),
        (ortho.ee, ortho.sh, 0.0),
        (ortho.ee, ortho.nn, 0.0),
        (ortho.sw, ortho.nn, 0.0),
    ]


def load_ortho_for_run(project_root: Path, frame: LocalFrame | None = None) -> Ortho:
    """Load Ortho from ``satellite/ortho.json``; fail-loud if missing/empty."""
    root = Path(project_root)
    meta_path = root / "satellite" / "ortho.json"
    if not meta_path.is_file():
        raise SatRoofError(f"no satellite ortho.json at {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise SatRoofError(f"invalid ortho.json at {meta_path}")
    path = Path(meta.get("path") or "")
    if not path.is_file():
        cand = root / "satellite" / "ortho.jpg"
        if cand.is_file():
            path = cand
            meta = dict(meta)
            meta["path"] = str(path)
        else:
            raise SatRoofError(f"empty/missing sat image: {meta.get('path')}")
    bbox = BBox.from_dict(meta["bbox"])
    if frame is None:
        frame = LocalFrame.from_bbox(bbox)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise SatRoofError(f"empty sat image: {path}")
    if image.shape[0] < 8 or image.shape[1] < 8:
        raise SatRoofError(f"sat image too small: {path} shape={image.shape}")
    return Ortho(image, bbox, frame)


def segment_roof_yard_mask(
    ortho_bgr: np.ndarray,
    *,
    canny_low: int = 40,
    canny_high: int = 120,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (roof_mask, yard_mask, street_mask) uint8 0/255 from Ortho BGR.

    Recipe: Canny barriers + flood from image border (= street/open), then
    classify remaining interiors as green yard vs roof/built. ``street_mask``
    is exterior flood ∪ dark asphalt leftovers (for footprint reject).
    """
    h, w = ortho_bgr.shape[:2]
    hsv = cv2.cvtColor(ortho_bgr, cv2.COLOR_BGR2HSV)
    edges = ortho_canny(ortho_bgr, low=canny_low, high=canny_high, close_ksize=3)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    barriers = cv2.dilate(edges, k3, iterations=2)

    # Pixels floodable = not strong edges
    open_px = (barriers == 0).astype(np.uint8) * 255
    # Flood from border → street / exterior
    flood = open_px.copy()
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    # Seed every border pixel that is open
    for x in range(w):
        if flood[0, x]:
            cv2.floodFill(flood, ff_mask, (x, 0), 128)
        if flood[h - 1, x]:
            cv2.floodFill(flood, ff_mask, (x, h - 1), 128)
    for y in range(h):
        if flood[y, 0]:
            cv2.floodFill(flood, ff_mask, (0, y), 128)
        if flood[y, w - 1]:
            cv2.floodFill(flood, ff_mask, (w - 1, y), 128)

    exterior = flood == 128
    interior = (open_px > 0) & (~exterior)
    # Also claim edge-barrier pixels that are not exterior-adjacent as interior fill
    # by closing small holes inside buildings
    interior_u8 = interior.astype(np.uint8) * 255
    interior_u8 = cv2.morphologyEx(interior_u8, cv2.MORPH_CLOSE, k5, iterations=2)
    interior_u8 = cv2.morphologyEx(interior_u8, cv2.MORPH_OPEN, k3, iterations=1)

    # Green vegetation / yards
    hue, sat, val = cv2.split(hsv)
    green = (
        (hue >= 30)
        & (hue <= 95)
        & (sat > 28)
        & (val > 35)
        & (interior_u8 > 0)
    )
    green_u8 = green.astype(np.uint8) * 255
    green_u8 = cv2.morphologyEx(green_u8, cv2.MORPH_OPEN, k3)
    green_u8 = cv2.morphologyEx(green_u8, cv2.MORPH_CLOSE, k5)

    roof_u8 = cv2.subtract(interior_u8, green_u8)
    # Soft asphalt street leftovers: low-sat mid-V still inside → drop from roof
    streetish = (sat < 35) & (val > 45) & (val < 130) & (roof_u8 > 0)
    # Only strip thin streetish corridors (keep colorful roofs)
    streetish_u8 = streetish.astype(np.uint8) * 255
    streetish_u8 = cv2.morphologyEx(streetish_u8, cv2.MORPH_OPEN, k5, iterations=2)
    roof_u8 = cv2.subtract(roof_u8, streetish_u8)

    # Yard = green interiors
    yard_u8 = green_u8.copy()
    # Street corridor mask: exterior flood ∪ asphalt leftovers
    street_u8 = (exterior.astype(np.uint8) * 255)
    street_u8 = cv2.bitwise_or(street_u8, streetish_u8)
    return roof_u8, yard_u8, street_u8


def _erode_mask_metres(
    mask: np.ndarray,
    ortho: Ortho,
    inset_m: float,
) -> np.ndarray:
    """Erode a uint8 mask by roughly ``inset_m`` metres (isotropic kernel)."""
    if inset_m <= 0:
        return mask
    m_e, m_n = _metres_per_px(ortho)
    m_px = 0.5 * (abs(m_e) + abs(m_n))
    r = max(1, int(round(inset_m / max(m_px, 1e-6))))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.erode(mask, k, iterations=1)


def _inset_bbox_px(
    bbox_px: tuple[int, int, int, int],
    ortho: Ortho,
    inset_m: float,
) -> tuple[int, int, int, int] | None:
    """Shrink pixel AABB by ``inset_m``; return None if degenerate."""
    x, y, bw, bh = bbox_px
    m_e, m_n = _metres_per_px(ortho)
    dx = int(round(inset_m / max(abs(m_e), 1e-6)))
    dy = int(round(inset_m / max(abs(m_n), 1e-6)))
    x2 = x + dx
    y2 = y + dy
    bw2 = bw - 2 * dx
    bh2 = bh - 2 * dy
    if bw2 < 2 or bh2 < 2:
        return None
    # Clamp to image
    x2 = max(0, x2)
    y2 = max(0, y2)
    bw2 = min(bw2, ortho.w - x2)
    bh2 = min(bh2, ortho.h - y2)
    if bw2 < 2 or bh2 < 2:
        return None
    return int(x2), int(y2), int(bw2), int(bh2)


def _contours_to_regions(
    mask: np.ndarray,
    ortho: Ortho,
    kind: str,
    *,
    min_area_m2: float = MIN_AREA_M2,
    max_area_frac: float = MAX_AREA_FRAC,
    inset_m: float = FOOTPRINT_INSET_M,
) -> list[dict[str, Any]]:
    m_e, m_n = _metres_per_px(ortho)
    px_area_m2 = abs(m_e * m_n)
    tile_m2 = abs((ortho.ee - ortho.sw) * (ortho.nn - ortho.sh))
    max_area = max_area_frac * tile_m2
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions: list[dict[str, Any]] = []
    for i, cnt in enumerate(contours):
        area_px = float(cv2.contourArea(cnt))
        area_m2 = area_px * px_area_m2
        if area_m2 < min_area_m2 or area_m2 > max_area:
            continue
        peri = cv2.arcLength(cnt, True)
        eps = max(1.5, 0.02 * peri)
        approx = cv2.approxPolyDP(cnt, eps, True)
        if len(approx) < 3:
            continue
        pts_px = [(float(p[0][0]), float(p[0][1])) for p in approx]
        pts_enu = [px_to_enu(ortho, u, v) for u, v in pts_px]
        x, y, bw, bh = cv2.boundingRect(cnt)
        inset = _inset_bbox_px((x, y, bw, bh), ortho, inset_m)
        if inset is None:
            log.info("sat roofs: skip %s_%03d — AABB vanished after %.2f m inset", kind, i, inset_m)
            continue
        x, y, bw, bh = inset
        # AABB corners in px → ENU (for flat quad shell)
        corners_px = [
            (x, y + bh),  # BL in image (south-west-ish after ENU flip)
            (x + bw, y + bh),
            (x + bw, y),
            (x, y),
        ]
        aabb_enu = [px_to_enu(ortho, u, v) for u, v in corners_px]
        # Inset area estimate (AABB)
        area_aabb_m2 = abs(bw * m_e) * abs(bh * m_n)
        regions.append(
            {
                "id": f"{kind}_{i:03d}",
                "kind": kind,
                "area_m2": float(min(area_m2, area_aabb_m2)),
                "polygon_enu": pts_enu,
                "aabb_enu": aabb_enu,
                "bbox_px": (int(x), int(y), int(bw), int(bh)),
            }
        )
    return regions


def extract_roof_yard_polygons(
    ortho: Ortho,
    *,
    min_area_m2: float = MIN_AREA_M2,
    inset_m: float = FOOTPRINT_INSET_M,
) -> list[dict[str, Any]]:
    """Extract roof + yard footprint dicts from Ortho. Fail-loud if none.

    Masks are eroded by ``inset_m`` before contours; AABBs are inset again so
    shells stay inside sat footprints and off the road rim.
    """
    if ortho.image is None or ortho.image.size == 0:
        raise SatRoofError("empty sat ortho image")
    roof_m, yard_m, street_m = segment_roof_yard_mask(ortho.image)
    # Light mask erode (half inset) — AABB inset does the rest; heavy erode
    # deleted most NL roof blobs on smoke-dense.
    roof_m = _erode_mask_metres(roof_m, ortho, inset_m * 0.5)
    # Yards are small green blobs — mask erode deletes them; AABB inset only.
    roofs = _contours_to_regions(
        roof_m, ortho, "roof", min_area_m2=min_area_m2, inset_m=inset_m
    )
    yards = _contours_to_regions(
        yard_m, ortho, "yard", min_area_m2=min_area_m2, inset_m=min(0.5, inset_m * 0.5)
    )
    regions = roofs + yards
    if not regions:
        raise SatRoofError(
            "no roof/yard footprints from sat (empty mask after Canny+flood); "
            "check ortho coverage / segmentation knobs"
        )
    # Stash street mask on ortho for later reject (caller may recompute)
    ortho._street_mask = street_m  # type: ignore[attr-defined]
    log.info(
        "sat roofs: extracted %d roofs + %d yards (min_area=%.1f m² inset=%.2f m)",
        len(roofs),
        len(yards),
        min_area_m2,
        inset_m,
    )
    return regions


def _point_in_poly_xy(e: float, n: float, poly: list[tuple[float, float]]) -> bool:
    """Ray-cast even-odd."""
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        ei, ni = poly[i]
        ej, nj = poly[j]
        if ((ni > n) != (nj > n)) and (
            e < (ej - ei) * (n - ni) / (nj - ni + 1e-15) + ei
        ):
            inside = not inside
        j = i
    return inside


def _facade_top_z(planes_json: Path | None) -> float | None:
    if planes_json is None or not Path(planes_json).is_file():
        return None
    try:
        payload = json.loads(Path(planes_json).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    planes = payload.get("planes") if isinstance(payload, dict) else None
    if not planes:
        return None
    tops: list[float] = []
    for pl in planes:
        if not isinstance(pl, dict):
            continue
        quad = pl.get("quad")
        if isinstance(quad, list) and quad:
            try:
                tops.append(max(float(v[2]) for v in quad))
            except (TypeError, ValueError, IndexError):
                continue
        h = pl.get("height_m")
        gz = float(payload.get("ground_z") or 0.0) if isinstance(payload, dict) else 0.0
        if h is not None and not tops:
            tops.append(gz + float(h))
    if not tops:
        return None
    return float(np.median(tops))


def _ground_z_fallback(planes_json: Path | None, xyz: np.ndarray | None) -> float:
    if planes_json is not None and Path(planes_json).is_file():
        try:
            payload = json.loads(Path(planes_json).read_text(encoding="utf-8"))
            if isinstance(payload, dict) and payload.get("ground_z") is not None:
                return float(payload["ground_z"])
        except Exception:  # noqa: BLE001
            pass
    if xyz is not None and len(xyz) >= 20:
        z = xyz[:, 2]
        p10 = float(np.percentile(z, 10))
        p50 = float(np.percentile(z, 50))
        # Roof-only clouds: low spread at height — don't treat as ground
        if (p50 - p10) < 1.5 and p10 > 3.0:
            return 0.0
        return p10
    return 0.0


def _zs_in_polygon(
    xyz: np.ndarray,
    poly: list[tuple[float, float]],
) -> list[float]:
    """Return z values of cloud points inside polygon (AABB prefilter)."""
    if len(xyz) == 0 or len(poly) < 3:
        return []
    es = [p[0] for p in poly]
    ns = [p[1] for p in poly]
    e0, e1 = min(es), max(es)
    n0, n1 = min(ns), max(ns)
    sel = (
        (xyz[:, 0] >= e0)
        & (xyz[:, 0] <= e1)
        & (xyz[:, 1] >= n0)
        & (xyz[:, 1] <= n1)
    )
    cand = xyz[sel]
    if len(cand) == 0:
        return []
    return [
        float(pt[2])
        for pt in cand
        if _point_in_poly_xy(float(pt[0]), float(pt[1]), poly)
    ]


def assign_shell_z(
    regions: list[dict[str, Any]],
    xyz: np.ndarray | None,
    *,
    planes_json: Path | None = None,
    default_roof_h: float = DEFAULT_ROOF_HEIGHT_M,
    cam_u: float | None = None,
) -> list[dict[str, Any]]:
    """Set ``z`` on each region.

    Roofs: high percentile of MA z in footprint if clearly above ground /
    cam slab; else façade top / ``ground+default``. Never street/cam height.
    Yards: low percentile ground band (thin near ground).
    """
    facade_top = _facade_top_z(planes_json)
    ground_z = _ground_z_fallback(planes_json, xyz)
    # Street-view slab height ≈ camera up; refuse roofs near this band.
    street_slab_z = float(cam_u) if cam_u is not None and math.isfinite(cam_u) else ground_z + 1.5
    min_roof_z = max(ground_z + MIN_ROOF_ABOVE_GROUND_M, street_slab_z + STREET_SLAB_HALF_M)
    out: list[dict[str, Any]] = []
    for reg in regions:
        poly = [(p[0], p[1]) for p in reg["polygon_enu"]]
        z_src = "default"
        z_val = float("nan")
        n_in = 0
        inside_z: list[float] = []
        if xyz is not None and len(xyz) > 0 and len(poly) >= 3:
            inside_z = _zs_in_polygon(xyz, poly)
            n_in = len(inside_z)

        if reg["kind"] == "yard":
            if n_in >= MIN_CLOUD_IN_POLY:
                z_low = float(np.percentile(inside_z, YARD_Z_PERCENTILE))
                # Keep yards in a thin ground band
                lo = ground_z - 0.3
                hi = ground_z + YARD_MAX_ABOVE_GROUND_M
                if lo <= z_low <= hi:
                    z_val = z_low
                    z_src = "ma_low"
                else:
                    z_val = ground_z + YARD_Z_OFFSET_M
                    z_src = "ground_offset_clamp"
            else:
                z_val = ground_z + YARD_Z_OFFSET_M
                z_src = "ground_offset"
        else:
            # Roof: high percentile only if above façade-ish / street slab
            if n_in >= MIN_CLOUD_IN_POLY:
                z_high = float(np.percentile(inside_z, ROOF_Z_PERCENTILE))
                if z_high >= min_roof_z:
                    z_val = z_high
                    z_src = "ma_high"
                else:
                    log.info(
                        "sat roofs: %s MA p%.0f=%.2f < min_roof_z=%.2f — fall back",
                        reg["id"],
                        ROOF_Z_PERCENTILE,
                        z_high,
                        min_roof_z,
                    )
            if not math.isfinite(z_val):
                if facade_top is not None and float(facade_top) >= min_roof_z - 0.5:
                    z_val = float(facade_top)
                    z_src = "facade_top"
                else:
                    z_val = ground_z + default_roof_h
                    z_src = "ground_plus_default"
            # Final guard: never leave a roof in the street slab
            if z_val < min_roof_z:
                if facade_top is not None:
                    z_val = max(float(facade_top), min_roof_z)
                    z_src = "facade_top_clamp"
                else:
                    z_val = max(ground_z + default_roof_h, min_roof_z)
                    z_src = "ground_plus_default_clamp"

        r = dict(reg)
        r["z"] = float(z_val)
        r["z_source"] = z_src
        r["n_cloud_in"] = int(n_in)
        r["ground_z"] = float(ground_z)
        r["min_roof_z"] = float(min_roof_z)
        out.append(r)
    return out


def load_cam_xy_enu(project_root: Path) -> tuple[np.ndarray | None, float | None]:
    """Load unique camera (e, n) and median ``u`` from align/ or scene.json."""
    root = Path(project_root)
    candidates = [
        root / "align" / "cameras.json",
        root / "align" / "poses.json",
        root / "recon" / "scene.json",
    ]
    cams: list[dict[str, Any]] = []
    for path in candidates:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if isinstance(payload, list):
            cams = [c for c in payload if isinstance(c, dict) and "e" in c and "n" in c]
        elif isinstance(payload, dict):
            raw = payload.get("cameras") or payload.get("poses") or []
            if isinstance(raw, list):
                cams = [c for c in raw if isinstance(c, dict) and "e" in c and "n" in c]
        if cams:
            break
    if not cams:
        return None, None
    xy = np.array([[float(c["e"]), float(c["n"])] for c in cams], dtype=np.float64)
    # Unique pano positions (many headings share e,n)
    rounded = np.round(xy, 3)
    _, idx = np.unique(rounded, axis=0, return_index=True)
    xy_u = xy[np.sort(idx)]
    us = [float(c["u"]) for c in cams if c.get("u") is not None]
    cam_u = float(np.median(us)) if us else None
    return xy_u, cam_u


def _aabb_bounds(aabb_enu: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    es = [p[0] for p in aabb_enu]
    ns = [p[1] for p in aabb_enu]
    return min(es), max(es), min(ns), max(ns)


def _aabb_area_m2(aabb_enu: list[tuple[float, float]]) -> float:
    e0, e1, n0, n1 = _aabb_bounds(aabb_enu)
    return abs(e1 - e0) * abs(n1 - n0)


def cam_corridor_overlap_frac(
    aabb_enu: list[tuple[float, float]],
    cam_xy: np.ndarray,
    *,
    margin_m: float = CAM_CORRIDOR_MARGIN_M,
) -> float:
    """Approx fraction of AABB covered by dilated camera XY disks."""
    e0, e1, n0, n1 = _aabb_bounds(aabb_enu)
    area = abs(e1 - e0) * abs(n1 - n0)
    if area < 1e-6 or cam_xy is None or len(cam_xy) == 0:
        return 0.0
    # Grid sample AABB and count points within margin of any cam
    n_e = max(4, int(round(abs(e1 - e0) / max(margin_m * 0.5, 0.5))))
    n_n = max(4, int(round(abs(n1 - n0) / max(margin_m * 0.5, 0.5))))
    n_e = min(n_e, 40)
    n_n = min(n_n, 40)
    ee = np.linspace(e0, e1, n_e)
    nn = np.linspace(n0, n1, n_n)
    eg, ng = np.meshgrid(ee, nn)
    pts = np.column_stack([eg.ravel(), ng.ravel()])
    # Distance to nearest cam
    d2 = ((pts[:, None, 0] - cam_xy[None, :, 0]) ** 2) + (
        (pts[:, None, 1] - cam_xy[None, :, 1]) ** 2
    )
    dmin = np.sqrt(d2.min(axis=1))
    return float((dmin <= margin_m).mean())


def street_mask_overlap_frac(
    bbox_px: tuple[int, int, int, int],
    street_mask: np.ndarray | None,
) -> float:
    """Fraction of AABB pixels that are street/asphalt."""
    if street_mask is None or street_mask.size == 0:
        return 0.0
    x, y, bw, bh = bbox_px
    h, w = street_mask.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w, x + bw), min(h, y + bh)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    crop = street_mask[y0:y1, x0:x1]
    return float((crop > 0).mean())


def reject_street_overlapping_regions(
    regions: list[dict[str, Any]],
    *,
    cam_xy: np.ndarray | None = None,
    street_mask: np.ndarray | None = None,
    cam_margin_m: float = CAM_CORRIDOR_MARGIN_M,
    street_overlap_frac: float = STREET_OVERLAP_FRAC,
    cam_overlap_frac: float = CAM_OVERLAP_FRAC,
) -> list[dict[str, Any]]:
    """Drop footprints that spill onto road / camera corridor or are over-wide."""
    kept: list[dict[str, Any]] = []
    for reg in regions:
        cam_frac = (
            cam_corridor_overlap_frac(reg["aabb_enu"], cam_xy, margin_m=cam_margin_m)
            if cam_xy is not None
            else 0.0
        )
        st_frac = street_mask_overlap_frac(reg["bbox_px"], street_mask)
        e0, e1, n0, n1 = _aabb_bounds(reg["aabb_enu"])
        side_e = abs(e1 - e0)
        side_n = abs(n1 - n0)
        area = float(reg.get("area_m2") or _aabb_area_m2(reg["aabb_enu"]))

        reasons: list[str] = []
        if reg["kind"] == "roof":
            if cam_frac >= cam_overlap_frac:
                reasons.append(f"cam_frac={cam_frac:.2f}")
            if st_frac >= street_overlap_frac:
                reasons.append(f"street_frac={st_frac:.2f}")
            if side_e > MAX_ROOF_AABB_SIDE_M or side_n > MAX_ROOF_AABB_SIDE_M:
                reasons.append(f"aabb_side={max(side_e, side_n):.1f}m")
            # Large area alone is OK on big blocks; require spill evidence
            if area > MAX_ROOF_AREA_M2 and (
                cam_frac >= cam_overlap_frac * 0.5 or st_frac >= 0.20
            ):
                reasons.append(f"area={area:.1f}m²+spill")
        else:
            # Yards: only reject severe cam/street swallow
            if cam_frac >= cam_overlap_frac * 1.5 or st_frac >= street_overlap_frac * 1.2:
                reasons.append(f"cam={cam_frac:.2f}/st={st_frac:.2f}")

        if reasons:
            log.warning(
                "sat roofs: reject %s — %s",
                reg["id"],
                ", ".join(reasons),
            )
            continue
        r = dict(reg)
        r["cam_overlap_frac"] = float(cam_frac)
        r["street_overlap_frac"] = float(st_frac)
        kept.append(r)
    return kept


def filter_street_slab_shells(
    regions: list[dict[str, Any]],
    *,
    cam_u: float | None = None,
) -> list[dict[str, Any]]:
    """Skip roof shells whose Z still sits in the street-view camera slab."""
    kept: list[dict[str, Any]] = []
    for reg in regions:
        if reg["kind"] != "roof":
            kept.append(reg)
            continue
        z = float(reg["z"])
        gz = float(reg.get("ground_z", 0.0))
        min_z = float(reg.get("min_roof_z", gz + MIN_ROOF_ABOVE_GROUND_M))
        if z < min_z:
            log.warning(
                "sat roofs: skip %s — z=%.2f below min_roof_z=%.2f (street slab)",
                reg["id"],
                z,
                min_z,
            )
            continue
        if cam_u is not None and abs(z - float(cam_u)) <= STREET_SLAB_HALF_M:
            log.warning(
                "sat roofs: skip %s — z=%.2f intersects cam slab u=%.2f ±%.1f",
                reg["id"],
                z,
                float(cam_u),
                STREET_SLAB_HALF_M,
            )
            continue
        kept.append(reg)
    return kept


def _crop_sat_texture(
    ortho: Ortho,
    bbox_px: tuple[int, int, int, int],
    dest: Path,
) -> bool:
    x, y, bw, bh = bbox_px
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(ortho.w, x + bw)
    y1 = min(ortho.h, y + bh)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return False
    crop = ortho.image[y0:y1, x0:x1]
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Nearest-friendly: write as-is (viewer uses NearestFilter)
    return bool(cv2.imwrite(str(dest), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90]))


def shell_edge_agreement_m(
    regions: list[dict[str, Any]],
    ortho: Ortho,
    *,
    n_samples: int = 24,
) -> dict[str, float]:
    """Mean distance (m) from shell AABB edges to nearest Ortho Canny edge.

    Gate target: ≤ ~1 m where measurable.
    """
    edges = ortho_canny(ortho.image)
    inv = np.where(edges > 0, 0, 255).astype(np.uint8)
    dt = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    m_e, m_n = _metres_per_px(ortho)
    m_px = 0.5 * (abs(m_e) + abs(m_n))
    dists: list[float] = []
    for reg in regions:
        x, y, bw, bh = reg["bbox_px"]
        # Sample perimeter of AABB in px
        perim: list[tuple[float, float]] = []
        for t in np.linspace(0, 1, max(4, n_samples // 4), endpoint=False):
            perim.append((x + t * bw, y))
            perim.append((x + t * bw, y + bh))
            perim.append((x, y + t * bh))
            perim.append((x + bw, y + t * bh))
        for u, v in perim:
            ui = int(round(u))
            vi = int(round(v))
            if 0 <= ui < ortho.w and 0 <= vi < ortho.h:
                dists.append(float(dt[vi, ui]) * m_px)
    if not dists:
        return {"mean_edge_m": float("nan"), "p90_edge_m": float("nan"), "n": 0}
    arr = np.asarray(dists, dtype=np.float64)
    return {
        "mean_edge_m": float(arr.mean()),
        "p90_edge_m": float(np.percentile(arr, 90)),
        "n": int(len(arr)),
    }


def write_roofs_obj(
    dest_obj: Path,
    regions: list[dict[str, Any]],
    ortho: Ortho,
    *,
    tex_subdir: str = "textures",
) -> dict[str, Any]:
    """Write flat AABB shell quads + sat-crop textures → roofs.obj / .mtl."""
    dest_obj = Path(dest_obj)
    dest_obj.parent.mkdir(parents=True, exist_ok=True)
    tex_dir = dest_obj.parent / tex_subdir
    tex_dir.mkdir(parents=True, exist_ok=True)
    mtl_name = dest_obj.with_suffix(".mtl").name

    materials: list[dict[str, Any]] = []
    faces: list[tuple[int, int, int, int, int]] = []
    verts: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]

    for i, reg in enumerate(regions):
        z = float(reg["z"])
        aabb = reg["aabb_enu"]  # 4 × (e,n)
        # Order: BL, BR, TR, TL in ENU (matches UV)
        quad = [
            (aabb[0][0], aabb[0][1], z),
            (aabb[1][0], aabb[1][1], z),
            (aabb[2][0], aabb[2][1], z),
            (aabb[3][0], aabb[3][1], z),
        ]
        tex_rel = f"{tex_subdir}/roof_{reg['id']}.jpg"
        tex_path = dest_obj.parent / tex_rel
        ok = _crop_sat_texture(ortho, reg["bbox_px"], tex_path)
        mat_name = f"roof_{reg['id']}"
        materials.append(
            {
                "name": mat_name,
                "map": tex_rel if ok else None,
                "kd": (0.55, 0.45, 0.40) if reg["kind"] == "roof" else (0.35, 0.55, 0.30),
            }
        )
        base = len(verts)
        verts.extend(quad)
        # reuse first 4 uvs pattern — append matching
        if i == 0:
            pass
        else:
            uvs.extend([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
        faces.append((base, base + 1, base + 2, base + 3, i))

    mtl_path = dest_obj.with_suffix(".mtl")
    with mtl_path.open("w", encoding="ascii") as fh:
        fh.write("# ps1-hood sat roof/yard materials\n")
        for mat in materials:
            fh.write(f"newmtl {mat['name']}\n")
            kd = mat["kd"]
            fh.write(f"Kd {kd[0]:.3f} {kd[1]:.3f} {kd[2]:.3f}\n")
            fh.write("Ka 0.050 0.050 0.050\n")
            fh.write("d 1.0\n")
            fh.write("illum 1\n")
            if mat.get("map"):
                fh.write(f"map_Kd {mat['map']}\n")
            fh.write("\n")

    with dest_obj.open("w", encoding="ascii") as fh:
        fh.write("# ps1-hood sat-locked roof/yard shells (Ortho ENU XY)\n")
        fh.write(f"mtllib {mtl_name}\n")
        for v in verts:
            fh.write(f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n")
        for uv in uvs:
            fh.write(f"vt {uv[0]:.4f} {uv[1]:.4f}\n")
        cur = None
        for v0, v1, v2, v3, mi in faces:
            name = materials[mi]["name"]
            if name != cur:
                fh.write(f"usemtl {name}\n")
                cur = name
            fh.write(
                f"f {v0 + 1}/{v0 + 1} {v1 + 1}/{v1 + 1} "
                f"{v2 + 1}/{v2 + 1} {v3 + 1}/{v3 + 1}\n"
            )

    n_tex = sum(1 for m in materials if m.get("map"))
    return {
        "obj": str(dest_obj),
        "mtl": str(mtl_path),
        "shells": len(regions),
        "textured": n_tex,
        "verts": len(verts),
    }


def _load_cloud_xyz(recon_dir: Path) -> np.ndarray | None:
    from ps1_hood.reconstruct.facades import _read_ply_xyz

    recon_dir = Path(recon_dir)
    for name in ("cloud.ply", "cloud_photo.ply", "cloud_flow.ply"):
        p = recon_dir / name
        if p.is_file() and p.stat().st_size > 200:
            try:
                xyz = _read_ply_xyz(p)
                if len(xyz) > 0:
                    log.info("sat roofs: Z from %s (%d pts)", p.name, len(xyz))
                    return xyz
            except Exception as exc:  # noqa: BLE001
                log.warning("sat roofs: failed reading %s: %s", p, exc)
    return None


def build_sat_roofs(
    project_root: Path,
    *,
    dest_obj: Path | None = None,
    min_area_m2: float = MIN_AREA_M2,
    edge_gate_m: float = EDGE_GATE_M,
    inset_m: float = FOOTPRINT_INSET_M,
) -> dict[str, Any]:
    """End-to-end: extract → street reject → Z → slab filter → roofs.obj.

    Fail-loud on empty sat / zero footprints after rejects.
    """
    root = Path(project_root)
    recon = root / "recon"
    dest = Path(dest_obj) if dest_obj else recon / "roofs.obj"

    ortho = load_ortho_for_run(root)
    regions = extract_roof_yard_polygons(
        ortho, min_area_m2=min_area_m2, inset_m=inset_m
    )
    street_mask = getattr(ortho, "_street_mask", None)
    if street_mask is None:
        _, _, street_mask = segment_roof_yard_mask(ortho.image)

    cam_xy, cam_u = load_cam_xy_enu(root)
    n_before = len(regions)
    regions = reject_street_overlapping_regions(
        regions, cam_xy=cam_xy, street_mask=street_mask
    )
    n_rejected_street = n_before - len(regions)
    if not regions:
        raise SatRoofError(
            "all roof/yard footprints rejected (street/cam corridor overlap); "
            "check sat segmentation / camera poses"
        )

    xyz = _load_cloud_xyz(recon)
    planes_json = recon / "planes.json"
    regions = assign_shell_z(
        regions, xyz, planes_json=planes_json, cam_u=cam_u
    )
    n_before_slab = len(regions)
    regions = filter_street_slab_shells(regions, cam_u=cam_u)
    n_rejected_slab = n_before_slab - len(regions)
    if not regions:
        raise SatRoofError(
            "all shells skipped (street-view slab / too-low Z); "
            "need façade tops or MA high-z in footprints"
        )

    edge = shell_edge_agreement_m(regions, ortho)
    mean_e = edge.get("mean_edge_m", float("nan"))
    write_meta = write_roofs_obj(dest, regions, ortho)

    n_roof = sum(1 for r in regions if r["kind"] == "roof")
    n_yard = sum(1 for r in regions if r["kind"] == "yard")
    gate_ok = (not math.isfinite(mean_e)) or (mean_e <= edge_gate_m * 2.5)
    # Soft gate: log loudly if >1 m; hard-fail only if absurd (>5 m) — sat Canny
    # is noisy; product still useful. Measurable ≤1 m is the *target*.
    if math.isfinite(mean_e) and mean_e > 5.0:
        raise SatRoofError(
            f"shell edge agreement mean_edge_m={mean_e:.2f} ≫ {edge_gate_m} m gate"
        )

    shells_json = {
        "frame": "ENU",
        "source": "sat_ortho_canny_flood",
        "shells": [
            {
                "id": r["id"],
                "kind": r["kind"],
                "z": r["z"],
                "z_source": r["z_source"],
                "area_m2": r["area_m2"],
                "aabb_enu": r["aabb_enu"],
                "n_cloud_in": r["n_cloud_in"],
                "cam_overlap_frac": r.get("cam_overlap_frac"),
                "street_overlap_frac": r.get("street_overlap_frac"),
            }
            for r in regions
        ],
        "edge_agreement": edge,
        "edge_gate_m": edge_gate_m,
        "gate_target_ok": bool(
            math.isfinite(mean_e) and mean_e <= edge_gate_m
        ),
        "rejected_street": int(n_rejected_street),
        "rejected_slab": int(n_rejected_slab),
        "cam_u_median": cam_u,
        "inset_m": inset_m,
    }
    shells_path = recon / "roofs.json"
    shells_path.write_text(json.dumps(shells_json, indent=2), encoding="utf-8")

    meta = {
        **write_meta,
        "n_roof": n_roof,
        "n_yard": n_yard,
        "mean_edge_m": mean_e,
        "p90_edge_m": edge.get("p90_edge_m"),
        "edge_gate_m": edge_gate_m,
        "gate_target_ok": shells_json["gate_target_ok"],
        "gate_soft_ok": gate_ok,
        "rejected_street": int(n_rejected_street),
        "rejected_slab": int(n_rejected_slab),
        "roofs_json": str(shells_path),
        "ortho_enu": {
            "sw": ortho.sw,
            "sh": ortho.sh,
            "ee": ortho.ee,
            "nn": ortho.nn,
        },
    }
    log.info(
        "sat roofs: wrote %s  roofs=%d yards=%d textured=%d  "
        "mean_edge_m=%.3f (gate≤%.1f target_ok=%s)  "
        "rejected_street=%d rejected_slab=%d",
        dest,
        n_roof,
        n_yard,
        write_meta["textured"],
        mean_e if math.isfinite(mean_e) else -1.0,
        edge_gate_m,
        meta["gate_target_ok"],
        n_rejected_street,
        n_rejected_slab,
    )
    return meta
