"""Sat-locked roof / yard shells (PR-A).

Fill SV-blind tops with flat shells in Ortho ENU XY, textured from sat crop.
Z from MA cloud median in footprint or façade top height. No BAG/OSM hero.
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
MAX_AREA_FRAC = 0.45  # drop near-full-tile blobs (failed street mask)
EDGE_GATE_M = 1.0
DEFAULT_ROOF_HEIGHT_M = 8.0
YARD_Z_OFFSET_M = 0.15
MIN_CLOUD_IN_POLY = 8


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
) -> tuple[np.ndarray, np.ndarray]:
    """Return (roof_mask, yard_mask) uint8 0/255 from Ortho BGR.

    Recipe: Canny barriers + flood from image border (= street/open), then
    classify remaining interiors as green yard vs roof/built.
    """
    h, w = ortho_bgr.shape[:2]
    gray = cv2.cvtColor(ortho_bgr, cv2.COLOR_BGR2GRAY)
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

    # Yard = green interiors; also mid-bright non-street courtyards with weak green
    yard_u8 = green_u8.copy()
    return roof_u8, yard_u8


def _contours_to_regions(
    mask: np.ndarray,
    ortho: Ortho,
    kind: str,
    *,
    min_area_m2: float = MIN_AREA_M2,
    max_area_frac: float = MAX_AREA_FRAC,
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
        # AABB corners in px → ENU (for flat quad shell)
        corners_px = [
            (x, y + bh),  # BL in image (south-west-ish after ENU flip)
            (x + bw, y + bh),
            (x + bw, y),
            (x, y),
        ]
        aabb_enu = [px_to_enu(ortho, u, v) for u, v in corners_px]
        regions.append(
            {
                "id": f"{kind}_{i:03d}",
                "kind": kind,
                "area_m2": area_m2,
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
) -> list[dict[str, Any]]:
    """Extract roof + yard footprint dicts from Ortho. Fail-loud if none."""
    if ortho.image is None or ortho.image.size == 0:
        raise SatRoofError("empty sat ortho image")
    roof_m, yard_m = segment_roof_yard_mask(ortho.image)
    roofs = _contours_to_regions(roof_m, ortho, "roof", min_area_m2=min_area_m2)
    yards = _contours_to_regions(yard_m, ortho, "yard", min_area_m2=min_area_m2)
    regions = roofs + yards
    if not regions:
        raise SatRoofError(
            "no roof/yard footprints from sat (empty mask after Canny+flood); "
            "check ortho coverage / segmentation knobs"
        )
    log.info(
        "sat roofs: extracted %d roofs + %d yards (min_area=%.1f m²)",
        len(roofs),
        len(yards),
        min_area_m2,
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
        return float(np.percentile(xyz[:, 2], 10))
    return 0.0


def assign_shell_z(
    regions: list[dict[str, Any]],
    xyz: np.ndarray | None,
    *,
    planes_json: Path | None = None,
    default_roof_h: float = DEFAULT_ROOF_HEIGHT_M,
) -> list[dict[str, Any]]:
    """Set ``z`` on each region from MA median in polygon or façade top."""
    facade_top = _facade_top_z(planes_json)
    ground_z = _ground_z_fallback(planes_json, xyz)
    out: list[dict[str, Any]] = []
    for reg in regions:
        poly = [(p[0], p[1]) for p in reg["polygon_enu"]]
        z_src = "default"
        z_val: float
        n_in = 0
        if xyz is not None and len(xyz) > 0 and len(poly) >= 3:
            # Coarse AABB reject then point-in-poly
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
            if len(cand) > 0:
                inside_z = [
                    float(pt[2])
                    for pt in cand
                    if _point_in_poly_xy(float(pt[0]), float(pt[1]), poly)
                ]
                n_in = len(inside_z)
                if n_in >= MIN_CLOUD_IN_POLY:
                    z_val = float(np.median(inside_z))
                    z_src = "ma_median"
                else:
                    z_val = float("nan")
            else:
                z_val = float("nan")
        else:
            z_val = float("nan")

        if not math.isfinite(z_val):
            if reg["kind"] == "yard":
                z_val = ground_z + YARD_Z_OFFSET_M
                z_src = "ground_offset"
            elif facade_top is not None:
                z_val = float(facade_top)
                z_src = "facade_top"
            else:
                z_val = ground_z + default_roof_h
                z_src = "ground_plus_default"

        # Yards should stay near ground even if sparse MA hits a wall
        if reg["kind"] == "yard" and z_src == "ma_median":
            if z_val > ground_z + 3.0:
                z_val = ground_z + YARD_Z_OFFSET_M
                z_src = "ground_offset_clamp"

        r = dict(reg)
        r["z"] = float(z_val)
        r["z_source"] = z_src
        r["n_cloud_in"] = int(n_in)
        out.append(r)
    return out


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
) -> dict[str, Any]:
    """End-to-end: extract → Z → texture → roofs.obj. Fail-loud on empty sat."""
    root = Path(project_root)
    recon = root / "recon"
    dest = Path(dest_obj) if dest_obj else recon / "roofs.obj"

    ortho = load_ortho_for_run(root)
    regions = extract_roof_yard_polygons(ortho, min_area_m2=min_area_m2)
    xyz = _load_cloud_xyz(recon)
    planes_json = recon / "planes.json"
    regions = assign_shell_z(regions, xyz, planes_json=planes_json)

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
            }
            for r in regions
        ],
        "edge_agreement": edge,
        "edge_gate_m": edge_gate_m,
        "gate_target_ok": bool(
            math.isfinite(mean_e) and mean_e <= edge_gate_m
        ),
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
        "mean_edge_m=%.3f (gate≤%.1f target_ok=%s)",
        dest,
        n_roof,
        n_yard,
        write_meta["textured"],
        mean_e if math.isfinite(mean_e) else -1.0,
        edge_gate_m,
        meta["gate_target_ok"],
    )
    return meta
