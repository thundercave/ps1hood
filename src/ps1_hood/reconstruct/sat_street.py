"""Sat-locked street / ground shells — fill parking / road holes under the drive.

Ortho street_mask minus dilated roof∪yard → flat ENU ground quads at ground_z,
textured from sat crop into recon/street.obj (separate from façades / roofs).
Sacred: sat absolute XY · no free-pose · don’t clobber facades.obj / roofs.obj / planes.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ps1_hood.capture.satellite import Ortho
from ps1_hood.reconstruct.sat_roofs import (
    MAX_AREA_FRAC,
    MIN_AREA_M2,
    SatRoofError,
    _crop_sat_texture,
    _metres_per_px,
    load_cam_xy_enu,
    load_ortho_for_run,
    px_to_enu,
    segment_roof_yard_mask,
)

log = logging.getLogger(__name__)

# Pack defaults (smoke-dense / NL residential)
DEFAULT_BUILDING_DILATE_M = 1.5
DEFAULT_CAM_CORRIDOR_M = 4.0
DEFAULT_CAMERA_HEIGHT_M = 2.5
MAX_TILE_SIDE_M = 25.0
STREET_KD = (0.42, 0.42, 0.40)  # asphalt-ish fallback


class SatStreetError(SatRoofError):
    """Fail-loud: missing ortho, empty support, or zero street tiles."""


def _dilate_mask_metres(
    mask: np.ndarray,
    ortho: Ortho,
    dilate_m: float,
) -> np.ndarray:
    """Dilate a uint8 mask by roughly ``dilate_m`` metres."""
    if dilate_m <= 0:
        return mask
    m_e, m_n = _metres_per_px(ortho)
    m_px = 0.5 * (abs(m_e) + abs(m_n))
    r = max(1, int(round(dilate_m / max(m_px, 1e-6))))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(mask, k, iterations=1)


def cam_xy_mask(
    ortho: Ortho,
    cam_xy: np.ndarray | None,
    *,
    corridor_m: float = DEFAULT_CAM_CORRIDOR_M,
) -> np.ndarray:
    """Uint8 mask of camera XY disks dilated by ``corridor_m`` (Ortho pixels)."""
    h, w = ortho.h, ortho.w
    out = np.zeros((h, w), dtype=np.uint8)
    if cam_xy is None or len(cam_xy) == 0 or corridor_m <= 0:
        return out
    m_e, m_n = _metres_per_px(ortho)
    m_px = 0.5 * (abs(m_e) + abs(m_n))
    r_px = max(1, int(round(corridor_m / max(m_px, 1e-6))))
    for e, n in cam_xy:
        u, v = ortho.enu_to_px(float(e), float(n))
        ui, vi = int(round(u)), int(round(v))
        if 0 <= ui < w and 0 <= vi < h:
            cv2.circle(out, (ui, vi), r_px, 255, thickness=-1)
    return out


def build_street_support_mask(
    ortho: Ortho,
    *,
    building_dilate_m: float = DEFAULT_BUILDING_DILATE_M,
    cam_xy: np.ndarray | None = None,
    cam_corridor_m: float = DEFAULT_CAM_CORRIDOR_M,
) -> tuple[np.ndarray, dict[str, Any]]:
    """street_mask − dilate(roof∪yard) ∪ optional cam corridor.

    Punch hard so walls don’t get a ground slab under building footprints.
    """
    roof_m, yard_m, street_m = segment_roof_yard_mask(ortho.image)
    building = cv2.bitwise_or(roof_m, yard_m)
    punched = _dilate_mask_metres(building, ortho, building_dilate_m)
    support = cv2.subtract(street_m, punched)
    cam_m = cam_xy_mask(ortho, cam_xy, corridor_m=cam_corridor_m)
    if cam_corridor_m > 0 and int((cam_m > 0).sum()) > 0:
        # Corridor also punched by buildings so we don't slab under walls
        cam_keep = cv2.subtract(cam_m, punched)
        support = cv2.bitwise_or(support, cam_keep)
    # Light open to drop 1-px noise
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    support = cv2.morphologyEx(support, cv2.MORPH_OPEN, k3, iterations=1)
    info = {
        "street_px": int((street_m > 0).sum()),
        "building_px": int((building > 0).sum()),
        "punched_px": int((punched > 0).sum()),
        "cam_px": int((cam_m > 0).sum()),
        "support_px": int((support > 0).sum()),
        "building_dilate_m": float(building_dilate_m),
        "cam_corridor_m": float(cam_corridor_m),
    }
    return support, info


def _aabb_intersects_roof(
    aabb_enu: list[tuple[float, float]],
    roof_aabbs: list[list[tuple[float, float]]],
) -> bool:
    """True if street AABB overlaps any roof AABB (axis-aligned)."""
    if not roof_aabbs:
        return False
    es = [p[0] for p in aabb_enu]
    ns = [p[1] for p in aabb_enu]
    e0, e1 = min(es), max(es)
    n0, n1 = min(ns), max(ns)
    for ra in roof_aabbs:
        res = [p[0] for p in ra]
        rns = [p[1] for p in ra]
        re0, re1 = min(res), max(res)
        rn0, rn1 = min(rns), max(rns)
        if e0 < re1 and e1 > re0 and n0 < rn1 and n1 > rn0:
            return True
    return False


def _load_roof_aabbs(recon_dir: Path) -> list[list[tuple[float, float]]]:
    path = Path(recon_dir) / "roofs.json"
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []
    shells = payload.get("shells") if isinstance(payload, dict) else None
    if not isinstance(shells, list):
        return []
    out: list[list[tuple[float, float]]] = []
    for s in shells:
        if not isinstance(s, dict) or s.get("kind") != "roof":
            continue
        aabb = s.get("aabb_enu")
        if isinstance(aabb, list) and len(aabb) >= 4:
            try:
                out.append([(float(p[0]), float(p[1])) for p in aabb[:4]])
            except (TypeError, ValueError, IndexError):
                continue
    return out




def extract_street_regions(
    ortho: Ortho,
    support: np.ndarray,
    *,
    min_area_m2: float = MIN_AREA_M2,
    max_area_frac: float = MAX_AREA_FRAC,
    max_tile_m: float = MAX_TILE_SIDE_M,
    roof_aabbs: list[list[tuple[float, float]]] | None = None,
    min_support_frac: float = 0.20,
) -> list[dict[str, Any]]:
    """Grid-tile Ortho support into ≤ ``max_tile_m`` street AABBs.

    Exterior street_mask is usually one connected flood (RETR_EXTERNAL ≈ full
    tile). Contour-area reject would drop everything, so we tile the Ortho in
    ENU and keep cells with enough support coverage. Near-full-tile *failed*
    masks (support covers almost every pixel, no building punch) yield zero
    tiles via the global failed-mask gate below.
    """
    m_e, m_n = _metres_per_px(ortho)
    tile_m2 = abs((ortho.ee - ortho.sw) * (ortho.nn - ortho.sh))
    support_frac_global = float((support > 0).mean()) if support.size else 0.0
    # Failed mask: nearly the whole ortho is "street" with no building punch
    if support_frac_global >= max(0.92, max_area_frac + 0.5):
        log.warning(
            "sat street: near-full-tile support (frac=%.2f) — failed mask, no tiles",
            support_frac_global,
        )
        return []

    # Pixel step for ~max_tile_m cells
    step_e = max(2, int(round(max_tile_m / max(abs(m_e), 1e-6))))
    step_n = max(2, int(round(max_tile_m / max(abs(m_n), 1e-6))))
    regions: list[dict[str, Any]] = []
    idx = 0
    y = 0
    while y < ortho.h:
        th = min(step_n, ortho.h - y)
        x = 0
        while x < ortho.w:
            tw = min(step_e, ortho.w - x)
            crop = support[y : y + th, x : x + tw]
            frac = float((crop > 0).mean()) if crop.size else 0.0
            tile_area = abs(tw * m_e) * abs(th * m_n)
            x_next = x + tw
            if (
                frac >= min_support_frac
                and tile_area >= min_area_m2
                and tile_area <= max_area_frac * tile_m2
            ):
                # Tighten AABB to support pixels inside the cell (sharper UVs)
                ys, xs = np.where(crop > 0)
                if len(xs) >= 4:
                    x0 = int(xs.min())
                    x1 = int(xs.max()) + 1
                    y0 = int(ys.min())
                    y1 = int(ys.max()) + 1
                    # Keep a little padding for texture
                    pad = 1
                    tx = max(0, x + x0 - pad)
                    ty = max(0, y + y0 - pad)
                    tw2 = min(ortho.w - tx, (x + x1) - (tx) + pad)
                    th2 = min(ortho.h - ty, (y + y1) - (ty) + pad)
                    if tw2 >= 2 and th2 >= 2:
                        tight_area = abs(tw2 * m_e) * abs(th2 * m_n)
                        if tight_area >= min_area_m2:
                            corners_px = [
                                (tx, ty + th2),
                                (tx + tw2, ty + th2),
                                (tx + tw2, ty),
                                (tx, ty),
                            ]
                            aabb_enu = [
                                px_to_enu(ortho, u, v) for u, v in corners_px
                            ]
                            if roof_aabbs and _aabb_intersects_roof(
                                aabb_enu, roof_aabbs
                            ):
                                log.info(
                                    "sat street: skip street_%03d — AABB ∩ roof",
                                    idx,
                                )
                            else:
                                regions.append(
                                    {
                                        "id": f"street_{idx:03d}",
                                        "kind": "street",
                                        "area_m2": float(tight_area),
                                        "aabb_enu": aabb_enu,
                                        "bbox_px": (
                                            int(tx),
                                            int(ty),
                                            int(tw2),
                                            int(th2),
                                        ),
                                        "support_frac": float(frac),
                                    }
                                )
                                idx += 1
            x = x_next
        y += th
    return regions


def resolve_ground_z(
    project_root: Path,
    *,
    cam_u: float | None = None,
    camera_height_m: float = DEFAULT_CAMERA_HEIGHT_M,
) -> tuple[float, str]:
    """Pick ground_z for street tiles (never MA high-z; never BAG).

    Prefer product ``planes.json`` / yard low-band so street meets façade feet.
    Else ``median(cam_u) − camera_height_m`` (≈ cam_u−2.5).
    """
    root = Path(project_root)
    planes = root / "recon" / "planes.json"
    if planes.is_file():
        try:
            payload = json.loads(planes.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and payload.get("ground_z") is not None:
                return float(payload["ground_z"]), "planes_ground_z"
        except Exception:  # noqa: BLE001
            pass
    roofs = root / "recon" / "roofs.json"
    if roofs.is_file():
        try:
            payload = json.loads(roofs.read_text(encoding="utf-8"))
            shells = payload.get("shells") if isinstance(payload, dict) else None
            if isinstance(shells, list):
                yards = [
                    float(s["z"])
                    for s in shells
                    if isinstance(s, dict)
                    and s.get("kind") == "yard"
                    and s.get("z") is not None
                ]
                if yards:
                    # yards sit at ground+offset — peel offset toward true ground
                    return float(np.median(yards)) - 0.15, "yard_low_band"
        except Exception:  # noqa: BLE001
            pass
    if cam_u is not None and math.isfinite(float(cam_u)):
        gz = float(cam_u) - float(camera_height_m)
        return gz, "cam_u_minus_height"
    return 0.0, "default_zero"


def write_street_obj(
    dest_obj: Path,
    regions: list[dict[str, Any]],
    ortho: Ortho,
    *,
    tex_subdir: str = "textures",
) -> dict[str, Any]:
    """Write flat AABB street quads + sat-crop textures → street.obj / .mtl."""
    dest_obj = Path(dest_obj)
    dest_obj.parent.mkdir(parents=True, exist_ok=True)
    tex_dir = dest_obj.parent / tex_subdir
    tex_dir.mkdir(parents=True, exist_ok=True)
    mtl_name = dest_obj.with_suffix(".mtl").name

    materials: list[dict[str, Any]] = []
    faces: list[tuple[int, int, int, int, int]] = []
    verts: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []

    for i, reg in enumerate(regions):
        z = float(reg["z"])
        aabb = reg["aabb_enu"]
        quad = [
            (aabb[0][0], aabb[0][1], z),
            (aabb[1][0], aabb[1][1], z),
            (aabb[2][0], aabb[2][1], z),
            (aabb[3][0], aabb[3][1], z),
        ]
        tex_rel = f"{tex_subdir}/street_{reg['id']}.jpg"
        tex_path = dest_obj.parent / tex_rel
        ok = _crop_sat_texture(ortho, reg["bbox_px"], tex_path)
        mat_name = f"street_{reg['id']}"
        materials.append(
            {
                "name": mat_name,
                "map": tex_rel if ok else None,
                "kd": STREET_KD,
            }
        )
        base = len(verts)
        verts.extend(quad)
        uvs.extend([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
        faces.append((base, base + 1, base + 2, base + 3, i))

    mtl_path = dest_obj.with_suffix(".mtl")
    with mtl_path.open("w", encoding="ascii") as fh:
        fh.write("# ps1-hood sat street/ground materials\n")
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
        fh.write("# ps1-hood sat-locked street/ground shells (Ortho ENU XY)\n")
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


def build_sat_street(
    project_root: Path,
    *,
    dest_obj: Path | None = None,
    min_area_m2: float = MIN_AREA_M2,
    building_dilate_m: float = DEFAULT_BUILDING_DILATE_M,
    cam_corridor_m: float = DEFAULT_CAM_CORRIDOR_M,
    camera_height_m: float = DEFAULT_CAMERA_HEIGHT_M,
    max_tile_m: float = MAX_TILE_SIDE_M,
    bak: bool = True,
) -> dict[str, Any]:
    """End-to-end: support mask → tiles → ground_z → street.obj.

    Idempotent overwrite of street.* only; never touches façades / roofs / planes.
    """
    root = Path(project_root)
    recon = root / "recon"
    dest = Path(dest_obj) if dest_obj else recon / "street.obj"

    # Snapshot sacred product files — must be unchanged after write
    sacred = {
        "facades.obj": recon / "facades.obj",
        "facades.mtl": recon / "facades.mtl",
        "roofs.obj": recon / "roofs.obj",
        "roofs.mtl": recon / "roofs.mtl",
        "roofs.json": recon / "roofs.json",
        "planes.json": recon / "planes.json",
    }
    sacred_mtime = {
        k: (p.stat().st_mtime_ns if p.is_file() else None)
        for k, p in sacred.items()
    }

    if bak and dest.is_file():
        bak_path = dest.with_suffix(dest.suffix + ".bak")
        shutil.copy2(dest, bak_path)
        log.info("sat street: backed up %s → %s", dest.name, bak_path.name)

    ortho = load_ortho_for_run(root)
    cam_xy, cam_u = load_cam_xy_enu(root)
    support, support_info = build_street_support_mask(
        ortho,
        building_dilate_m=building_dilate_m,
        cam_xy=cam_xy,
        cam_corridor_m=cam_corridor_m,
    )
    if support_info["support_px"] < 50:
        raise SatStreetError(
            "empty street support after roof∪yard punch; "
            "check ortho / segmentation / dilate knobs"
        )

    roof_aabbs = _load_roof_aabbs(recon)
    regions = extract_street_regions(
        ortho,
        support,
        min_area_m2=min_area_m2,
        max_tile_m=max_tile_m,
        roof_aabbs=roof_aabbs,
    )
    if not regions:
        raise SatStreetError(
            "no street tiles from support mask (all rejected tiny/full-tile/"
            "roof-overlap); check min-area / dilate"
        )

    ground_z, z_src = resolve_ground_z(
        root, cam_u=cam_u, camera_height_m=camera_height_m
    )
    for reg in regions:
        reg["z"] = float(ground_z)
        reg["z_source"] = z_src
        reg["ground_z"] = float(ground_z)
        reg["cam_u_median"] = float(cam_u) if cam_u is not None else None

    write_meta = write_street_obj(dest, regions, ortho)

    shells_json = {
        "frame": "ENU",
        "source": "sat_ortho_street_mask",
        "kind": "street",
        "ground_z": float(ground_z),
        "z_source": z_src,
        "cam_u_median": float(cam_u) if cam_u is not None else None,
        "camera_height_m": float(camera_height_m),
        "building_dilate_m": float(building_dilate_m),
        "cam_corridor_m": float(cam_corridor_m),
        "max_tile_m": float(max_tile_m),
        "support": support_info,
        "shells": [
            {
                "id": r["id"],
                "kind": "street",
                "z": r["z"],
                "z_source": r["z_source"],
                "area_m2": r["area_m2"],
                "aabb_enu": r["aabb_enu"],
            }
            for r in regions
        ],
    }
    shells_path = recon / "street.json"
    shells_path.write_text(json.dumps(shells_json, indent=2), encoding="utf-8")

    # Sacred-file guard
    for name, mtime in sacred_mtime.items():
        p = sacred[name]
        now = p.stat().st_mtime_ns if p.is_file() else None
        if mtime != now:
            raise SatStreetError(
                f"sacred product clobbered: {name} mtime changed during street build"
            )

    meta = {
        **write_meta,
        "n_street": len(regions),
        "ground_z": float(ground_z),
        "z_source": z_src,
        "cam_u_median": float(cam_u) if cam_u is not None else None,
        "building_dilate_m": float(building_dilate_m),
        "cam_corridor_m": float(cam_corridor_m),
        "street_json": str(shells_path),
        "support_px": support_info["support_px"],
        "ortho_enu": {
            "sw": ortho.sw,
            "sh": ortho.sh,
            "ee": ortho.ee,
            "nn": ortho.nn,
        },
    }
    log.info(
        "sat street: wrote %s  tiles=%d textured=%d  ground_z=%.3f (%s)  "
        "support_px=%d dilate=%.2f corridor=%.1f",
        dest,
        len(regions),
        write_meta["textured"],
        ground_z,
        z_src,
        support_info["support_px"],
        building_dilate_m,
        cam_corridor_m,
    )
    return meta
