"""Vertical-plane facades + ground quad — textured when cameras are available."""

from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ps1_hood.geo import LocalFrame, camera_rotation_cv


def _read_ply_xyz(path: Path) -> np.ndarray:
    """Read XYZ from ascii or binary little-endian PLY (x/y/z floats first)."""
    raw = path.read_bytes()
    header_end = raw.find(b"end_header")
    if header_end < 0:
        raise RuntimeError(f"not a PLY: {path}")
    header = raw[: header_end].decode("ascii", errors="replace")
    body = raw[header_end + len(b"end_header") :]
    if body.startswith(b"\n"):
        body = body[1:]
    elif body.startswith(b"\r\n"):
        body = body[2:]

    fmt = "ascii"
    n_verts = 0
    props: list[tuple[str, str]] = []
    in_vertex = False
    for line in header.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element" and parts[1] == "vertex":
            n_verts = int(parts[2])
            in_vertex = True
        elif parts[0] == "element":
            in_vertex = False
        elif in_vertex and parts[0] == "property":
            props.append((parts[1], parts[2]))

    if n_verts <= 0:
        return np.zeros((0, 3), dtype=np.float64)

    if fmt == "ascii":
        pts = []
        for line in body.decode("ascii", errors="replace").splitlines():
            parts = line.split()
            if len(parts) < 3:
                continue
            pts.append((float(parts[0]), float(parts[1]), float(parts[2])))
            if len(pts) >= n_verts:
                break
        return np.array(pts, dtype=np.float64)

    # binary_little_endian / binary_big_endian
    type_map = {
        "char": "b",
        "uchar": "B",
        "int8": "b",
        "uint8": "B",
        "short": "h",
        "ushort": "H",
        "int16": "h",
        "uint16": "H",
        "int": "i",
        "uint": "I",
        "int32": "i",
        "uint32": "I",
        "float": "f",
        "float32": "f",
        "double": "d",
        "float64": "d",
    }
    endian = "<" if "little" in fmt else ">"
    fmt_str = endian + "".join(type_map[t] for t, _ in props)
    size = struct.calcsize(fmt_str)
    # locate x,y,z property indices
    names = [n for _, n in props]
    try:
        ix, iy, iz = names.index("x"), names.index("y"), names.index("z")
    except ValueError as exc:
        raise RuntimeError(f"PLY missing x/y/z: {path}") from exc
    pts = np.empty((n_verts, 3), dtype=np.float64)
    for i in range(n_verts):
        chunk = body[i * size : (i + 1) * size]
        if len(chunk) < size:
            return pts[:i]
        vals = struct.unpack(fmt_str, chunk)
        pts[i] = (vals[ix], vals[iy], vals[iz])
    return pts


def _plane_quad(pl: dict) -> list[tuple[float, float, float]]:
    nx, ny, d = pl["nx"], pl["ny"], pl["d"]
    tx, ty = -ny, nx
    c = np.array(pl["min"][:2]) * 0.5 + np.array(pl["max"][:2]) * 0.5
    c = c - (c @ np.array([nx, ny]) - d) * np.array([nx, ny])
    half = 0.5 * np.linalg.norm(np.array(pl["max"][:2]) - np.array(pl["min"][:2]))
    gz = float(pl.get("ground_z", 0.0))
    z0 = max(gz, float(pl["min"][2]))
    z1 = max(z0 + 2.0, float(pl["max"][2]))
    p1 = (float(c[0] - tx * half), float(c[1] - ty * half), z0)
    p2 = (float(c[0] + tx * half), float(c[1] + ty * half), z0)
    p3 = (float(c[0] + tx * half), float(c[1] + ty * half), z1)
    p4 = (float(c[0] - tx * half), float(c[1] - ty * half), z1)
    return [p1, p2, p3, p4]


def _K(width: int, height: int, fov_deg: float) -> np.ndarray:
    f = 0.5 * width / math.tan(math.radians(fov_deg) / 2.0)
    return np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1.0]], dtype=np.float64)


def _project_points(pose: dict[str, Any], width: int, height: int, pts: np.ndarray) -> np.ndarray:
    """Project ENU points → image pixels (N×2). Behind-camera → NaN."""
    R = np.array(camera_rotation_cv(pose["heading"], pose["pitch"]), dtype=np.float64)
    C = np.array([pose["e"], pose["n"], pose["u"]], dtype=np.float64)
    Rcw = R.T
    t = -Rcw @ C
    K = _K(width, height, float(pose["fov"]))
    cam = (Rcw @ pts.T).T + t
    out = np.full((len(pts), 2), np.nan, dtype=np.float64)
    front = cam[:, 2] > 0.2
    if not np.any(front):
        return out
    uv = (K @ cam[front].T).T
    uv = uv[:, :2] / np.clip(uv[:, 2:3], 1e-8, None)
    out[front] = uv
    return out


def _forward_xy(pose: dict[str, Any]) -> np.ndarray:
    R = np.array(camera_rotation_cv(pose["heading"], pose["pitch"]), dtype=np.float64)
    fwd = R[:, 2]
    return fwd[:2]


def _pick_frontal_camera(
    pl: dict,
    quad: list[tuple[float, float, float]],
    frames: list[dict[str, Any]],
    *,
    min_frontal: float = 0.20,
) -> dict[str, Any] | None:
    """Camera whose view is most frontal to the plane and not too grazing."""
    nx, ny = pl["nx"], pl["ny"]
    n_xy = np.array([nx, ny], dtype=np.float64)
    center = np.mean(np.array(quad), axis=0)
    best: dict[str, Any] | None = None
    best_score = -1.0
    for fr in frames:
        cam = np.array([fr["e"], fr["n"], fr["u"]], dtype=np.float64)
        # Orient plane normal toward the camera.
        n = n_xy.copy()
        if n @ (cam[:2] - center[:2]) < 0:
            n = -n
        fwd = _forward_xy(fr)
        fn = float(np.linalg.norm(fwd))
        if fn < 1e-6:
            continue
        fwd = fwd / fn
        # Frontal: camera looks into the facade (toward -n).
        frontal = float((-fwd) @ n)
        if frontal < min_frontal:
            continue
        img = cv2.imread(fr["path"], cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        uv_c = _project_points(fr, w, h, center.reshape(1, 3))[0]
        if not np.isfinite(uv_c).all():
            continue
        uv_q = _project_points(fr, w, h, np.array(quad, dtype=np.float64))
        finite = np.isfinite(uv_q).all(axis=1)
        in_frame = finite & (uv_q[:, 0] >= -20) & (uv_q[:, 1] >= -20) & (uv_q[:, 0] < w + 20) & (uv_q[:, 1] < h + 20)
        coverage = float(in_frame.sum()) / max(len(quad), 1)
        if coverage < 0.5:
            continue
        # Soft preference for nearer, more covered facades.
        dist = float(np.linalg.norm(cam - center))
        score = frontal * (0.5 + 0.5 * coverage) * (1.0 / (1.0 + 0.02 * dist))
        if score > best_score:
            best_score = score
            best = {**fr, "_img": img, "_frontal": frontal}
    return best


def _next_pot(n: int) -> int:
    p = 64
    while p < n and p < 1024:
        p *= 2
    return p


def _warp_facade_texture(
    quad: list[tuple[float, float, float]],
    frame: dict[str, Any],
    dest: Path,
) -> bool:
    img = frame.get("_img")
    if img is None:
        img = cv2.imread(frame["path"], cv2.IMREAD_COLOR)
    if img is None:
        return False
    h, w = img.shape[:2]
    pts = np.array(quad, dtype=np.float64)
    uv = _project_points(frame, w, h, pts)
    if not np.isfinite(uv).all():
        return False
    # Allow slight out-of-frame corners; clamp for warp stability.
    margin = 40.0
    if (
        (uv[:, 0] < -margin).any()
        or (uv[:, 1] < -margin).any()
        or (uv[:, 0] > w + margin).any()
        or (uv[:, 1] > h + margin).any()
    ):
        return False
    width_m = float(np.linalg.norm(pts[1][:2] - pts[0][:2]))
    height_m = float(abs(pts[3][2] - pts[0][2]))
    aspect = max(width_m, 0.5) / max(height_m, 0.5)
    th = _next_pot(int(round(256 * max(1.0, 1.0 / aspect))))
    tw = _next_pot(int(round(th * aspect)))
    tw = int(np.clip(tw, 64, 1024))
    th = int(np.clip(th, 64, 1024))
    src = uv.astype(np.float32)
    # Keep corners near the frame so getPerspectiveTransform stays stable.
    src[:, 0] = np.clip(src[:, 0], -margin, w + margin)
    src[:, 1] = np.clip(src[:, 1], -margin, h + margin)
    dst = np.array(
        [[0, th - 1], [tw - 1, th - 1], [tw - 1, 0], [0, 0]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (tw, th), flags=cv2.INTER_LINEAR)
    dest.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(dest), warped, [int(cv2.IMWRITE_JPEG_QUALITY), 88]))


def _ground_satellite_texture(
    xyz: np.ndarray,
    dest: Path,
    satellite: dict[str, Any] | None,
    frame: LocalFrame | None,
) -> bool:
    if not satellite or frame is None:
        return False
    path = Path(satellite.get("path") or "")
    if not path.is_file():
        return False
    try:
        from ps1_hood.capture.satellite import Ortho
        from ps1_hood.geo import BBox

        ortho = Ortho(cv2.imread(str(path), cv2.IMREAD_COLOR), BBox.from_dict(satellite["bbox"]), frame)
    except Exception:
        return False
    if ortho.image is None:
        return False
    mn, mx = xyz.min(axis=0), xyz.max(axis=0)
    # Sample the AABB corners into pixel space and crop.
    corners = [
        ortho.enu_to_px(mn[0], mn[1]),
        ortho.enu_to_px(mx[0], mn[1]),
        ortho.enu_to_px(mx[0], mx[1]),
        ortho.enu_to_px(mn[0], mx[1]),
    ]
    us = [c[0] for c in corners]
    vs = [c[1] for c in corners]
    x0 = int(max(0, math.floor(min(us))))
    x1 = int(min(ortho.w, math.ceil(max(us)) + 1))
    y0 = int(max(0, math.floor(min(vs))))
    y1 = int(min(ortho.h, math.ceil(max(vs)) + 1))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return False
    crop = ortho.image[y0:y1, x0:x1]
    dest.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(dest), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 88]))


def _voxel_downsample_xyz(xyz: np.ndarray, voxel: float = 0.20) -> np.ndarray:
    if len(xyz) == 0:
        return xyz
    keys = np.floor(xyz / voxel).astype(np.int64)
    view = np.ascontiguousarray(keys).view([("", keys.dtype)] * 3).ravel()
    _, idx = np.unique(view, return_index=True)
    return xyz[idx]


def _fit_ground_z(xyz: np.ndarray) -> float:
    """Photo-seeded ground height: robust low percentile of near-horizontal band."""
    if len(xyz) < 20:
        return 0.0
    z = xyz[:, 2]
    lo, hi = np.percentile(z, [5, 40])
    band = xyz[(z >= lo - 0.5) & (z <= hi + 0.5)]
    if len(band) < 10:
        return float(np.percentile(z, 10))
    # RANSAC horizontal plane z = const
    rng = np.random.default_rng(1)
    best_z, best_n = float(np.median(band[:, 2])), 0
    for _ in range(40):
        z0 = float(band[int(rng.integers(0, len(band))), 2])
        inl = np.abs(band[:, 2] - z0) < 0.35
        n = int(inl.sum())
        if n > best_n:
            best_n = n
            best_z = float(np.median(band[inl, 2]))
    return best_z


def extract_facades(
    ply_path: Path,
    dest_obj: Path,
    n_planes: int = 12,
    *,
    frames: list[dict[str, Any]] | None = None,
    satellite: dict[str, Any] | None = None,
    local_frame: LocalFrame | None = None,
    min_points: int = 80,
) -> dict:
    """RANSAC vertical planes seeded by the *photo* cloud (not OSM/BAG).

    Voxel-downsample → cluster vertical planes + ground from photo extents →
    texture with existing SV warp. Cadastral shells are not used.
    """
    xyz = _read_ply_xyz(ply_path)
    if len(xyz) < min_points:
        raise RuntimeError(
            f"not enough photo points for facade extraction ({len(xyz)} < {min_points})"
        )
    xyz = _voxel_downsample_xyz(xyz, 0.20)
    ground_z = _fit_ground_z(xyz)
    remaining = xyz.copy()
    planes: list[dict] = []
    rng = np.random.default_rng(0)
    min_inliers = max(25, min(60, len(xyz) // 20))
    for _ in range(n_planes):
        if len(remaining) < max(40, min_inliers):
            break
        best_inliers = None
        best_n = None
        best_d = None
        for _try in range(120):
            i0, i1 = rng.integers(0, len(remaining), size=2)
            p0, p1 = remaining[i0], remaining[i1]
            v = p1[:2] - p0[:2]
            if np.linalg.norm(v) < 0.2:
                continue
            nrm = np.array([-v[1], v[0]], dtype=np.float64)
            nrm /= np.linalg.norm(nrm)
            d = float(nrm @ p0[:2])
            dist = np.abs(remaining[:, :2] @ nrm - d)
            # Prefer facade-height points (above ground)
            height_ok = remaining[:, 2] > (ground_z + 0.4)
            inl = (dist < 0.50) & height_ok
            if best_inliers is None or int(inl.sum()) > int(best_inliers.sum()):
                best_inliers = inl
                best_n = nrm
                best_d = d
        if best_inliers is None or int(best_inliers.sum()) < min_inliers:
            break
        pts = remaining[best_inliers]
        # Quads from photo point extents (not cadastral footprints)
        planes.append(
            {
                "nx": float(best_n[0]),
                "ny": float(best_n[1]),
                "d": float(best_d),
                "min": pts.min(axis=0).tolist(),
                "max": pts.max(axis=0).tolist(),
                "count": int(len(pts)),
                "ground_z": ground_z,
            }
        )
        remaining = remaining[~best_inliers]

    dest_obj.parent.mkdir(parents=True, exist_ok=True)
    tex_dir = dest_obj.parent / "textures"
    if tex_dir.is_dir():
        for old in tex_dir.glob("*.jpg"):
            old.unlink(missing_ok=True)
    tex_dir.mkdir(parents=True, exist_ok=True)
    materials: list[dict[str, Any]] = []

    # Ground
    ground_tex = tex_dir / "ground.jpg"
    has_ground_tex = _ground_satellite_texture(xyz, ground_tex, satellite, local_frame)
    materials.append(
        {
            "name": "ground",
            "map": "textures/ground.jpg" if has_ground_tex else None,
            "kd": (0.35, 0.38, 0.32),
        }
    )

    textured = 0
    for i, pl in enumerate(planes):
        quad = _plane_quad(pl)
        mat_name = f"facade_{i:02d}"
        map_rel = None
        if frames:
            cam = _pick_frontal_camera(pl, quad, frames)
            if cam is not None:
                tex_path = tex_dir / f"facade_{i:02d}.jpg"
                if _warp_facade_texture(quad, cam, tex_path):
                    map_rel = f"textures/facade_{i:02d}.jpg"
                    textured += 1
        materials.append(
            {
                "name": mat_name,
                "map": map_rel,
                "kd": (0.55, 0.50, 0.42),
                "quad": quad,
            }
        )

    mtl_path = dest_obj.with_suffix(".mtl")
    _write_mtl(mtl_path, materials)
    _write_obj(dest_obj, planes, xyz, materials, mtl_path.name)
    return {
        "path": str(dest_obj),
        "mtl": str(mtl_path),
        "planes": len(planes),
        "textured": textured,
        "ground_textured": has_ground_tex,
        "source": "photo_ransac",
        "ground_z": ground_z,
        "points_used": int(len(xyz)),
    }


def _write_mtl(path: Path, materials: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="ascii") as fh:
        fh.write("# ps1-hood facade materials\n")
        for mat in materials:
            fh.write(f"newmtl {mat['name']}\n")
            kd = mat.get("kd") or (0.7, 0.7, 0.7)
            fh.write(f"Kd {kd[0]:.3f} {kd[1]:.3f} {kd[2]:.3f}\n")
            fh.write("Ka 0.050 0.050 0.050\n")
            fh.write("d 1.0\n")
            fh.write("illum 1\n")
            if mat.get("map"):
                fh.write(f"map_Kd {mat['map']}\n")
            fh.write("\n")


def _write_obj(
    path: Path,
    planes: list[dict],
    xyz: np.ndarray,
    materials: list[dict[str, Any]],
    mtl_name: str,
) -> None:
    # ground quad from point AABB
    mn, mx = xyz.min(axis=0), xyz.max(axis=0)
    gz = float(min(0.0, mn[2])) if len(xyz) else 0.0
    # Prefer near-ground band from photo cloud
    if len(xyz) >= 20:
        gz = float(np.percentile(xyz[:, 2], 8))
    verts: list[tuple[float, float, float]] = [
        (mn[0], mn[1], gz),
        (mx[0], mn[1], gz),
        (mx[0], mx[1], gz),
        (mn[0], mx[1], gz),
    ]
    uvs: list[tuple[float, float]] = [
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
    ]
    # face: (v0,v1,v2,v3, mat_index) — 0-based vertex indices into verts
    faces: list[tuple[int, int, int, int, int]] = [(0, 1, 2, 3, 0)]

    for i, pl in enumerate(planes):
        mat = materials[i + 1]
        quad = mat.get("quad") or _plane_quad(pl)
        base = len(verts)
        verts.extend(quad)
        uvs.extend([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
        faces.append((base, base + 1, base + 2, base + 3, i + 1))

    with path.open("w", encoding="ascii") as fh:
        fh.write("# ps1-hood facade blockout (textured)\n")
        fh.write(f"mtllib {mtl_name}\n")
        for v in verts:
            fh.write(f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n")
        for uv in uvs:
            fh.write(f"vt {uv[0]:.4f} {uv[1]:.4f}\n")
        cur_mat = None
        for v0, v1, v2, v3, mi in faces:
            name = materials[mi]["name"]
            if name != cur_mat:
                fh.write(f"usemtl {name}\n")
                cur_mat = name
            # OBJ is 1-indexed; vt index matches v index for our layout
            fh.write(
                f"f {v0 + 1}/{v0 + 1} {v1 + 1}/{v1 + 1} "
                f"{v2 + 1}/{v2 + 1} {v3 + 1}/{v3 + 1}\n"
            )
