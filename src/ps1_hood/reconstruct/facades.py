"""Photo-consistent vertical façades + ground quad (known poses, ZNCC)."""

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



def _facade_product_looks_nonempty(dest_obj: Path) -> bool:
    """True if existing façades.obj / planes.json / facade textures look like real product."""
    dest_obj = Path(dest_obj)
    if dest_obj.is_file() and dest_obj.stat().st_size > 400:
        body = dest_obj.read_text(encoding="ascii", errors="ignore")
        # Prior photo façades have façade materials / multiple usemtl
        if "usemtl facade_" in body or body.count("usemtl") >= 2:
            return True
        if sum(1 for line in body.splitlines() if line.startswith("v ")) >= 12:
            return True
    planes_json = dest_obj.parent / "planes.json"
    if planes_json.is_file() and planes_json.stat().st_size > 80:
        try:
            import json

            payload = json.loads(planes_json.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and len(payload.get("planes") or []) > 0:
                return True
        except Exception:  # noqa: BLE001
            pass
    tex_dir = dest_obj.parent / "textures"
    if tex_dir.is_dir() and any(tex_dir.glob("facade_*.jpg")):
        return True
    return False


def _read_facade_quality(dest_obj: Path) -> dict[str, Any] | None:
    """Quality tuple from existing product: textured count, mean_zncc, plane_count."""
    import json

    dest_obj = Path(dest_obj)
    planes_json = dest_obj.parent / "planes.json"
    tex_dir = dest_obj.parent / "textures"
    file_tex = (
        len(list(tex_dir.glob("facade_*.jpg")))
        if tex_dir.is_dir()
        else 0
    )
    if not planes_json.is_file():
        if file_tex <= 0 and not _facade_product_looks_nonempty(dest_obj):
            return None
        return {
            "textured": int(file_tex),
            "mean_zncc": None,
            "plane_count": int(file_tex),
        }
    try:
        payload = json.loads(planes_json.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        if file_tex <= 0:
            return None
        return {
            "textured": int(file_tex),
            "mean_zncc": None,
            "plane_count": int(file_tex),
        }
    if not isinstance(payload, dict):
        return None
    planes = payload.get("planes") or []
    if not isinstance(planes, list):
        planes = []
    textured = sum(1 for p in planes if isinstance(p, dict) and p.get("texture"))
    if textured <= 0:
        textured = int(file_tex)
    gates = payload.get("gates") if isinstance(payload.get("gates"), dict) else {}
    mean_zncc = gates.get("mean_zncc")
    if mean_zncc is None:
        znccs = [
            float(p["zncc"])
            for p in planes
            if isinstance(p, dict) and p.get("zncc") is not None
        ]
        mean_zncc = float(np.mean(znccs)) if znccs else None
    else:
        mean_zncc = float(mean_zncc)
    if len(planes) <= 0 and textured <= 0 and not _facade_product_looks_nonempty(dest_obj):
        return None
    return {
        "textured": int(textured),
        "mean_zncc": mean_zncc,
        "plane_count": int(len(planes)),
    }


def _is_strictly_better(
    new_q: dict[str, Any],
    prev_q: dict[str, Any],
    *,
    min_textured: int = 3,
    min_mean_zncc: float = 0.35,
    zncc_eps: float = 0.02,
) -> tuple[bool, str]:
    """Promote only if new is strictly better than existing product.

    Order: more textured → more planes (same textured) → higher mean_zncc
    (same textured+planes, by > eps). Never demote a product that already
    meets soft floors (textured≥min or mean_zncc≥min).
    """
    nt = int(new_q.get("textured") or 0)
    pt = int(prev_q.get("textured") or 0)
    np_ = int(new_q.get("plane_count") or new_q.get("n_planes") or 0)
    pp_ = int(prev_q.get("plane_count") or prev_q.get("n_planes") or 0)
    nm = new_q.get("mean_zncc")
    pm = prev_q.get("mean_zncc")

    if pp_ <= 0 and pt <= 0:
        return True, "no existing product"

    old_good = pt >= min_textured or (
        pm is not None and float(pm) >= float(min_mean_zncc)
    )
    if old_good:
        if nt < pt:
            return False, f"fewer textured ({nt}<{pt})"
        if nt == pt and np_ < pp_:
            return False, f"fewer planes ({np_}<{pp_})"
        if (
            nt == pt
            and np_ == pp_
            and pm is not None
            and (nm is None or float(nm) <= float(pm) + float(zncc_eps))
        ):
            return False, (
                f"not higher mean_zncc ({nm} vs {pm})"
            )

    if nt > pt:
        return True, "more textured walls"
    if nt == pt and np_ > pp_:
        return True, "more planes"
    if (
        nt == pt
        and np_ == pp_
        and nm is not None
        and pm is not None
        and float(nm) > float(pm) + float(zncc_eps)
    ):
        return True, "higher mean_zncc"
    if not old_good and (np_ > 0 or nt > 0):
        return True, "replace empty/weak product"
    return False, "candidate not strictly better"


def _rewrite_texture_prefix(text: str, old: str, new: str) -> str:
    return text.replace(old, new)


def _promote_candidate_facades(
    dest_obj: Path,
    *,
    cand_obj: Path,
    cand_mtl: Path,
    cand_json: Path,
    cand_tex_dir: Path,
) -> None:
    """Move candidate product into primary façades.obj / planes.json / textures/."""
    import json
    import shutil

    dest_obj = Path(dest_obj)
    parent = dest_obj.parent
    tex_dir = parent / "textures"
    mtl_path = dest_obj.with_suffix(".mtl")
    planes_json = parent / "planes.json"
    tex_dir.mkdir(parents=True, exist_ok=True)

    for old_tex in tex_dir.glob("facade_*.jpg"):
        old_tex.unlink(missing_ok=True)

    if cand_tex_dir.is_dir():
        for src in sorted(cand_tex_dir.glob("facade_*.jpg")):
            shutil.copy2(src, tex_dir / src.name)
        shutil.rmtree(cand_tex_dir, ignore_errors=True)

    old_prefix = "textures/candidate/"
    new_prefix = "textures/"
    if cand_mtl.is_file():
        mtl_path.write_text(
            _rewrite_texture_prefix(
                cand_mtl.read_text(encoding="ascii", errors="ignore"),
                old_prefix,
                new_prefix,
            ),
            encoding="ascii",
        )
        cand_mtl.unlink(missing_ok=True)
    if cand_obj.is_file():
        body = cand_obj.read_text(encoding="ascii", errors="ignore")
        body = _rewrite_texture_prefix(body, old_prefix, new_prefix)
        # mtllib may point at candidate mtl name
        body = body.replace(cand_mtl.name, mtl_path.name)
        body = body.replace(dest_obj.stem + ".candidate.mtl", mtl_path.name)
        dest_obj.write_text(body, encoding="ascii")
        cand_obj.unlink(missing_ok=True)
    if cand_json.is_file():
        payload = json.loads(cand_json.read_text(encoding="utf-8"))
        for pl in payload.get("planes") or []:
            if isinstance(pl, dict) and isinstance(pl.get("texture"), str):
                pl["texture"] = pl["texture"].replace(old_prefix, new_prefix)
        planes_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        cand_json.unlink(missing_ok=True)


def extract_facades(
    ply_path: Path | None,
    dest_obj: Path,
    n_planes: int = 12,
    *,
    frames: list[dict[str, Any]] | None = None,
    satellite: dict[str, Any] | None = None,
    local_frame: LocalFrame | None = None,
    min_points: int = 80,
    zncc_accept: float | None = None,
    planarize: bool | None = None,
    voxel_m: float = 0.08,
    plane_dist_m: float = 0.08,
    keep_previous_on_fail: bool = True,
    fallback_heading: bool = True,
    hybrid_heading: bool = True,
    split_trigger_width_m: float | None = None,
    split_window_m: float | None = None,
    split_overlap_m: float | None = None,
    max_heading_seeds: int = 24,
) -> dict:
    """Photo-consistent vertical façades under known poses.

    Path α (planarize): when ``planarize=True`` or auto (dense PLY ≳50k pts),
    seed vertical planes from MapAnything ENU cloud via Open3D/numpy
    ``segment_plane`` peel, then ZNCC-gate + ortho bake (Milestone A).

    With ``hybrid_heading`` (default on for Path α), Milestone A
    heading×distance seeds are injected into the same ``score_planar_hyps``
    pass so NMS keeps a union(A, MA). Promote still uses
    ``_is_strictly_better`` on the union bake metrics.

    If hybrid/Path α keeps 0 planes and ``fallback_heading``, retry full
    Milestone A search on the same run (``photo_consistency_fallback``).

    Fail-loud: never invent flow-RANSAC or OSM/BAG walls. When accepts==0 and
    ``keep_previous_on_fail`` (default), do **not** clobber non-empty product
    ``facades.obj`` / ``.mtl`` / ``textures/facade_*.jpg`` / ``planes.json`` —
    write diagnostics to ``*.failed`` instead.

    Quality gate: with ``keep_previous_on_fail``, **promote only if strictly
    better** than the existing product (more textured walls, or same textured
    with more planes / higher ``mean_zncc``). Weaker or equal results —
    including Milestone A fallback after Path α 0 accepts — go to
    ``*.candidate`` and the live product is kept.
    """
    import logging

    from ps1_hood.reconstruct.photo_planes import (
        hypothesize_vertical_planes,
        plane_dict_for_obj,
        search_photo_consistent_planes,
    )
    from ps1_hood.reconstruct.planarize import (
        DEFAULT_SPLIT_OVERLAP_M,
        DEFAULT_SPLIT_TRIGGER_WIDTH_M,
        DEFAULT_SPLIT_WINDOW_M,
        DEFAULT_ZNCC_ACCEPT_MA,
        PLANARIZE_AUTO_MIN_POINTS,
        planes_from_mapanything_ply,
        score_planar_hyps,
        write_planes_json,
    )

    log = logging.getLogger(__name__)
    frames = list(frames or [])
    dest_obj = Path(dest_obj)

    xyz_raw = np.zeros((0, 3), dtype=np.float64)
    if ply_path is not None and Path(ply_path).is_file():
        try:
            xyz_raw = _read_ply_xyz(Path(ply_path))
        except Exception as exc:  # noqa: BLE001
            log.warning("facades: could not read seed PLY %s (%s)", ply_path, exc)

    use_planarize = planarize if planarize is not None else (
        len(xyz_raw) >= PLANARIZE_AUTO_MIN_POINTS
    )
    if zncc_accept is None:
        zncc_accept = DEFAULT_ZNCC_ACCEPT_MA if use_planarize else 0.35

    residual_pts = 0
    source_tag = "photo_consistency"
    path_alpha_attempted = False

    if len(xyz_raw) >= 20:
        xyz = _voxel_downsample_xyz(xyz_raw, 0.20 if not use_planarize else voxel_m)
        ground_z = _fit_ground_z(xyz)
    else:
        xyz = xyz_raw
        ground_z = 0.0
        if frames:
            ground_z = float(np.median([float(f.get("u") or 0.0) for f in frames])) - 2.5

    if use_planarize and ply_path is not None and Path(ply_path).is_file() and len(xyz_raw) >= 100:
        path_alpha_attempted = True
        cam_c = (
            np.mean(
                [[float(f["e"]), float(f["n"]), float(f["u"])] for f in frames],
                axis=0,
            )
            if frames
            else xyz.mean(axis=0)
        )
        hyps, ground, residual_pts = planes_from_mapanything_ply(
            Path(ply_path),
            cam_c,
            xyz=xyz_raw,
            ground_z=ground_z,
            voxel=voxel_m,
            distance_threshold=plane_dist_m,
            max_planes=max(n_planes, 24),
        )
        if ground is not None and ground.get("z") is not None:
            ground_z = float(ground["z"])

        split_trigger = (
            DEFAULT_SPLIT_TRIGGER_WIDTH_M
            if split_trigger_width_m is None
            else float(split_trigger_width_m)
        )
        split_window = (
            DEFAULT_SPLIT_WINDOW_M if split_window_m is None else float(split_window_m)
        )
        split_overlap = (
            DEFAULT_SPLIT_OVERLAP_M
            if split_overlap_m is None
            else float(split_overlap_m)
        )

        seed_hyps: list[dict] = []
        if hybrid_heading and frames:
            seed_hyps = hypothesize_vertical_planes(
                frames,
                xyz if len(xyz) >= 30 else None,
                ground_z=ground_z,
                max_heading_seeds=int(max_heading_seeds),
            )
            log.info(
                "facades Path α hybrid: injecting %s A heading seeds into scorer",
                len(seed_hyps),
            )

        accepted = score_planar_hyps(
            hyps,
            frames,
            zncc_accept=float(zncc_accept),
            max_keep=n_planes,
            seed_hyps=seed_hyps or None,
            split_trigger_width_m=split_trigger,
            split_window_m=split_window,
            split_overlap_m=split_overlap,
        )

        def _is_ma(src: str | None) -> bool:
            s = (src or "ma_segment").lower()
            return s.startswith("ma_") or s in {"ma_segment", "mapanything_planarize"}

        def _is_a(src: str | None) -> bool:
            s = (src or "").lower()
            return s in {"heading_distance", "manhattan", "sparse"} or s.startswith(
                "photo_"
            )

        ma_n = sum(1 for p in accepted if _is_ma(p.get("source")))
        a_n = sum(1 for p in accepted if _is_a(p.get("source")))
        if ma_n and a_n:
            source_tag = "mapanything_hybrid"
        elif ma_n:
            source_tag = "mapanything_planarize"
        elif a_n:
            source_tag = "mapanything_hybrid" if seed_hyps else "photo_consistency"
        else:
            source_tag = "mapanything_planarize"

        log.info(
            "facades Path α: %s segmented + %s A seeds → %s ZNCC-kept "
            "(ma_kept=%s a_kept=%s; ply=%s pts; source=%s)",
            len(hyps),
            len(seed_hyps),
            len(accepted),
            ma_n,
            a_n,
            len(xyz_raw),
            source_tag,
        )
        if not accepted and fallback_heading:
            log.warning(
                "facades Path α: 0 ZNCC accepts after hybrid — falling back to "
                "Milestone A heading×distance search (same run; no OSM/BAG invent)"
            )
            accepted = search_photo_consistent_planes(
                frames,
                xyz if len(xyz) >= 30 else None,
                zncc_accept=min(float(zncc_accept), 0.35),
                ground_z=ground_z,
                max_keep=n_planes,
            )
            if accepted:
                source_tag = "photo_consistency_fallback"
    else:
        if planarize is True and len(xyz_raw) < 100:
            log.warning(
                "facades: planarize requested but PLY too thin (%s pts) — "
                "falling back to Milestone A heading×distance",
                len(xyz_raw),
            )
        accepted = search_photo_consistent_planes(
            frames,
            xyz if len(xyz) >= 30 else None,
            zncc_accept=float(zncc_accept),
            ground_z=ground_z,
            max_keep=n_planes,
        )

    planes: list[dict] = []
    for pl in accepted:
        planes.append(plane_dict_for_obj(pl, ground_z))

    dest_obj.parent.mkdir(parents=True, exist_ok=True)
    tex_dir = dest_obj.parent / "textures"
    mtl_path = dest_obj.with_suffix(".mtl")
    planes_json = dest_obj.parent / "planes.json"

    # Ground extents from photo cloud or camera AABB
    if len(xyz) >= 4:
        extent_xyz = xyz
    elif frames:
        pts = np.array(
            [[float(f["e"]), float(f["n"]), float(f.get("u") or ground_z)] for f in frames],
            dtype=np.float64,
        )
        pad = 15.0
        extent_xyz = np.vstack(
            [
                pts,
                pts.min(axis=0) - pad,
                pts.max(axis=0) + pad,
            ]
        )
    else:
        extent_xyz = np.array(
            [[-10, -10, ground_z], [10, 10, ground_z]], dtype=np.float64
        )

    # --- FAIL-LOUD preserve: do not wipe prior product ---
    if not planes and keep_previous_on_fail and _facade_product_looks_nonempty(dest_obj):
        failed_obj = dest_obj.with_name(dest_obj.stem + ".failed.obj")
        failed_mtl = dest_obj.with_name(dest_obj.stem + ".failed.mtl")
        failed_json = dest_obj.parent / "planes.failed.json"
        ground_only = [
            {
                "name": "ground",
                "map": None,
                "kd": (0.35, 0.38, 0.32),
            }
        ]
        _write_mtl(failed_mtl, ground_only)
        _write_obj(failed_obj, [], extent_xyz, ground_only, failed_mtl.name)
        write_planes_json(
            failed_json,
            planes=[],
            ground_z=ground_z,
            residual_points=residual_pts,
            source=f"{source_tag}_failed",
            textured_maps=[],
        )
        log.error(
            "facades: 0 photo-consistent walls (ZNCC≥%.2f, source=%s) — "
            "kept previous %s / textures / planes.json; diagnostics → %s / %s. "
            "Not inventing RANSAC/OSM blocks.",
            zncc_accept,
            source_tag,
            dest_obj.name,
            failed_obj.name,
            failed_json.name,
        )
        return {
            "path": str(dest_obj),
            "mtl": str(mtl_path),
            "planes_json": str(planes_json),
            "planes": 0,
            "textured": 0,
            "ground_textured": bool((tex_dir / "ground.jpg").is_file()),
            "source": source_tag,
            "planarize": bool(path_alpha_attempted and source_tag == "mapanything_planarize"),
            "zncc_accept": float(zncc_accept),
            "ground_z": ground_z,
            "points_used": int(len(xyz_raw) if len(xyz_raw) else len(xyz)),
            "points_voxel": int(len(xyz)),
            "residual_points": int(residual_pts),
            "mean_zncc": None,
            "path_alpha": source_tag == "mapanything_planarize",
            "preserved_previous": True,
            "ok": False,
            "failed_obj": str(failed_obj),
            "failed_planes_json": str(failed_json),
        }

    # Success path — optionally stage to *.candidate when a better product exists
    prev_q = (
        _read_facade_quality(dest_obj)
        if keep_previous_on_fail and _facade_product_looks_nonempty(dest_obj)
        else None
    )
    stage_candidate = bool(prev_q is not None and planes)

    if stage_candidate:
        out_obj = dest_obj.with_name(dest_obj.stem + ".candidate.obj")
        out_mtl = dest_obj.with_name(dest_obj.stem + ".candidate.mtl")
        out_json = dest_obj.parent / "planes.candidate.json"
        bake_tex_dir = tex_dir / "candidate"
        tex_map_prefix = "textures/candidate"
    else:
        out_obj = dest_obj
        out_mtl = mtl_path
        out_json = planes_json
        bake_tex_dir = tex_dir
        tex_map_prefix = "textures"
        # Wipe-on-write only when replacing the live product
        tex_dir.mkdir(parents=True, exist_ok=True)
        if tex_dir.is_dir():
            for old_tex in tex_dir.glob("facade_*.jpg"):
                old_tex.unlink(missing_ok=True)

    bake_tex_dir.mkdir(parents=True, exist_ok=True)
    if stage_candidate:
        for old_tex in bake_tex_dir.glob("facade_*.jpg"):
            old_tex.unlink(missing_ok=True)

    materials: list[dict[str, Any]] = []
    # Ground sat texture always lives at textures/ground.jpg (shared)
    tex_dir.mkdir(parents=True, exist_ok=True)
    ground_tex = tex_dir / "ground.jpg"
    has_ground_tex = _ground_satellite_texture(
        extent_xyz, ground_tex, satellite, local_frame
    )
    materials.append(
        {
            "name": "ground",
            "map": "textures/ground.jpg" if has_ground_tex else None,
            "kd": (0.35, 0.38, 0.32),
        }
    )

    textured = 0
    for i, pl in enumerate(planes):
        quad = pl.get("quad") or _plane_quad(pl)
        mat_name = f"facade_{i:02d}"
        map_rel = None
        if frames:
            cam = _pick_frontal_camera(pl, quad, frames)
            if cam is not None:
                tex_path = bake_tex_dir / f"facade_{i:02d}.jpg"
                if _warp_facade_texture(quad, cam, tex_path):
                    map_rel = f"{tex_map_prefix}/facade_{i:02d}.jpg"
                    textured += 1
        materials.append(
            {
                "name": mat_name,
                "map": map_rel,
                "kd": (0.55, 0.50, 0.42),
                "quad": quad,
            }
        )

    if not planes:
        log.error(
            "facades.obj: 0 photo-consistent walls (ZNCC≥%.2f, source=%s). "
            "Studio will show ground only — not uncorrelated RANSAC/OSM blocks.",
            zncc_accept,
            source_tag,
        )

    for i, pl in enumerate(planes):
        if accepted and i < len(accepted):
            pl["width_m"] = float(accepted[i].get("width_m") or 0.0)
            pl["height_m"] = float(accepted[i].get("height_m") or 0.0)
            pl["zncc"] = accepted[i].get("zncc")
            pl["inliers"] = accepted[i].get("inliers") or accepted[i].get("count") or 0
            if "n" not in pl and "nx" in pl:
                pl["n"] = np.array([pl["nx"], pl["ny"], 0.0], dtype=np.float64)

    _write_mtl(out_mtl, materials)
    _write_obj(out_obj, planes, extent_xyz, materials, out_mtl.name)

    tex_maps = [m.get("map") for m in materials[1:]]
    write_planes_json(
        out_json,
        planes=planes,
        ground_z=ground_z,
        residual_points=residual_pts,
        source=source_tag,
        textured_maps=tex_maps,
    )

    mean_zncc = (
        float(np.mean([p.get("zncc", 0.0) for p in planes if p.get("zncc") is not None]))
        if any(p.get("zncc") is not None for p in planes)
        else None
    )
    new_q = {
        "textured": int(textured),
        "mean_zncc": mean_zncc,
        "plane_count": int(len(planes)),
    }

    if stage_candidate and prev_q is not None:
        ok_promote, why = _is_strictly_better(new_q, prev_q)
        if not ok_promote:
            log.error(
                "facades: candidate NOT promoted — %s "
                "(new textured=%s planes=%s mean_zncc=%s source=%s vs "
                "prev textured=%s planes=%s mean_zncc=%s) — "
                "kept previous %s / textures / planes.json; candidate → %s / %s. "
                "Not inventing RANSAC/OSM blocks.",
                why,
                new_q["textured"],
                new_q["plane_count"],
                new_q["mean_zncc"],
                source_tag,
                prev_q.get("textured"),
                prev_q.get("plane_count"),
                prev_q.get("mean_zncc"),
                dest_obj.name,
                out_obj.name,
                out_json.name,
            )
            return {
                "path": str(dest_obj),
                "mtl": str(mtl_path),
                "planes_json": str(planes_json),
                # Report *product* plane count so CLI/Studio see the kept result
                "planes": int(prev_q.get("plane_count") or 0),
                "textured": int(prev_q.get("textured") or 0),
                "ground_textured": has_ground_tex
                or bool((tex_dir / "ground.jpg").is_file()),
                "source": source_tag,
                "planarize": bool(
                    path_alpha_attempted and source_tag == "mapanything_planarize"
                ),
                "zncc_accept": float(zncc_accept),
                "ground_z": ground_z,
                "points_used": int(len(xyz_raw) if len(xyz_raw) else len(xyz)),
                "points_voxel": int(len(xyz)),
                "residual_points": int(residual_pts),
                "mean_zncc": prev_q.get("mean_zncc"),
                "path_alpha": source_tag == "mapanything_planarize",
                "preserved_previous": True,
                "rejected_weaker": True,
                "candidate_reason": why,
                "candidate_planes": int(len(planes)),
                "candidate_textured": int(textured),
                "candidate_mean_zncc": mean_zncc,
                "ok": False,
                "candidate_obj": str(out_obj),
                "candidate_planes_json": str(out_json),
                "previous_textured": int(prev_q.get("textured") or 0),
                "previous_mean_zncc": prev_q.get("mean_zncc"),
            }

    if stage_candidate:
        _promote_candidate_facades(
            dest_obj,
            cand_obj=out_obj,
            cand_mtl=out_mtl,
            cand_json=out_json,
            cand_tex_dir=bake_tex_dir,
        )
        # Product maps are now under textures/facade_*.jpg
        mean_zncc = (
            float(
                np.mean(
                    [p.get("zncc", 0.0) for p in planes if p.get("zncc") is not None]
                )
            )
            if any(p.get("zncc") is not None for p in planes)
            else None
        )

    return {
        "path": str(dest_obj),
        "mtl": str(mtl_path),
        "planes_json": str(planes_json),
        "planes": len(planes),
        "textured": textured,
        "ground_textured": has_ground_tex,
        "source": source_tag,
        "planarize": bool(
            path_alpha_attempted and source_tag == "mapanything_planarize"
        ),
        "zncc_accept": float(zncc_accept),
        "ground_z": ground_z,
        "points_used": int(len(xyz_raw) if len(xyz_raw) else len(xyz)),
        "points_voxel": int(len(xyz)),
        "residual_points": int(residual_pts),
        "mean_zncc": mean_zncc,
        "path_alpha": source_tag == "mapanything_planarize",
        "preserved_previous": False,
        "rejected_weaker": False,
        "ok": len(planes) > 0,
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
