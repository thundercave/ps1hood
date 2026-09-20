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
    *,
    ps1_tex_size: int | None = 128,
    margin_px: float = 40.0,
    allow_clip: bool = True,
) -> bool:
    """Perspective-bake one façade quad into ``dest``.

    Soft fail recovery (post-rectify full-quad coverage):
      1. try ``margin_px`` (default 40 — historic hard gate)
      2. on fail, retry with ``max(margin_px, 120)``
      3. if still out-of-frame and ``allow_clip``, clip UV to the image
         rectangle (keeps center; notes via logger) and warp anyway
    """
    import logging

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

    width_m = float(np.linalg.norm(pts[1][:2] - pts[0][:2]))
    height_m = float(abs(pts[3][2] - pts[0][2]))
    aspect = max(width_m, 0.5) / max(height_m, 0.5)
    th = _next_pot(int(round(256 * max(1.0, 1.0 / aspect))))
    tw = _next_pot(int(round(th * aspect)))
    tw = int(np.clip(tw, 64, 1024))
    th = int(np.clip(th, 64, 1024))
    dst = np.array(
        [[0, th - 1], [tw - 1, th - 1], [tw - 1, 0], [0, 0]],
        dtype=np.float32,
    )

    def _outside(uv_arr: np.ndarray, margin: float) -> bool:
        return bool(
            (uv_arr[:, 0] < -margin).any()
            or (uv_arr[:, 1] < -margin).any()
            or (uv_arr[:, 0] > w + margin).any()
            or (uv_arr[:, 1] > h + margin).any()
        )

    def _write(src_uv: np.ndarray, margin: float) -> bool:
        src = src_uv.astype(np.float32).copy()
        src[:, 0] = np.clip(src[:, 0], -margin, w + margin)
        src[:, 1] = np.clip(src[:, 1], -margin, h + margin)
        try:
            M = cv2.getPerspectiveTransform(src, dst)
        except cv2.error:
            return False
        warped = cv2.warpPerspective(img, M, (tw, th), flags=cv2.INTER_LINEAR)
        if ps1_tex_size is not None and int(ps1_tex_size) > 0:
            from ps1_hood.reconstruct.ps1_facades import apply_ps1_texture

            warped = apply_ps1_texture(warped, tex_size=int(ps1_tex_size))
            jpeg_q = 85
        else:
            jpeg_q = 88
        dest.parent.mkdir(parents=True, exist_ok=True)
        return bool(cv2.imwrite(str(dest), warped, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_q]))

    margins = [float(margin_px)]
    loose = max(float(margin_px), 120.0)
    if loose not in margins:
        margins.append(loose)

    for margin in margins:
        if _outside(uv, margin):
            continue
        if _write(uv, margin):
            return True

    if allow_clip:
        # Last resort: clip UV to the frame (crop to in-frame bbox / corners).
        logging.getLogger(__name__).info(
            "facades warp: clipping UV to frame for %s (corners outside margin %.0f)",
            dest.name,
            loose,
        )
        clipped = uv.copy()
        clipped[:, 0] = np.clip(clipped[:, 0], 0.0, float(w - 1))
        clipped[:, 1] = np.clip(clipped[:, 1], 0.0, float(h - 1))
        if _write(clipped, 0.0):
            return True
    return False


def _rank_frontal_cameras(
    pl: dict,
    quad: list[tuple[float, float, float]],
    frames: list[dict[str, Any]],
    *,
    min_frontal: float = 0.20,
    top_k: int = 5,
    prefer_indices: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Rank cameras by frontal×coverage; prefer accepted ``view_indices`` first."""
    nx, ny = float(pl["nx"]), float(pl["ny"])
    n_xy = np.array([nx, ny], dtype=np.float64)
    center = np.mean(np.array(quad, dtype=np.float64), axis=0)
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for fi, fr in enumerate(frames):
        cam = np.array([fr["e"], fr["n"], fr["u"]], dtype=np.float64)
        n = n_xy.copy()
        if n @ (cam[:2] - center[:2]) < 0:
            n = -n
        fwd = _forward_xy(fr)
        fn = float(np.linalg.norm(fwd))
        if fn < 1e-6:
            continue
        fwd = fwd / fn
        frontal = float((-fwd) @ n)
        if frontal < min_frontal:
            continue
        img = fr.get("_img")
        if img is None:
            img = cv2.imread(fr["path"], cv2.IMREAD_COLOR) if fr.get("path") else None
        if img is None:
            continue
        h, w = img.shape[:2]
        uv_c = _project_points(fr, w, h, center.reshape(1, 3))[0]
        if not np.isfinite(uv_c).all():
            continue
        uv_q = _project_points(fr, w, h, np.array(quad, dtype=np.float64))
        finite = np.isfinite(uv_q).all(axis=1)
        in_frame = (
            finite
            & (uv_q[:, 0] >= -40)
            & (uv_q[:, 1] >= -40)
            & (uv_q[:, 0] < w + 40)
            & (uv_q[:, 1] < h + 40)
        )
        coverage = float(in_frame.sum()) / max(len(quad), 1)
        dist = float(np.linalg.norm(cam - center))
        score = frontal * (0.5 + 0.5 * coverage) * (1.0 / (1.0 + 0.02 * dist))
        scored.append((score, fi, {**fr, "_img": img, "_frontal": frontal, "_score": score}))

    if not scored:
        return []

    prefer = set(int(i) for i in (prefer_indices or []) if isinstance(i, (int, float)))
    # Preferred view_indices that scored, in prefer order, then rest by score desc.
    by_idx = {fi: (sc, cam) for sc, fi, cam in scored}
    ordered: list[dict[str, Any]] = []
    seen: set[int] = set()
    if prefer_indices:
        for raw in prefer_indices:
            fi = int(raw)
            if fi in by_idx and fi not in seen:
                ordered.append(by_idx[fi][1])
                seen.add(fi)
    rest = sorted(
        ((sc, fi, cam) for sc, fi, cam in scored if fi not in seen),
        key=lambda t: t[0],
        reverse=True,
    )
    for _sc, fi, cam in rest:
        ordered.append(cam)
        seen.add(fi)
    return ordered[: max(1, int(top_k))]


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

    Never promote on textured↑ alone when plane count drops or mean ZNCC
    regresses by more than ``zncc_eps``. Soft floors (textured≥min or
    mean_zncc≥min) still protect an already-good product from weak replaces.

    Clauses (first match wins; reason string names the clause):
      1. textured_new > textured_prior AND planes_new ≥ planes_prior
         AND mean_zncc_new ≥ mean_prior − eps
      2. textured equal AND planes_new > planes_prior
         AND mean_zncc_new ≥ mean_prior − eps
      3. textured and planes equal AND mean_zncc_new ≥ mean_prior + eps
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

    def _mean_not_regress() -> bool:
        if pm is None:
            return True
        if nm is None:
            return False
        return float(nm) >= float(pm) - float(zncc_eps)

    def _mean_improved() -> bool:
        if nm is None or pm is None:
            return False
        return float(nm) >= float(pm) + float(zncc_eps)

    # Clause 1: more textured, planes not down, mean within eps of prior
    if nt > pt and np_ >= pp_ and _mean_not_regress():
        return True, "clause1:more textured (planes≥, mean within eps)"

    # Clause 2: same textured, more planes, mean not material regress
    if nt == pt and np_ > pp_ and _mean_not_regress():
        return True, "clause2:more planes (mean within eps)"

    # Clause 3: same textured + planes, mean up by ≥ eps
    if nt == pt and np_ == pp_ and _mean_improved():
        return True, "clause3:higher mean_zncc"

    if not old_good and (np_ > 0 or nt > 0):
        return True, "replace empty/weak product"

    if nt < pt:
        return False, f"fewer textured ({nt}<{pt})"
    if nt > pt and np_ < pp_:
        return False, (
            f"more textured but fewer planes ({np_}<{pp_})"
        )
    if nt > pt and not _mean_not_regress():
        return False, (
            f"more textured but mean_zncc regress ({nm} vs {pm})"
        )
    if nt == pt and np_ < pp_:
        return False, f"fewer planes ({np_}<{pp_})"
    if nt == pt and np_ > pp_ and not _mean_not_regress():
        return False, (
            f"more planes but mean_zncc regress ({nm} vs {pm})"
        )
    if nt == pt and np_ == pp_:
        return False, f"not higher mean_zncc ({nm} vs {pm})"
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


def _ground_z_from_cams(frames: list[dict[str, Any]]) -> float:
    if not frames:
        return 0.0
    return float(np.median([float(f.get("u") or 0.0) for f in frames])) - 2.5


def _facade_output_paths(dest_obj: Path, kind: str) -> dict[str, Path | str]:
    """Product / candidate / control artefact paths. Control never shares candidate."""
    dest_obj = Path(dest_obj)
    parent = dest_obj.parent
    tex_dir = parent / "textures"
    if kind == "control":
        return {
            "obj": dest_obj.with_name(dest_obj.stem + ".control.obj"),
            "mtl": dest_obj.with_name(dest_obj.stem + ".control.mtl"),
            "json": parent / "planes.control.json",
            "bake_tex": tex_dir / "control",
            "tex_prefix": "textures/control",
        }
    if kind == "candidate":
        return {
            "obj": dest_obj.with_name(dest_obj.stem + ".candidate.obj"),
            "mtl": dest_obj.with_name(dest_obj.stem + ".candidate.mtl"),
            "json": parent / "planes.candidate.json",
            "bake_tex": tex_dir / "candidate",
            "tex_prefix": "textures/candidate",
        }
    return {
        "obj": dest_obj,
        "mtl": dest_obj.with_suffix(".mtl"),
        "json": parent / "planes.json",
        "bake_tex": tex_dir,
        "tex_prefix": "textures",
    }


def _candidate_artifacts_exist(dest_obj: Path) -> bool:
    dest_obj = Path(dest_obj)
    return (
        dest_obj.with_name(dest_obj.stem + ".candidate.obj").is_file()
        or (dest_obj.parent / "planes.candidate.json").is_file()
    )


def _resolve_output_kind(
    output_kind: str,
    *,
    control_out: bool,
    use_planarize: bool,
    prev_product: bool,
    dest_obj: Path | None = None,
) -> str:
    """auto → control for A-only when product/candidate exist (never clobber).

    First-run A (no product, no candidate) still writes live product.
    Explicit ``control_out`` / ``output_kind=control`` always stages ``*.control``.
    """
    kind = (output_kind or "auto").lower().strip()
    if control_out or kind == "control":
        return "control"
    if kind in {"candidate", "product"}:
        return kind
    if not use_planarize:
        has_candidate = bool(dest_obj and _candidate_artifacts_exist(dest_obj))
        if prev_product or has_candidate:
            return "control"
        return "product"
    if prev_product:
        return "candidate"
    return "product"


def _resolve_a_support_ply(
    dest_obj: Path,
    *,
    a_ply_path: Path | None,
    a_source: str,
    ply_path: Path | None,
) -> Path | None:
    """A-arm PLY: explicit path, then flow/recon next to dest, then ply_path fallback."""
    from ps1_hood.reconstruct.planarize import resolve_a_ply

    src = (a_source or "flow").lower().strip()
    if src == "product":
        return None
    if a_ply_path is not None and Path(a_ply_path).is_file():
        return Path(a_ply_path)
    dest_obj = Path(dest_obj)
    for base in (dest_obj.parent, dest_obj.parent.parent):
        found = resolve_a_ply(base, src)
        if found is not None:
            return found
    if src == "mapanything" and ply_path is not None and Path(ply_path).is_file():
        return Path(ply_path)
    # Single-PLY callers / tests: fall back to the peel cloud so A still runs.
    if ply_path is not None and Path(ply_path).is_file():
        return Path(ply_path)
    return None


def extract_facades(
    ply_path: Path | None,
    dest_obj: Path,
    n_planes: int = 16,
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
    min_inliers: int | None = None,
    residual_stop: int | None = None,
    vertical_dot: float | None = None,
    peel_max_planes: int | None = None,
    nms_xy_m: float | None = None,
    nms_xy_split_m: float | None = None,
    union_strategy: str | None = None,
    hybrid_a_full_search: bool = True,
    a_ply_path: Path | None = None,
    a_source: str = "flow",
    output_kind: str = "auto",
    control_out: bool = False,
    ps1_rectify: bool = True,
    ps1_tex_size: int | None = 128,
    gap_fill: bool = False,
    project_root: Path | None = None,
    worst_cam_ids: list[str] | None = None,
    worst_from_compare: int | None = None,
    max_gap_adds: int | None = None,
    sat_aabb_gate_m: float | None = None,
    ma_peel_cap: int | None = None,
    gap_min_views: int | None = None,
    gap_seeds_mode: str | None = None,
    sat_edge_reanchor: bool = True,
    sat_edge_max_drift_m: float | None = None,
    sat_aabb_gate_sat_edge_m: float | None = None,
    gap_nms_xy_m: float | None = None,
    gap_nms_d_tol_m: float | None = None,
    gap_nms_no_opposite: bool = True,
) -> dict:
    """Photo-consistent vertical façades under known poses.

    Path α (planarize): when ``planarize=True`` or auto (dense PLY ≳50k pts),
    seed vertical planes from MapAnything ENU cloud via Open3D/numpy
    ``segment_plane`` peel, then ZNCC-gate + ortho bake (Milestone A).

    Dual-**source** (PR post-#25): A arm xyz / ``ground_z`` come from
    ``a_ply_path`` / ``--a-source flow`` (``recon/cloud_flow.ply`` else
    flow ``recon/cloud.ply``). MA peels stay on ``ply_path``
    (``--ma-source mapanything``). Do not force both arms onto one MA PLY.

    With ``hybrid_heading`` (default on for Path α) and
    ``hybrid_a_full_search`` (default on): **dual arm** — A arm runs full
    Milestone A ``search_photo_consistent_planes`` (ungated historic path);
    MA arm peels through ``score_planar_hyps`` only (no A seed inject);
    then ``union_keep_planes`` with default ``a_priority`` keeps A first and
    adds non-dup MA. Opt out with ``hybrid_a_full_search=False`` to restore
    the older hypothesize→``score_planar_hyps`` seed-inject path.
    Promote still uses ``_is_strictly_better`` on the union bake metrics.

    ``--a-source product`` locks live ``planes.json`` as the A family.
    A-only / ``--no-planarize`` / ``control_out`` write ``*.control`` and
    never clobber ``*.candidate`` or live product.

    If hybrid/Path α keeps 0 planes and ``fallback_heading``, retry full
    Milestone A search on the same run (``photo_consistency_fallback``).

    Fail-loud: never invent flow-RANSAC or OSM/BAG walls. When accepts==0 and
    ``keep_previous_on_fail`` (default), do **not** clobber non-empty product
    ``facades.obj`` / ``.mtl`` / ``textures/facade_*.jpg`` / ``planes.json`` —
    write diagnostics to ``*.failed`` instead.

    Quality gate: with ``keep_previous_on_fail``, **promote only if strictly
    better** than the existing product — more textured only when planes do
    not drop and mean ZNCC stays within ``zncc_eps`` of prior; or same
    textured with more planes (mean within eps) / higher mean (+eps). Never
    textured↑ alone when planes↓ or mean regresses >eps. Weaker or equal
    results — including Milestone A fallback after Path α 0 accepts — go to
    ``*.candidate`` and the live product is kept.

    PR-C ``gap_fill``: keep product / A core; add ZNCC-gated manhattan + sat
    corner seeds and road-rejected MA peels for side/return walls; ``a_priority``
    union; quality-keep vs existing product. Cap MA peels (default ≤3). No BAG.

    Gap-*adds* only (not product_lock) also pass stricter multi-view + sat AABB:
    ZNCC≥0.35 on ≥``gap_min_views`` cams, median ZNCC≥0.10, max |n·cam_fwd|≥0.4,
    center ≤``sat_aabb_gate_m`` of a roof boundary with n∥edge; cap with
    ``max_gap_adds``. Soft-pass sat gate when no roofs. ``gap_seeds_mode``:
    ``sat-edge`` (default) = uncovered roof AABB edges → same-side facing cams
    (else ``gap_needs.json``); ``legacy`` = manhattan/corner/worst-cam; ``both``.
    ``sat_edge_reanchor`` snaps accepted sat_edge centers back onto the seed
    segment (reject refine_drift beyond ``sat_edge_max_drift_m``); AABB uses
    that edge_id. ``sat_aabb_gate_sat_edge_m`` optionally softens AABB for
    sat_edge only (never MA peels). ``gap_nms_*`` tighten a_priority dup of
    ``ma_gap_sat_edge`` vs product_lock only (cover check, no opposite-d,
    default XY 3 m / Δd 1 m) — global NMS for other sources stays 6/2.5.

    Worst-cam targeted gap-fill: ``worst_cam_ids`` / ``worst_from_compare`` seed
    planes in front of those cams (dist×yaw); filter manhattan/corner to visible
    in those cams; lock product first, NMS-add new only; ``max_keep`` 24; bake
    textures for *new* indices only; stage candidate + quality-keep (never wipe
    existing ``textures/facade_*.jpg`` on fail).
    """
    import logging

    from ps1_hood.reconstruct.photo_planes import (
        hypothesize_vertical_planes,
        plane_dict_for_obj,
        search_photo_consistent_planes,
    )
    from ps1_hood.reconstruct.planarize import (
        DEFAULT_MAX_PLANES,
        DEFAULT_MIN_INLIERS,
        DEFAULT_NMS_XY_M,
        DEFAULT_NMS_XY_SPLIT_M,
        DEFAULT_RESIDUAL_STOP,
        DEFAULT_SPLIT_OVERLAP_M,
        DEFAULT_SPLIT_TRIGGER_WIDTH_M,
        DEFAULT_SPLIT_WINDOW_M,
        DEFAULT_VERTICAL_DOT,
        DEFAULT_UNION_STRATEGY,
        DEFAULT_ZNCC_ACCEPT_MA,
        PLANARIZE_AUTO_MIN_POINTS,
        is_a_source,
        is_ma_source,
        planes_from_mapanything_ply,
        planes_from_product_json,
        score_planar_hyps,
        union_keep_planes,
        write_planes_json,
    )

    log = logging.getLogger(__name__)
    frames = list(frames or [])
    dest_obj = Path(dest_obj)
    a_source = (a_source or "flow").lower().strip()

    xyz_ma_raw = np.zeros((0, 3), dtype=np.float64)
    if ply_path is not None and Path(ply_path).is_file():
        try:
            xyz_ma_raw = _read_ply_xyz(Path(ply_path))
        except Exception as exc:  # noqa: BLE001
            log.warning("facades: could not read MA peel PLY %s (%s)", ply_path, exc)

    ply_a = _resolve_a_support_ply(
        dest_obj,
        a_ply_path=a_ply_path,
        a_source=a_source,
        ply_path=ply_path,
    )
    xyz_a_raw = np.zeros((0, 3), dtype=np.float64)
    if ply_a is not None and Path(ply_a).is_file():
        try:
            xyz_a_raw = _read_ply_xyz(Path(ply_a))
        except Exception as exc:  # noqa: BLE001
            log.warning("facades: could not read A support PLY %s (%s)", ply_a, exc)

    # Auto planarize on MA density (or the single seed if no MA).
    n_auto = len(xyz_ma_raw) if len(xyz_ma_raw) else len(xyz_a_raw)
    use_planarize = planarize if planarize is not None else (
        n_auto >= PLANARIZE_AUTO_MIN_POINTS
    )
    if zncc_accept is None:
        zncc_accept = DEFAULT_ZNCC_ACCEPT_MA if use_planarize else 0.35

    residual_pts = 0
    source_tag = "photo_consistency"
    path_alpha_attempted = False
    a_kept_n = 0
    ma_added_n = 0
    gap_bake_lock_n = 0  # product_lock count for gap-fill bake-new-only
    a_ply_str = str(ply_a) if ply_a is not None else ""
    ma_ply_str = str(ply_path) if ply_path is not None else ""

    # A voxel = historic 0.20 (not MA 0.08). MA peels use raw + their voxel.
    if len(xyz_a_raw) >= 20:
        xyz_a = _voxel_downsample_xyz(xyz_a_raw, 0.20)
        ground_z_a = _fit_ground_z(xyz_a)
    else:
        xyz_a = xyz_a_raw
        ground_z_a = _ground_z_from_cams(frames) if frames else 0.0

    xyz_raw = xyz_ma_raw if len(xyz_ma_raw) else xyz_a_raw
    if len(xyz_ma_raw) >= 20:
        xyz = _voxel_downsample_xyz(xyz_ma_raw, voxel_m if use_planarize else 0.20)
        ground_z = _fit_ground_z(xyz)
    elif len(xyz_a) >= 20:
        xyz = xyz_a
        ground_z = ground_z_a
    else:
        xyz = xyz_raw
        ground_z = ground_z_a if frames else 0.0

    if use_planarize and ply_path is not None and Path(ply_path).is_file() and len(xyz_ma_raw) >= 100:
        path_alpha_attempted = True
        cam_c = (
            np.mean(
                [[float(f["e"]), float(f["n"]), float(f["u"])] for f in frames],
                axis=0,
            )
            if frames
            else xyz.mean(axis=0)
        )
        peel_n = (
            int(peel_max_planes)
            if peel_max_planes is not None
            else int(DEFAULT_MAX_PLANES)
        )
        peel_min_inl = (
            int(min_inliers) if min_inliers is not None else int(DEFAULT_MIN_INLIERS)
        )
        peel_resid = (
            int(residual_stop)
            if residual_stop is not None
            else int(DEFAULT_RESIDUAL_STOP)
        )
        peel_vdot = (
            float(vertical_dot)
            if vertical_dot is not None
            else float(DEFAULT_VERTICAL_DOT)
        )
        hyps, ground, residual_pts = planes_from_mapanything_ply(
            Path(ply_path),
            cam_c,
            xyz=xyz_raw,
            ground_z=ground_z,
            voxel=voxel_m,
            distance_threshold=plane_dist_m,
            min_inliers=peel_min_inl,
            max_planes=peel_n,
            residual_stop=peel_resid,
            vertical_dot_max=peel_vdot,
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

        nms_xy = float(DEFAULT_NMS_XY_M if nms_xy_m is None else nms_xy_m)
        nms_xy_split = float(
            DEFAULT_NMS_XY_SPLIT_M if nms_xy_split_m is None else nms_xy_split_m
        )
        strategy = (
            DEFAULT_UNION_STRATEGY
            if union_strategy is None
            else str(union_strategy)
        )
        # PR-C gap_fill requires dual-source Path α (product/A + MA peels).
        if gap_fill:
            hybrid_heading = True
            hybrid_a_full_search = True
            strategy = "a_priority"
        dual_arm = bool(hybrid_heading and hybrid_a_full_search and frames)
        seed_hyps: list[dict] = []
        union_tel: dict[str, Any] = {}

        def _is_ma(src: str | None) -> bool:
            return is_ma_source(src)

        def _is_a(src: str | None) -> bool:
            return is_a_source(src)

        def _run_a_arm(*, zncc: float) -> list[dict]:
            """A arm on flow xyz, or locked product planes.json."""
            product_json = dest_obj.parent / "planes.json"
            flow_thin = len(xyz_a) < 30
            use_lock = a_source == "product" or (
                a_source in {"flow", "recon"}
                and flow_thin
                and product_json.is_file()
            )
            if use_lock:
                locked = planes_from_product_json(product_json)
                if locked:
                    log.info(
                        "facades A arm: product_lock %s planes from %s "
                        "(a_source=%s flow_pts=%s)",
                        len(locked),
                        product_json,
                        a_source,
                        len(xyz_a),
                    )
                    return locked
                if a_source == "product":
                    log.warning(
                        "facades: --a-source product but no usable planes at %s",
                        product_json,
                    )
                    return []
            return search_photo_consistent_planes(
                frames,
                xyz_a if len(xyz_a) >= 30 else None,
                zncc_accept=float(zncc),
                ground_z=ground_z_a,
                max_keep=n_planes,
            )

        if dual_arm:
            # Dual-source dual-arm: A on flow xyz / product lock; MA peels only.
            # PR-C gap_fill: lock product 10/8 core when present; score gap seeds
            # + road-rejected MA peels; softer min_frontal for far-side only.
            gap_seeds: list[dict] = []
            ma_hyps = list(hyps)
            min_frontal_ma = 0.25
            keep_cap = int(n_planes)
            prefer_frame_idxs: list[int] | None = None
            n_product_lock = 0
            if gap_fill:
                from ps1_hood.reconstruct.gap_fill import (
                    GAP_FILL_MAX_KEEP,
                    GAP_FILL_MIN_FRONTAL,
                    GAP_FILL_PEEL_CAP,
                    build_gap_fill_seeds,
                    count_untextured_product,
                    filter_ma_peels_for_gaps,
                    frame_indices_for_cam_ids,
                    load_worst_cam_ids_from_compare,
                )

                strategy = "a_priority"
                keep_cap = max(int(n_planes), int(GAP_FILL_MAX_KEEP))
                peel_cap = min(
                    int(peel_n),
                    int(ma_peel_cap) if ma_peel_cap is not None else int(GAP_FILL_PEEL_CAP),
                )
                root = Path(project_root) if project_root is not None else dest_obj.parent.parent
                # Resolve worst-cam ids (CLI list and/or compare top-N)
                worst_ids: list[str] = []
                if worst_cam_ids:
                    worst_ids.extend(str(c).strip() for c in worst_cam_ids if str(c).strip())
                if worst_from_compare is not None and int(worst_from_compare) > 0:
                    for cid in load_worst_cam_ids_from_compare(
                        root, n=int(worst_from_compare)
                    ):
                        if cid not in worst_ids:
                            worst_ids.append(cid)
                prefer_frame_idxs = (
                    frame_indices_for_cam_ids(frames, worst_ids) if worst_ids else None
                )
                n_prod, n_untex = count_untextured_product(dest_obj)
                if n_untex:
                    log.info(
                        "facades gap_fill: product has %s/%s untextured — "
                        "re-warp on bake before inventing new planes",
                        n_untex,
                        n_prod,
                    )
                # Prefer product lock as A core (keep 18/18 / 10/8)
                product_json = dest_obj.parent / "planes.json"
                locked = planes_from_product_json(product_json) if product_json.is_file() else []
                if locked:
                    accepted_a = locked
                    n_product_lock = len(locked)
                    log.info(
                        "facades gap_fill: A core = product_lock %s planes",
                        len(accepted_a),
                    )
                else:
                    accepted_a = _run_a_arm(zncc=min(float(zncc_accept), 0.35))
                    n_product_lock = len(accepted_a)
                ma_hyps = filter_ma_peels_for_gaps(
                    hyps,
                    frames,
                    root,
                    peel_cap=peel_cap,
                )
                _seeds_mode = (
                    str(gap_seeds_mode).strip().lower()
                    if gap_seeds_mode
                    else "sat-edge"
                )
                gap_seeds = build_gap_fill_seeds(
                    frames,
                    root,
                    ground_z=ground_z_a,
                    worst_cam_ids=worst_ids or None,
                    gap_seeds_mode=_seeds_mode,
                    product_planes=list(accepted_a),
                    write_needs=True,
                    run_name=root.name,
                )
                min_frontal_ma = float(GAP_FILL_MIN_FRONTAL)
                log.info(
                    "facades gap_fill: gap_seeds_mode=%s",
                    _seeds_mode,
                )
                log.info(
                    "facades gap_fill: ma_peels=%s→%s gap_seeds=%s "
                    "min_frontal=%.2f keep_cap=%s peel_cap=%s worst_cams=%s "
                    "prefer_frames=%s",
                    len(hyps),
                    len(ma_hyps),
                    len(gap_seeds),
                    min_frontal_ma,
                    keep_cap,
                    peel_cap,
                    worst_ids or None,
                    prefer_frame_idxs,
                )
            else:
                accepted_a = _run_a_arm(zncc=min(float(zncc_accept), 0.35))

            accepted_ma = score_planar_hyps(
                ma_hyps,
                frames,
                zncc_accept=float(zncc_accept),
                max_keep=keep_cap,
                seed_hyps=gap_seeds or None,
                split_trigger_width_m=split_trigger,
                split_window_m=split_window,
                split_overlap_m=split_overlap,
                nms_xy_m=nms_xy,
                nms_xy_split_m=nms_xy_split,
                union_strategy="nms",  # MA/gap NMS; final union below
                min_frontal=min_frontal_ma,
                prefer_frame_indices=prefer_frame_idxs,
            )
            # Lock product: retag A-family gap seeds so a_priority NMS-adds only
            # (never evict product_lock via higher-ZNCC manhattan/worst_cam).
            def _tag_gap_addable(p: dict) -> dict:
                src = str(p.get("source") or "")
                if src == "product_lock":
                    return p
                if _is_a(src):
                    q = dict(p)
                    q["source"] = f"ma_gap_{src}"
                    return q
                if src == "worst_cam":
                    q = dict(p)
                    q["source"] = "ma_gap_worst_cam"
                    return q
                if src == "sat_edge":
                    q = dict(p)
                    q["source"] = "ma_gap_sat_edge"
                    return q
                return p

            accepted_new = (
                [_tag_gap_addable(p) for p in accepted_ma] if gap_fill else list(accepted_ma)
            )
            if gap_fill and accepted_new:
                from ps1_hood.reconstruct.gap_fill import (
                    GAP_ADD_MAX_ADDS,
                    GAP_ADD_MIN_VIEWS,
                    GAP_ADD_SAT_EDGE_M,
                    SAT_EDGE_MAX_DRIFT_M,
                    filter_gap_adds,
                    load_sat_roof_regions,
                )

                _sat_m = (
                    float(GAP_ADD_SAT_EDGE_M)
                    if sat_aabb_gate_m is None
                    else float(sat_aabb_gate_m)
                )
                _min_views = (
                    int(GAP_ADD_MIN_VIEWS)
                    if gap_min_views is None
                    else int(gap_min_views)
                )
                _max_adds = (
                    int(GAP_ADD_MAX_ADDS)
                    if max_gap_adds is None
                    else int(max_gap_adds)
                )
                _drift = (
                    float(SAT_EDGE_MAX_DRIFT_M)
                    if sat_edge_max_drift_m is None
                    else float(sat_edge_max_drift_m)
                )
                _sat_edge_soft = (
                    None
                    if sat_aabb_gate_sat_edge_m is None
                    else float(sat_aabb_gate_sat_edge_m)
                )
                roof_regs = load_sat_roof_regions(root)
                before_gate = len(accepted_new)
                accepted_new = filter_gap_adds(
                    accepted_new,
                    frames,
                    roof_regions=roof_regs,
                    sat_aabb_gate_m=_sat_m,
                    sat_aabb_gate_sat_edge_m=_sat_edge_soft,
                    min_views=_min_views,
                    max_gap_adds=_max_adds,
                    sat_edge_reanchor=bool(sat_edge_reanchor),
                    sat_edge_max_drift_m=_drift,
                )
                log.info(
                    "facades gap_fill stricter gate: %s → %s "
                    "(sat_aabb=%.2f sat_edge_soft=%s min_views=%s max_adds=%s "
                    "reanchor=%s max_drift=%.2f)",
                    before_gate,
                    len(accepted_new),
                    _sat_m,
                    _sat_edge_soft,
                    _min_views,
                    _max_adds,
                    sat_edge_reanchor,
                    _drift,
                )
            accepted = union_keep_planes(
                list(accepted_a) + accepted_new,
                strategy=strategy,
                max_keep=keep_cap,
                nms_xy_m=nms_xy,
                nms_xy_split_m=nms_xy_split,
                gap_nms_xy_m=gap_nms_xy_m,
                gap_nms_d_tol_m=gap_nms_d_tol_m,
                gap_nms_no_opposite=bool(gap_nms_no_opposite),
                telemetry=union_tel,
            )
            if gap_fill:
                if n_product_lock <= 0:
                    n_product_lock = int(union_tel.get("a_kept") or len(accepted_a))
                gap_bake_lock_n = int(n_product_lock)
            a_pre = int(union_tel.get("a_pre_nms") or len(accepted_a))
            a_n = int(union_tel.get("a_kept") or sum(1 for p in accepted if _is_a(p.get("source"))))
            ma_added = int(
                union_tel.get("ma_added")
                or sum(1 for p in accepted if _is_ma(p.get("source")))
            )
            ma_n = ma_added
            if gap_fill:
                source_tag = "mapanything_gap_fill"
            elif a_n and ma_n:
                source_tag = "mapanything_hybrid"
            elif a_n:
                source_tag = "mapanything_hybrid"
            elif ma_n:
                source_tag = "mapanything_planarize"
            else:
                source_tag = "mapanything_planarize"
            a_kept_n = a_n
            ma_added_n = ma_added
            log.info(
                "facades Path α dual_arm%s: a_arm=%s kept=%s; ma_arm=%s; "
                "strategy=%s a_pre_nms=%s a_kept=%s ma_pre_nms=%s ma_added=%s "
                "n_dup_vs_a=%s remaining=%s union_kept=%s; "
                "a_ply=%s ma_ply=%s a_pts=%s ma_pts=%s; source=%s; "
                "promote vs product 10/8/0.418",
                " gap_fill" if gap_fill else "",
                "product_lock" if gap_fill and a_source else "search_photo_consistent_planes",
                len(accepted_a),
                len(accepted_ma),
                strategy,
                a_pre,
                a_n,
                int(union_tel.get("ma_pre_nms") or len(accepted_new)),
                ma_added,
                int(union_tel.get("n_dup_vs_a") or 0),
                int(union_tel.get("remaining") or 0),
                len(accepted),
                a_ply_str,
                ma_ply_str,
                len(xyz_a_raw),
                len(xyz_ma_raw),
                source_tag,
            )
        else:
            if hybrid_heading and frames:
                # Legacy opt-out: inject A seeds into gated score_planar_hyps
                seed_hyps = hypothesize_vertical_planes(
                    frames,
                    xyz_a if len(xyz_a) >= 30 else None,
                    ground_z=ground_z_a,
                    max_heading_seeds=int(max_heading_seeds),
                )
                log.info(
                    "facades Path α hybrid (seeds): injecting %s A heading "
                    "seeds into scorer (--no-hybrid-a-full-search)",
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
                nms_xy_m=nms_xy,
                nms_xy_split_m=nms_xy_split,
                union_strategy=strategy,
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
                "(ma_kept=%s a_kept=%s; strategy=%s; ply=%s pts; source=%s; "
                "promote vs product 7/5/0.42)",
                len(hyps),
                len(seed_hyps),
                len(accepted),
                ma_n,
                a_n,
                strategy,
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
                xyz_a if len(xyz_a) >= 30 else None,
                zncc_accept=min(float(zncc_accept), 0.35),
                ground_z=ground_z_a,
                max_keep=n_planes,
            )
            if accepted:
                source_tag = "photo_consistency_fallback"
                a_kept_n = len(accepted)
    else:
        if planarize is True and len(xyz_ma_raw) < 100:
            log.warning(
                "facades: planarize requested but MA PLY too thin (%s pts) — "
                "falling back to Milestone A heading×distance on a_ply=%s",
                len(xyz_ma_raw),
                a_ply_str or "(none)",
            )
        product_json = dest_obj.parent / "planes.json"
        flow_thin = len(xyz_a) < 30
        use_lock = a_source == "product" or (
            a_source in {"flow", "recon"}
            and flow_thin
            and product_json.is_file()
        )
        if use_lock:
            accepted = planes_from_product_json(product_json)
            if accepted:
                source_tag = "product_lock"
                a_kept_n = len(accepted)
                log.info(
                    "facades A-only: product_lock %s planes from %s",
                    len(accepted),
                    product_json,
                )
            elif a_source == "product":
                accepted = []
                log.warning(
                    "facades: --a-source product but no usable planes at %s",
                    product_json,
                )
            else:
                accepted = search_photo_consistent_planes(
                    frames,
                    xyz_a if len(xyz_a) >= 30 else None,
                    zncc_accept=float(zncc_accept),
                    ground_z=ground_z_a,
                    max_keep=n_planes,
                )
                a_kept_n = len(accepted)
        else:
            accepted = search_photo_consistent_planes(
                frames,
                xyz_a if len(xyz_a) >= 30 else None,
                zncc_accept=float(zncc_accept),
                ground_z=ground_z_a,
                max_keep=n_planes,
            )
            a_kept_n = len(accepted)

    planes: list[dict] = []
    for pl in accepted:
        planes.append(plane_dict_for_obj(pl, ground_z))

    if ps1_rectify and planes:
        from ps1_hood.reconstruct.ps1_facades import manhattan_rectify_planes

        planes = manhattan_rectify_planes(planes)
        log.info("facades: PS1 manhattan_rectify_planes on %s accepts", len(planes))

    dest_obj.parent.mkdir(parents=True, exist_ok=True)
    tex_dir = dest_obj.parent / "textures"
    mtl_path = dest_obj.with_suffix(".mtl")
    planes_json = dest_obj.parent / "planes.json"

    # Ground extents from flow/A cloud (historic) or MA / camera AABB
    if len(xyz_a) >= 4:
        extent_xyz = xyz_a
    elif len(xyz) >= 4:
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

    prev_product = bool(
        keep_previous_on_fail and _facade_product_looks_nonempty(dest_obj)
    )
    kind = _resolve_output_kind(
        output_kind,
        control_out=control_out,
        use_planarize=use_planarize,
        prev_product=prev_product,
        dest_obj=dest_obj,
    )
    tel = {
        "a_ply": a_ply_str,
        "ma_ply": ma_ply_str,
        "a_kept": int(a_kept_n),
        "ma_added": int(ma_added_n),
        "output_kind": kind,
    }

    # --- FAIL-LOUD preserve: do not wipe prior product ---
    # Control runs never use *.failed / *.candidate — they write *.control.
    if (
        kind != "control"
        and not planes
        and keep_previous_on_fail
        and _facade_product_looks_nonempty(dest_obj)
    ):
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
            **tel,
        }

    # Success path — candidate / product / control staging
    prev_q = _read_facade_quality(dest_obj) if prev_product else None
    stage_candidate = bool(kind == "candidate" and prev_q is not None and planes)
    skip_promote = kind == "control"

    if kind == "control":
        paths = _facade_output_paths(dest_obj, "control")
        out_obj = Path(paths["obj"])
        out_mtl = Path(paths["mtl"])
        out_json = Path(paths["json"])
        bake_tex_dir = Path(paths["bake_tex"])
        tex_map_prefix = str(paths["tex_prefix"])
    elif stage_candidate:
        paths = _facade_output_paths(dest_obj, "candidate")
        out_obj = Path(paths["obj"])
        out_mtl = Path(paths["mtl"])
        out_json = Path(paths["json"])
        bake_tex_dir = Path(paths["bake_tex"])
        tex_map_prefix = str(paths["tex_prefix"])
    else:
        out_obj = dest_obj
        out_mtl = mtl_path
        out_json = planes_json
        bake_tex_dir = tex_dir
        tex_map_prefix = "textures"
        # Wipe-on-write only when replacing the live product.
        # Gap-fill bake-new-only: never delete existing facade_00..N.jpg.
        tex_dir.mkdir(parents=True, exist_ok=True)
        if tex_dir.is_dir() and not (gap_fill and gap_bake_lock_n > 0):
            for old_tex in tex_dir.glob("facade_*.jpg"):
                old_tex.unlink(missing_ok=True)

    bake_tex_dir.mkdir(parents=True, exist_ok=True)
    bake_new_only = bool(gap_fill and gap_bake_lock_n > 0 and kind != "control")
    if kind in {"candidate", "control"}:
        for old_tex in bake_tex_dir.glob("facade_*.jpg"):
            old_tex.unlink(missing_ok=True)
        # Gap-fill: seed candidate with existing product textures for locked planes
        # so we never depend on re-baking 18/18 (and never touch product on fail).
        if bake_new_only and tex_dir.is_dir():
            import shutil

            for i in range(int(gap_bake_lock_n)):
                src = tex_dir / f"facade_{i:02d}.jpg"
                if src.is_file():
                    shutil.copy2(src, bake_tex_dir / src.name)

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
        # Gap-fill: keep locked product textures; bake *new* indices only
        if bake_new_only and i < int(gap_bake_lock_n):
            existing = bake_tex_dir / f"facade_{i:02d}.jpg"
            product_jpg = tex_dir / f"facade_{i:02d}.jpg"
            if existing.is_file() or product_jpg.is_file():
                if not existing.is_file() and product_jpg.is_file():
                    import shutil

                    shutil.copy2(product_jpg, existing)
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
            continue
        if frames:
            cam = _pick_frontal_camera(pl, quad, frames)
            if cam is not None:
                tex_path = bake_tex_dir / f"facade_{i:02d}.jpg"
                if _warp_facade_texture(
                    quad, cam, tex_path, ps1_tex_size=ps1_tex_size
                ):
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

    if skip_promote or kind == "control":
        log.info(
            "facades: control write %s / %s (planes=%s textured=%s "
            "a_ply=%s ma_ply=%s a_kept=%s ma_added=%s) — "
            "did not touch product or candidate",
            out_obj.name,
            out_json.name,
            len(planes),
            textured,
            a_ply_str,
            ma_ply_str,
            a_kept_n,
            ma_added_n,
        )
        return {
            "path": str(dest_obj),
            "mtl": str(mtl_path),
            "planes_json": str(planes_json),
            "planes": int(prev_q.get("plane_count") or 0) if prev_q else len(planes),
            "textured": int(prev_q.get("textured") or 0) if prev_q else textured,
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
            "mean_zncc": prev_q.get("mean_zncc") if prev_q else mean_zncc,
            "path_alpha": source_tag == "mapanything_planarize",
            "preserved_previous": bool(prev_q),
            "rejected_weaker": False,
            "ok": bool(prev_q) or len(planes) > 0,
            "control_obj": str(out_obj),
            "control_planes_json": str(out_json),
            "control_planes": int(len(planes)),
            "control_textured": int(textured),
            "control_mean_zncc": mean_zncc,
            **tel,
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
                **tel,
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
        **tel,
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


def _bare_plane_indices(
    recon_dir: Path,
    planes: list[dict[str, Any]],
    *,
    mtl_path: Path | None = None,
) -> list[int]:
    """Indices where texture JPG is missing or MTL lacks map_Kd."""
    recon_dir = Path(recon_dir)
    tex_dir = recon_dir / "textures"
    mtl_path = Path(mtl_path) if mtl_path is not None else recon_dir / "facades.mtl"
    map_ok: set[str] = set()
    if mtl_path.is_file():
        block_name = None
        has_map = False
        for line in mtl_path.read_text(encoding="ascii", errors="ignore").splitlines():
            s = line.strip()
            if s.startswith("newmtl "):
                if block_name and has_map:
                    map_ok.add(block_name)
                block_name = s.split(None, 1)[1]
                has_map = False
            elif s.lower().startswith("map_kd "):
                has_map = True
        if block_name and has_map:
            map_ok.add(block_name)

    bare: list[int] = []
    for i, pl in enumerate(planes):
        name = f"facade_{i:02d}"
        jpg = tex_dir / f"{name}.jpg"
        tex_field = pl.get("texture") if isinstance(pl, dict) else None
        has_jpg = jpg.is_file() and jpg.stat().st_size > 32
        if tex_field and not has_jpg:
            # planes.json claims a map but file gone → bare
            bare.append(i)
            continue
        if not has_jpg:
            bare.append(i)
            continue
        if map_ok and name not in map_ok:
            bare.append(i)
            continue
        if not tex_field and name not in map_ok and not has_jpg:
            bare.append(i)
    return bare


def _ensure_plane_nxny(pl: dict[str, Any]) -> dict[str, Any]:
    """Return plane dict with nx/ny/d_xy for frontal pick / warp."""
    out = dict(pl)
    if "nx" in out:
        return out
    if out.get("n") is not None:
        n = np.asarray(out["n"], dtype=np.float64).reshape(-1)
        out["nx"], out["ny"] = float(n[0]), float(n[1])
        # planes.json stores n·X+d=0 → facades d_xy = -d
        if "d" in out:
            out["d"] = float(-float(out["d"]))
    return out


def _backup_pre_retex(recon_dir: Path) -> dict[str, str]:
    """Copy product façades artefacts to ``*.pre_retex`` (once)."""
    import shutil

    recon_dir = Path(recon_dir)
    backed: dict[str, str] = {}
    pairs = [
        (recon_dir / "facades.obj", recon_dir / "facades.obj.pre_retex"),
        (recon_dir / "facades.mtl", recon_dir / "facades.mtl.pre_retex"),
        (recon_dir / "planes.json", recon_dir / "planes.json.pre_retex"),
    ]
    for src, dst in pairs:
        if src.is_file() and not dst.is_file():
            shutil.copy2(src, dst)
            backed[src.name] = str(dst)
    tex = recon_dir / "textures"
    bak_tex = recon_dir / "textures.pre_retex"
    if tex.is_dir() and not bak_tex.is_dir():
        shutil.copytree(tex, bak_tex)
        backed["textures"] = str(bak_tex)
    return backed


def retexture_bare_planes(
    run: Path,
    *,
    margin_px: float = 120.0,
    top_k_cams: int = 5,
    ps1_tex_size: int | None = 128,
    frames: list[dict[str, Any]] | None = None,
    bare_only: bool = True,
    candidate: bool = False,
    in_place: bool = False,
    dry_run: bool = False,
    promote: bool = True,
) -> dict[str, Any]:
    """Re-bake ortho textures for bare façades only — no gap-fill / planarize / densify.

    Identifies bare = missing ``textures/facade_XX.jpg`` OR MTL without ``map_Kd``.
    Tries top-K frontal cameras (preferring ``view_indices`` when present) with a
    loosened warp margin. Stages as ``facades.candidate.*`` or ``--in-place`` with
    ``*.pre_retex`` backup. Quality-keep: promote when textured rises and plane
    count is unchanged (clause 1).
    """
    import json
    import logging
    import shutil

    from ps1_hood.reconstruct.planarize import write_planes_json
    from ps1_hood.reconstruct.ps1_facades import load_product_planes

    log = logging.getLogger(__name__)
    run = Path(run)
    if (run / "recon" / "planes.json").is_file() or (run / "recon").is_dir():
        recon_dir = run / "recon"
    elif run.name == "recon" or (run / "planes.json").is_file() or (run / "facades.obj").is_file():
        recon_dir = run
    else:
        recon_dir = run / "recon"

    dest_obj = recon_dir / "facades.obj"
    mtl_path = recon_dir / "facades.mtl"
    planes_json = recon_dir / "planes.json"
    tex_dir = recon_dir / "textures"

    planes, prev_meta = load_product_planes(recon_dir)
    if not planes:
        raise FileNotFoundError(f"no product planes under {recon_dir}")

    # Normalize nx/ny for pick/warp
    planes = [_ensure_plane_nxny(p) for p in planes]
    bare = _bare_plane_indices(recon_dir, planes, mtl_path=mtl_path)
    if not bare_only:
        # Escape hatch: treat all as candidates for bake (still won't wipe good JPGs
        # unless warp succeeds — we skip indices that already have map_Kd+JPG when
        # bare_only is the default path).
        bare = list(range(len(planes)))

    prev_q = _read_facade_quality(dest_obj) or {
        "textured": max(0, len(planes) - len(bare)),
        "mean_zncc": (prev_meta or {}).get("gates", {}).get("mean_zncc")
        if isinstance(prev_meta, dict)
        else None,
        "plane_count": len(planes),
    }
    # Prefer counting from current bare set for the pre snapshot
    n_tex_before = sum(
        1
        for i in range(len(planes))
        if (tex_dir / f"facade_{i:02d}.jpg").is_file()
        and i not in bare
    )
    # Also count bare that somehow already have jpg+map (shouldn't be in bare)
    n_tex_before = int(prev_q.get("textured") or n_tex_before)

    report = {
        "recon_dir": str(recon_dir),
        "planes": len(planes),
        "bare": bare,
        "bare_ids": [f"facade_{i:02d}" for i in bare],
        "textured_before": n_tex_before,
        "dry_run": bool(dry_run),
        "candidate": bool(candidate),
        "in_place": bool(in_place),
        "baked": [],
        "failed": [],
        "ok": True,
    }

    if dry_run:
        log.info(
            "facades-retexture dry-run: %s bare of %s planes → %s",
            len(bare),
            len(planes),
            report["bare_ids"],
        )
        report["would_bake"] = report["bare_ids"]
        return report

    if candidate and in_place:
        raise ValueError("pass only one of candidate= / in_place=")
    if not candidate and not in_place:
        # Default to candidate staging (sacred: don't clobber product)
        candidate = True
        report["candidate"] = True

    frames = list(frames or [])
    if not frames:
        raise RuntimeError(
            "retexture_bare_planes requires frames= (CLI loads align/cameras.json)"
        )

    if in_place:
        report["backup"] = _backup_pre_retex(recon_dir)
        paths = _facade_output_paths(dest_obj, "product")
    else:
        paths = _facade_output_paths(dest_obj, "candidate")

    out_obj = Path(paths["obj"])
    out_mtl = Path(paths["mtl"])
    out_json = Path(paths["json"])
    bake_tex_dir = Path(paths["bake_tex"])
    tex_map_prefix = str(paths["tex_prefix"])
    bake_tex_dir.mkdir(parents=True, exist_ok=True)

    # Seed bake dir with existing good textures so promote / candidate is complete.
    if candidate:
        for old in bake_tex_dir.glob("facade_*.jpg"):
            old.unlink(missing_ok=True)
        if tex_dir.is_dir():
            for src in sorted(tex_dir.glob("facade_*.jpg")):
                # Skip bare targets — they will be re-baked (or stay missing)
                try:
                    idx = int(src.stem.split("_")[1])
                except (IndexError, ValueError):
                    idx = -1
                if idx in bare:
                    continue
                shutil.copy2(src, bake_tex_dir / src.name)

    materials: list[dict[str, Any]] = []
    # Preserve ground material from existing MTL when possible
    ground_map = None
    if (tex_dir / "ground.jpg").is_file():
        ground_map = "textures/ground.jpg"
    materials.append(
        {"name": "ground", "map": ground_map, "kd": (0.35, 0.38, 0.32)}
    )

    baked: list[int] = []
    failed: list[int] = []
    tex_maps: list[str | None] = []

    for i, pl in enumerate(planes):
        quad = pl.get("quad") or pl.get("corners") or _plane_quad(pl)
        quad_t = [tuple(map(float, c)) for c in quad]
        mat_name = f"facade_{i:02d}"
        map_rel: str | None = None
        existing = bake_tex_dir / f"facade_{i:02d}.jpg"
        product_jpg = tex_dir / f"facade_{i:02d}.jpg"

        if bare_only and i not in bare:
            # Keep good texture — copy for in-place is no-op; candidate already seeded
            if in_place and product_jpg.is_file():
                map_rel = f"textures/facade_{i:02d}.jpg"
            elif existing.is_file():
                map_rel = f"{tex_map_prefix}/facade_{i:02d}.jpg"
            elif product_jpg.is_file():
                if candidate:
                    shutil.copy2(product_jpg, existing)
                map_rel = f"{tex_map_prefix}/facade_{i:02d}.jpg"
            materials.append(
                {"name": mat_name, "map": map_rel, "kd": (0.55, 0.50, 0.42), "quad": quad_t}
            )
            tex_maps.append(map_rel)
            continue

        prefer = pl.get("view_indices")
        if prefer is not None and not isinstance(prefer, list):
            prefer = list(prefer) if prefer else None
        cams = _rank_frontal_cameras(
            pl,
            quad_t,
            frames,
            top_k=int(top_k_cams),
            prefer_indices=prefer,
        )
        ok = False
        tex_path = bake_tex_dir / f"facade_{i:02d}.jpg"
        for cam in cams:
            if _warp_facade_texture(
                quad_t,
                cam,
                tex_path,
                ps1_tex_size=ps1_tex_size,
                margin_px=float(margin_px),
                allow_clip=True,
            ):
                ok = True
                break
        if ok:
            map_rel = f"{tex_map_prefix}/facade_{i:02d}.jpg"
            baked.append(i)
        else:
            failed.append(i)
            # Keep prior map if any (don't wipe)
            if product_jpg.is_file() and i not in bare:
                map_rel = f"{tex_map_prefix}/facade_{i:02d}.jpg"
            else:
                map_rel = None
        materials.append(
            {"name": mat_name, "map": map_rel, "kd": (0.55, 0.50, 0.42), "quad": quad_t}
        )
        tex_maps.append(map_rel)

    # Extent for ground quad — reuse obj verts if present else quad AABB
    extent_xyz = np.zeros((0, 3), dtype=np.float64)
    for pl in planes:
        q = pl.get("quad") or pl.get("corners")
        if q is not None:
            extent_xyz = (
                np.asarray(q, dtype=np.float64)
                if extent_xyz.size == 0
                else np.vstack([extent_xyz, np.asarray(q, dtype=np.float64)])
            )
    if extent_xyz.size == 0:
        extent_xyz = np.array([[-10, -10, 0], [10, 10, 0]], dtype=np.float64)

    if in_place:
        # Only rewrite MTL map_Kd lines + planes.json texture fields; keep obj geometry.
        _write_mtl(out_mtl, materials)
        # Obj mtllib / usemtl unchanged — maps live in mtl
    else:
        _write_mtl(out_mtl, materials)
        _write_obj(out_obj, planes, extent_xyz, materials, out_mtl.name)

    ground_z = float((prev_meta or {}).get("ground_z") or 0.0)
    source = str((prev_meta or {}).get("source") or "photo_consistency")
    residual = int((prev_meta or {}).get("residual_points") or 0)
    write_planes_json(
        out_json,
        planes=planes,
        ground_z=ground_z,
        residual_points=residual,
        source=source,
        textured_maps=tex_maps,
    )

    n_tex_after = sum(1 for m in materials[1:] if m.get("map"))
    new_q = {
        "textured": int(n_tex_after),
        "mean_zncc": prev_q.get("mean_zncc"),
        "plane_count": int(len(planes)),
    }
    report["baked"] = baked
    report["failed"] = failed
    report["textured_after"] = n_tex_after
    report["obj"] = str(out_obj if not in_place else dest_obj)
    report["mtl"] = str(out_mtl)
    report["planes_json"] = str(out_json)
    report["new_quality"] = new_q
    report["prev_quality"] = prev_q

    promoted = False
    promote_reason = None
    if candidate and promote and baked:
        better, why = _is_strictly_better(new_q, prev_q)
        report["quality_keep"] = {"better": better, "reason": why}
        if better:
            _promote_candidate_facades(
                dest_obj,
                cand_obj=out_obj,
                cand_mtl=out_mtl,
                cand_json=out_json,
                cand_tex_dir=bake_tex_dir,
            )
            promoted = True
            promote_reason = why
            log.info(
                "facades-retexture: promoted candidate (%s) textured %s→%s planes=%s",
                why,
                n_tex_before,
                n_tex_after,
                len(planes),
            )
        else:
            log.info(
                "facades-retexture: candidate NOT promoted — %s "
                "(baked=%s failed=%s); product kept",
                why,
                baked,
                failed,
            )
    elif in_place:
        log.info(
            "facades-retexture in-place: baked=%s failed=%s textured %s→%s",
            baked,
            failed,
            n_tex_before,
            n_tex_after,
        )

    report["promoted"] = promoted
    report["promote_reason"] = promote_reason
    report["ok"] = bool(baked) or n_tex_after >= n_tex_before
    return report
