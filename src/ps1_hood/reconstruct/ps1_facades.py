"""PS1 façade simplify — Manhattan rectify + RGB555 nearest textures (PR1).

Sacred: keep photo ZNCC centers / normals; sat XY unchanged; cosmetic only
(no quality-keep re-run for post-process).
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np

log = logging.getLogger(__name__)

WIDTH_CLAMP_M = (3.0, 20.0)
HEIGHT_CLAMP_M = (2.5, 15.0)


def _as_n_xy(pl: dict[str, Any]) -> np.ndarray:
    """Horizontal unit normal (nx, ny, 0) from plane dict."""
    if "n" in pl and pl["n"] is not None:
        n = np.asarray(pl["n"], dtype=np.float64).reshape(-1)
    elif "nx" in pl:
        n = np.array([float(pl["nx"]), float(pl["ny"]), 0.0], dtype=np.float64)
    else:
        raise ValueError("plane missing n / nx,ny")
    n_xy = np.array([float(n[0]), float(n[1]), 0.0], dtype=np.float64)
    nrm = float(np.linalg.norm(n_xy[:2]) + 1e-12)
    n_xy[:2] /= nrm
    return n_xy


def _plane_center(pl: dict[str, Any], n: np.ndarray) -> np.ndarray:
    """Photo-derived center: explicit center, else quad mean, else n·X+d=0."""
    if pl.get("center") is not None:
        c = np.asarray(pl["center"], dtype=np.float64).reshape(3)
    else:
        quad = pl.get("quad") or pl.get("corners")
        if quad is not None:
            c = np.mean(np.asarray(quad, dtype=np.float64).reshape(-1, 3), axis=0)
        else:
            # facades dict: nx*e+ny*n = d_xy → n·X + d = 0 with d = -d_xy
            if "nx" in pl and "d" in pl:
                d_plane = float(-pl["d"])
            else:
                d_plane = float(pl.get("d", 0.0))
            c = (-d_plane) * n
            gz = float(pl.get("ground_z", 0.0))
            h = float(pl.get("height_m") or 6.0)
            c = np.array([c[0], c[1], gz + 0.5 * h], dtype=np.float64)
    # Project onto vertical plane through n (preserve XY lock along normal)
    if "nx" in pl and "d" in pl:
        d_plane = float(-pl["d"])
    elif "d" in pl:
        d_plane = float(pl["d"])
    else:
        d_plane = float(-(n @ c))
    c = c - (n @ c + d_plane) * n
    return c.astype(np.float64)


def _width_height_m(pl: dict[str, Any], n: np.ndarray, center: np.ndarray) -> tuple[float, float]:
    """Width/height from fields or from projecting current corners onto (right, up)."""
    w = float(pl.get("width_m") or 0.0)
    h = float(pl.get("height_m") or 0.0)
    right = np.array([-n[1], n[0], 0.0], dtype=np.float64)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    quad = pl.get("quad") or pl.get("corners")
    if quad is not None:
        pts = np.asarray(quad, dtype=np.float64).reshape(-1, 3)
        rel = pts - center.reshape(1, 3)
        along = rel @ right
        vert = rel @ up
        if len(along) >= 2:
            # percentile 5–95 then force equal opposite edges (true rectangle)
            lo_a, hi_a = np.percentile(along, [5, 95])
            lo_v, hi_v = np.percentile(vert, [5, 95])
            w_q = float(hi_a - lo_a)
            h_q = float(hi_v - lo_v)
            if w <= 1e-3:
                w = w_q
            if h <= 1e-3:
                h = h_q
    if w <= 1e-3:
        # fallback from min/max AABB
        if "min" in pl and "max" in pl:
            mn = np.asarray(pl["min"], dtype=np.float64)
            mx = np.asarray(pl["max"], dtype=np.float64)
            w = float(np.linalg.norm((mx - mn)[:2]))
        else:
            w = 6.0
    if h <= 1e-3:
        if "min" in pl and "max" in pl:
            h = float(abs(pl["max"][2] - pl["min"][2]))
        else:
            h = 6.0
    w = float(np.clip(w, WIDTH_CLAMP_M[0], WIDTH_CLAMP_M[1]))
    h = float(np.clip(h, HEIGHT_CLAMP_M[0], HEIGHT_CLAMP_M[1]))
    return w, h


def _optional_snap_n_to_street(n: np.ndarray, street_heading_deg: float | None) -> np.ndarray:
    """Snap n_xy to nearest 90° of dominant street heading (±90° partners)."""
    if street_heading_deg is None:
        return n
    # Street travel heading → wall normals are heading±90°
    rad = np.deg2rad(float(street_heading_deg))
    candidates = [
        np.array([np.cos(rad + np.pi / 2), np.sin(rad + np.pi / 2), 0.0]),
        np.array([np.cos(rad - np.pi / 2), np.sin(rad - np.pi / 2), 0.0]),
        np.array([np.cos(rad), np.sin(rad), 0.0]),
        np.array([np.cos(rad + np.pi), np.sin(rad + np.pi), 0.0]),
    ]
    best = max(candidates, key=lambda c: float(abs(np.dot(c[:2], n[:2]))))
    # Preserve original facing hemisphere
    if float(np.dot(best[:2], n[:2])) < 0:
        best = -best
    best = best.astype(np.float64)
    best[:2] /= float(np.linalg.norm(best[:2]) + 1e-12)
    return best


def manhattan_rectify_planes(
    planes: list[dict[str, Any]],
    *,
    street_heading_deg: float | None = None,
) -> list[dict[str, Any]]:
    """Rebuild each accepted plane as an ENU Manhattan rectangle about photo center.

    Keeps photo-derived **center** and **n** (optionally snapped to street 90°).
    Corners order: BL, BR, TR, TL. Opposite edges are equal (true rectangle).
    """
    out: list[dict[str, Any]] = []
    for pl in planes:
        pl2 = dict(pl)
        n = _as_n_xy(pl2)
        n = _optional_snap_n_to_street(n, street_heading_deg)
        center = _plane_center(pl2, n)
        width_m, height_m = _width_height_m(pl2, n, center)
        right = np.array([-n[1], n[0], 0.0], dtype=np.float64)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        hw, hh = width_m / 2.0, height_m / 2.0
        corners = np.stack(
            [
                center - hw * right - hh * up,  # BL
                center + hw * right - hh * up,  # BR
                center + hw * right + hh * up,  # TR
                center - hw * right + hh * up,  # TL
            ],
            axis=0,
        )
        # facades._plane_quad form: nx*e + ny*n = d_xy
        d_xy = float(n[0] * center[0] + n[1] * center[1])
        d_plane = float(-(n @ center))  # n·X + d = 0
        pl2["n"] = [float(n[0]), float(n[1]), 0.0]
        pl2["nx"] = float(n[0])
        pl2["ny"] = float(n[1])
        pl2["d"] = d_xy  # facades-style d_xy (plane_dict_for_obj / _plane_quad)
        pl2["d_plane"] = d_plane
        pl2["center"] = [float(center[0]), float(center[1]), float(center[2])]
        pl2["width_m"] = float(width_m)
        pl2["height_m"] = float(height_m)
        pl2["quad"] = [tuple(map(float, c)) for c in corners]
        pl2["corners"] = pl2["quad"]
        pl2["min"] = corners.min(axis=0).tolist()
        pl2["max"] = corners.max(axis=0).tolist()
        pl2["ps1_rectified"] = True
        out.append(pl2)
    return out


def ps1_quantize(img: np.ndarray) -> np.ndarray:
    """RGB555-style bit crush: drop low 3 bits per channel (OpenCV BGR ok)."""
    if img is None or img.size == 0:
        return img
    out = np.asarray(img)
    if out.dtype != np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)
    return ((out >> 3) << 3).astype(np.uint8)


def nearest_resize(img: np.ndarray, tex_size: int = 128) -> np.ndarray:
    """Nearest-neighbor resize to 128² or 128×256 (aspect-aware, INTER_NEAREST)."""
    if img is None or img.size == 0:
        return img
    h, w = img.shape[:2]
    ts = max(8, int(tex_size))
    aspect = float(w) / float(max(h, 1))
    if aspect >= 1.5:
        tw, th = ts * 2, ts  # wide → 256×128 when tex_size=128
    elif aspect <= (1.0 / 1.5):
        tw, th = ts, ts * 2  # tall → 128×256
    else:
        tw, th = ts, ts
    # Cap long side at 2×tex_size
    tw = int(min(tw, ts * 2))
    th = int(min(th, ts * 2))
    return cv2.resize(img, (tw, th), interpolation=cv2.INTER_NEAREST)


def apply_ps1_texture(img: np.ndarray, tex_size: int = 128) -> np.ndarray:
    """Nearest resize then RGB555 quantize (PS1 look)."""
    return ps1_quantize(nearest_resize(img, tex_size=tex_size))


def _parse_facades_obj_planes(obj_path: Path) -> list[dict[str, Any]]:
    """Recover plane dicts from facades.obj quads when planes.json is missing."""
    text = Path(obj_path).read_text(encoding="ascii", errors="replace")
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[list[int], str | None]] = []
    cur_mat: str | None = None
    for line in text.splitlines():
        if line.startswith("v "):
            parts = line.split()
            verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
        elif line.startswith("usemtl "):
            cur_mat = line.split(None, 1)[1].strip()
        elif line.startswith("f "):
            idxs = []
            for tok in line.split()[1:]:
                idxs.append(int(tok.split("/")[0]) - 1)
            faces.append((idxs, cur_mat))
    planes: list[dict[str, Any]] = []
    for idxs, mat in faces:
        if mat is None or not mat.startswith("facade_"):
            continue
        if len(idxs) < 4:
            continue
        quad = [verts[i] for i in idxs[:4]]
        pts = np.asarray(quad, dtype=np.float64)
        center = pts.mean(axis=0)
        # Normal from BL→BR × BL→TL (or cross of edges)
        e1 = pts[1] - pts[0]
        e2 = pts[3] - pts[0]
        n = np.cross(e1, e2)
        n[2] = 0.0
        nrm = float(np.linalg.norm(n[:2]) + 1e-12)
        n[:2] /= nrm
        width_m = float(np.linalg.norm(pts[1][:2] - pts[0][:2]))
        height_m = float(abs(pts[3][2] - pts[0][2]))
        d_xy = float(n[0] * center[0] + n[1] * center[1])
        planes.append(
            {
                "nx": float(n[0]),
                "ny": float(n[1]),
                "n": [float(n[0]), float(n[1]), 0.0],
                "d": d_xy,
                "center": center.tolist(),
                "quad": [tuple(map(float, c)) for c in quad],
                "width_m": width_m,
                "height_m": height_m,
                "min": pts.min(axis=0).tolist(),
                "max": pts.max(axis=0).tolist(),
                "ground_z": float(pts[:, 2].min()),
                "source": "obj_parse",
                "zncc": None,
            }
        )
    return planes


def load_product_planes(recon_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Load planes from planes.json, else parse facades.obj."""
    recon_dir = Path(recon_dir)
    planes_json = recon_dir / "planes.json"
    meta: dict[str, Any] | None = None
    if planes_json.is_file():
        payload = json.loads(planes_json.read_text(encoding="utf-8"))
        meta = payload if isinstance(payload, dict) else None
        raw = (payload.get("planes") if isinstance(payload, dict) else payload) or []
        planes: list[dict[str, Any]] = []
        for pl in raw:
            if not isinstance(pl, dict):
                continue
            p = dict(pl)
            if "n" in p and "nx" not in p:
                n = np.asarray(p["n"], dtype=np.float64)
                p["nx"], p["ny"] = float(n[0]), float(n[1])
                # planes.json stores n·X+d=0 → facades d_xy = -d
                if "d" in p:
                    p["d"] = float(-p["d"])
            planes.append(p)
        if planes:
            return planes, meta
    obj = recon_dir / "facades.obj"
    if obj.is_file():
        return _parse_facades_obj_planes(obj), meta
    return [], meta


def rewrite_facades_obj_quads(
    obj_path: Path,
    planes: list[dict[str, Any]],
    *,
    backup: bool = True,
) -> None:
    """Rewrite façade vertex blocks in an existing OBJ (keep ground + mtllib/uv/faces)."""
    obj_path = Path(obj_path)
    if backup and obj_path.is_file():
        bak = obj_path.with_suffix(obj_path.suffix + ".bak")
        if not bak.is_file():
            shutil.copy2(obj_path, bak)
    text = obj_path.read_text(encoding="ascii", errors="replace")
    lines = text.splitlines(keepends=True)
    v_idxs: list[int] = []
    for i, line in enumerate(lines):
        if line.startswith("v "):
            v_idxs.append(i)
    if len(v_idxs) < 4:
        raise RuntimeError(f"no vertices in {obj_path}")
    # First 4 verts = ground; remaining groups of 4 = façades
    n_facade_verts = len(v_idxs) - 4
    n_expected = len(planes) * 4
    if n_facade_verts < n_expected:
        raise RuntimeError(
            f"obj has {n_facade_verts} façade verts, need {n_expected} for {len(planes)} planes"
        )
    for pi, pl in enumerate(planes):
        quad = pl.get("quad") or pl.get("corners")
        if quad is None:
            continue
        for k, corner in enumerate(quad):
            li = v_idxs[4 + pi * 4 + k]
            c = corner
            lines[li] = f"v {float(c[0]):.3f} {float(c[1]):.3f} {float(c[2]):.3f}\n"
    obj_path.write_text("".join(lines), encoding="ascii")


def postprocess_run(
    recon_dir: Path,
    *,
    tex_size: int = 128,
    street_heading_deg: float | None = None,
    backup_obj: bool = True,
    quantize_textures: bool = True,
) -> dict[str, Any]:
    """Rectify planes.json (or obj) + nearest/RGB555 textures + rewrite obj — no re-extract."""
    from ps1_hood.reconstruct.planarize import write_planes_json

    recon_dir = Path(recon_dir)
    planes, prev_meta = load_product_planes(recon_dir)
    if not planes:
        raise FileNotFoundError(f"no planes/obj façades under {recon_dir}")

    planes = manhattan_rectify_planes(planes, street_heading_deg=street_heading_deg)

    tex_dir = recon_dir / "textures"
    n_tex = 0
    if quantize_textures and tex_dir.is_dir():
        for i in range(len(planes)):
            path = tex_dir / f"facade_{i:02d}.jpg"
            if not path.is_file():
                continue
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            out = apply_ps1_texture(img, tex_size=tex_size)
            cv2.imwrite(str(path), out, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            n_tex += 1

    obj_path = recon_dir / "facades.obj"
    if obj_path.is_file():
        rewrite_facades_obj_quads(obj_path, planes, backup=backup_obj)

    ground_z = float((prev_meta or {}).get("ground_z") or planes[0].get("ground_z") or 0.0)
    source = str((prev_meta or {}).get("source") or "")
    if not source or source in {"obj_parse", "ps1_facades_post"}:
        # Prefer live scene / prior product tag; cosmetic post must not invent a new family
        scene_path = recon_dir / "scene.json"
        if scene_path.is_file():
            try:
                scene = json.loads(scene_path.read_text(encoding="utf-8"))
                fac = scene.get("facades") or (scene.get("cloud") or {}).get("facades") or {}
                if isinstance(fac, dict) and fac.get("source"):
                    source = str(fac["source"])
            except Exception:  # noqa: BLE001
                pass
    if not source:
        source = str(planes[0].get("source") or "photo_consistency")
    residual = int((prev_meta or {}).get("residual_points") or 0)
    tex_maps = []
    for i, pl in enumerate(planes):
        tex = pl.get("texture")
        if tex is None and (tex_dir / f"facade_{i:02d}.jpg").is_file():
            tex = f"textures/facade_{i:02d}.jpg"
        tex_maps.append(tex)

    # write_planes_json expects facades-style d_xy when nx present — we have that
    write_planes_json(
        recon_dir / "planes.json",
        planes=planes,
        ground_z=ground_z,
        residual_points=residual,
        source=source,
        textured_maps=tex_maps,
    )

    return {
        "planes": len(planes),
        "textures_rewritten": n_tex,
        "obj": str(obj_path) if obj_path.is_file() else None,
        "planes_json": str(recon_dir / "planes.json"),
        "ps1_tex_size": int(tex_size),
        "ok": True,
    }
