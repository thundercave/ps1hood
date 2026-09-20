"""Pano ↔ mesh compare (diagnose-only).

Reproject product façade/roof quads into SV crops with known ENU poses,
score ZNCC (photo ortho patch vs mesh tex/gray) + edge Chamfer, and
project footprints onto Ortho for sat XY Chamfer. Writes side-by-side
overlays + ``recon/compare/summary.json`` ranked by worst ZNCC.

Sacred: no free-pose, no densify, no product wipe — diagnose only.
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
from ps1_hood.reconstruct.photo_planes import (
    K_from_frame,
    P_from_Rt,
    Rt_from_frame,
    load_view,
    project_points,
    zncc,
)

log = logging.getLogger(__name__)

DEFAULT_PATCH = 64
DEFAULT_MAX_CAMS = 40
SOFT_ZNCC_WARN = 0.35
MIN_CORNERS = 3


class CompareError(RuntimeError):
    """Fail-loud compare (empty product / no cameras / nothing visible)."""


# ---------------------------------------------------------------------------
# Load product + cameras
# ---------------------------------------------------------------------------


def _resolve_shot(frame: dict[str, Any], project_root: Path) -> Path | None:
    raw = frame.get("path") or frame.get("shot_path")
    if not raw:
        return None
    p = Path(str(raw))
    if p.is_file():
        return p
    name = p.name
    parent = p.parent.name
    for cand in (
        project_root / "cropped" / parent / name,
        project_root / "cropped" / name,
        project_root / Path(*p.parts[-3:]),
    ):
        if cand.is_file():
            return cand
    return None


def load_cameras(project_root: Path) -> list[dict[str, Any]]:
    """Horizon keyframes from ``align/cameras.json`` with readable crops."""
    path = Path(project_root) / "align" / "cameras.json"
    if not path.is_file():
        raise CompareError(f"no align/cameras.json at {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise CompareError(f"empty cameras.json at {path}")
    out: list[dict[str, Any]] = []
    for fr in raw:
        if not isinstance(fr, dict):
            continue
        if abs(float(fr.get("pitch") or 0.0)) > 8.0:
            continue
        shot = _resolve_shot(fr, Path(project_root))
        if shot is None:
            continue
        fr2 = dict(fr)
        fr2["path"] = str(shot)
        fr2["shot_path"] = str(shot)
        out.append(fr2)
    if not out:
        raise CompareError("no horizon cameras with readable pano crops")
    return out


def subsample_cameras(
    frames: list[dict[str, Any]], max_cams: int
) -> list[dict[str, Any]]:
    """Unique (pano, heading≈45°) slots; even stride to max_cams."""
    seen: set[tuple[str, int]] = set()
    uniq: list[dict[str, Any]] = []
    for fr in frames:
        pid = str(fr.get("pano_id") or fr.get("path") or id(fr))
        h = int(round(float(fr.get("heading") or 0.0) / 45.0) * 45) % 360
        key = (pid, h)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(fr)
    if max_cams <= 0 or len(uniq) <= max_cams:
        return uniq
    step = len(uniq) / float(max_cams)
    return [uniq[int(i * step)] for i in range(max_cams)]


def load_facade_quads(project_root: Path) -> list[dict[str, Any]]:
    path = Path(project_root) / "recon" / "planes.json"
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    planes = payload.get("planes") if isinstance(payload, dict) else None
    if not isinstance(planes, list):
        return []
    recon = Path(project_root) / "recon"
    out: list[dict[str, Any]] = []
    for i, pl in enumerate(planes):
        if not isinstance(pl, dict):
            continue
        quad = pl.get("quad") or pl.get("corners")
        if not quad or len(quad) < 4:
            continue
        corners = np.asarray(quad[:4], dtype=np.float64).reshape(4, 3)
        n_raw = pl.get("n")
        if n_raw is None:
            e1 = corners[1] - corners[0]
            e2 = corners[3] - corners[0]
            n = np.cross(e1, e2)
            n = n / (np.linalg.norm(n) + 1e-12)
        else:
            n = np.asarray(n_raw, dtype=np.float64).reshape(3)
            n = n / (np.linalg.norm(n) + 1e-12)
        d = float(pl["d"]) if "d" in pl else float(-n @ corners.mean(axis=0))
        pid = str(pl.get("id") or f"facade_{i:02d}")
        tex_path: Path | None = None
        tex = pl.get("texture")
        if tex:
            cand = recon / str(tex)
            if cand.is_file():
                tex_path = cand
        if tex_path is None:
            for alt in (
                recon / "textures" / f"{pid}.jpg",
                recon / "textures" / f"facade_{i:02d}.jpg",
            ):
                if alt.is_file():
                    tex_path = alt
                    break
        out.append(
            {
                "id": pid,
                "kind": "facade",
                "n": n,
                "d": d,
                "corners": corners,
                "texture": tex_path,
            }
        )
    return out


def load_roof_quads(project_root: Path) -> list[dict[str, Any]]:
    path = Path(project_root) / "recon" / "roofs.json"
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    shells = payload.get("shells") if isinstance(payload, dict) else None
    if not isinstance(shells, list):
        return []
    tex_root = Path(project_root) / "recon" / "textures"
    out: list[dict[str, Any]] = []
    for sh in shells:
        if not isinstance(sh, dict):
            continue
        aabb = sh.get("aabb_enu") or sh.get("polygon_enu")
        if not aabb or len(aabb) < 4:
            continue
        z = float(sh.get("z") or 0.0)
        xy = [(float(p[0]), float(p[1])) for p in aabb[:4]]
        corners = np.array(
            [
                [xy[0][0], xy[0][1], z],
                [xy[1][0], xy[1][1], z],
                [xy[2][0], xy[2][1], z],
                [xy[3][0], xy[3][1], z],
            ],
            dtype=np.float64,
        )
        sid = str(sh.get("id") or "roof")
        tex_path = None
        for cand in (
            tex_root / f"roof_{sid}.jpg",
            tex_root / f"{sid}.jpg",
        ):
            if cand.is_file():
                tex_path = cand
                break
        out.append(
            {
                "id": sid,
                "kind": str(sh.get("kind") or "roof"),
                "n": np.array([0.0, 0.0, 1.0], dtype=np.float64),
                "d": float(-z),
                "corners": corners,
                "texture": tex_path,
            }
        )
    return out


def load_product_quads(project_root: Path) -> list[dict[str, Any]]:
    facades = load_facade_quads(project_root)
    roofs = load_roof_quads(project_root)
    if not facades and not roofs:
        raise CompareError(
            "no product quads: need recon/planes.json and/or recon/roofs.json"
        )
    return facades + roofs


# ---------------------------------------------------------------------------
# Projection / scoring
# ---------------------------------------------------------------------------


def project_quad(
    frame: dict[str, Any],
    corners: np.ndarray,
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Project 4×3 ENU corners → uv (4×2) + front mask (known pose only)."""
    K = K_from_frame(width, height, float(frame.get("fov") or 90.0))
    Rcw, t = Rt_from_frame(frame)
    P = P_from_Rt(K, Rcw, t)
    return project_points(P, np.asarray(corners, dtype=np.float64).reshape(-1, 3))


def quad_visible(
    uv: np.ndarray,
    front: np.ndarray,
    width: int,
    height: int,
    *,
    min_corners: int = MIN_CORNERS,
    margin: float = 8.0,
) -> bool:
    if int(np.asarray(front).sum()) < min_corners:
        return False
    inside = (
        front
        & (uv[:, 0] >= -margin)
        & (uv[:, 0] < width + margin)
        & (uv[:, 1] >= -margin)
        & (uv[:, 1] < height + margin)
    )
    return int(inside.sum()) >= min_corners


def _plane_color(qid: str) -> tuple[int, int, int]:
    h = abs(hash(qid)) % (180 * 1000)
    hue = (h // 1000) % 180
    sat = 160 + (h % 70)
    val = 200 + (h % 55)
    bgr = cv2.cvtColor(np.uint8([[[hue, sat, val]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _ortho_patches(
    photo_bgr: np.ndarray,
    uv: np.ndarray,
    tex_bgr: np.ndarray | None,
    *,
    patch: int = DEFAULT_PATCH,
) -> tuple[np.ndarray, np.ndarray | None] | None:
    h, w = photo_bgr.shape[:2]
    src = uv.astype(np.float32).copy()
    src[:, 0] = np.clip(src[:, 0], -40.0, w + 40.0)
    src[:, 1] = np.clip(src[:, 1], -40.0, h + 40.0)
    dst = np.array(
        [[0, patch - 1], [patch - 1, patch - 1], [patch - 1, 0], [0, 0]],
        dtype=np.float32,
    )
    try:
        H = cv2.getPerspectiveTransform(src, dst)
    except cv2.error:
        return None
    photo_gray = cv2.cvtColor(photo_bgr, cv2.COLOR_BGR2GRAY)
    photo_p = cv2.warpPerspective(photo_gray, H, (patch, patch), flags=cv2.INTER_LINEAR)
    if float((photo_p > 0).mean()) < 0.25:
        return None
    if tex_bgr is None or tex_bgr.size == 0:
        # No texture → no ZNCC (constant gray would always NaN via zero variance).
        return photo_p, None
    th, tw = tex_bgr.shape[:2]
    tex_src = np.array(
        [[0, th - 1], [tw - 1, th - 1], [tw - 1, 0], [0, 0]],
        dtype=np.float32,
    )
    try:
        H_tex = cv2.getPerspectiveTransform(tex_src, dst)
        tex_gray = (
            cv2.cvtColor(tex_bgr, cv2.COLOR_BGR2GRAY)
            if tex_bgr.ndim == 3
            else tex_bgr
        )
        mesh_p = cv2.warpPerspective(
            tex_gray, H_tex, (patch, patch), flags=cv2.INTER_LINEAR
        )
    except cv2.error:
        return photo_p, None
    return photo_p, mesh_p


def edge_chamfer_agree(
    photo_bgr: np.ndarray, uv: np.ndarray, *, sigma_px: float = 4.0
) -> float:
    """Agreement [0,1] = exp(-mean_DT/sigma) of projected edges vs photo Canny."""
    h, w = photo_bgr.shape[:2]
    edge_img = np.zeros((h, w), dtype=np.uint8)
    pts = np.round(uv).astype(np.int32)
    for i in range(4):
        cv2.line(
            edge_img,
            tuple(pts[i]),
            tuple(pts[(i + 1) % 4]),
            255,
            1,
            lineType=cv2.LINE_AA,
        )
    gray = cv2.cvtColor(photo_bgr, cv2.COLOR_BGR2GRAY)
    photo_e = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 60, 160)
    if int((photo_e > 0).sum()) < 40:
        return float("nan")
    inv = np.where(photo_e > 0, 0, 255).astype(np.uint8)
    dt = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    ys, xs = np.where(edge_img > 0)
    if len(ys) < 8:
        return float("nan")
    m = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    if int(m.sum()) < 8:
        return float("nan")
    mean_d = float(dt[ys[m], xs[m]].mean())
    if not math.isfinite(mean_d):
        return float("nan")
    return float(math.exp(-mean_d / max(sigma_px, 1e-3)))


def score_quad_in_view(
    photo_bgr: np.ndarray,
    frame: dict[str, Any],
    quad: dict[str, Any],
    *,
    patch: int = DEFAULT_PATCH,
) -> dict[str, Any] | None:
    h, w = photo_bgr.shape[:2]
    corners = np.asarray(quad["corners"], dtype=np.float64)
    uv, front = project_quad(frame, corners, width=w, height=h)
    if not quad_visible(uv, front, w, h):
        return None
    tex = None
    if quad.get("texture") is not None:
        tex = cv2.imread(str(quad["texture"]), cv2.IMREAD_COLOR)
    patches = _ortho_patches(photo_bgr, uv, tex, patch=patch)
    if patches is None:
        return None
    photo_p, mesh_p = patches
    if mesh_p is None:
        z = float("nan")
    else:
        z = zncc(photo_p.astype(np.float64), mesh_p.astype(np.float64))
        z = float(z) if z == z else float("nan")
    edge = edge_chamfer_agree(photo_bgr, uv)
    return {
        "id": quad["id"],
        "kind": quad.get("kind") or "facade",
        "zncc": z,
        "edge": float(edge) if edge == edge else float("nan"),
        "uv": uv,
        "has_texture": mesh_p is not None,
    }


def render_mesh_panel(
    photo_bgr: np.ndarray,
    quads: list[dict[str, Any]],
    scored: list[dict[str, Any]],
) -> np.ndarray:
    h, w = photo_bgr.shape[:2]
    canvas = np.zeros_like(photo_bgr)
    by_id = {s["id"]: s for s in scored}
    for q in quads:
        s = by_id.get(q["id"])
        if s is None:
            continue
        uv = np.asarray(s["uv"], dtype=np.float32)
        color = _plane_color(q["id"])
        pts = np.round(uv).astype(np.int32).reshape(-1, 1, 2)
        if q.get("texture") is not None:
            tex = cv2.imread(str(q["texture"]), cv2.IMREAD_COLOR)
            if tex is not None:
                th, tw = tex.shape[:2]
                tex_src = np.array(
                    [[0, th - 1], [tw - 1, th - 1], [tw - 1, 0], [0, 0]],
                    dtype=np.float32,
                )
                try:
                    H = cv2.getPerspectiveTransform(tex_src, uv)
                    warped = cv2.warpPerspective(tex, H, (w, h), flags=cv2.INTER_LINEAR)
                    mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillConvexPoly(mask, pts, 255)
                    canvas[mask > 0] = warped[mask > 0]
                    tint = np.zeros_like(photo_bgr)
                    cv2.fillConvexPoly(tint, pts, color)
                    canvas[mask > 0] = cv2.addWeighted(
                        canvas[mask > 0], 0.75, tint[mask > 0], 0.25, 0
                    )
                    cv2.polylines(canvas, [pts], True, (255, 255, 255), 1)
                    continue
                except cv2.error:
                    pass
        fill = np.zeros_like(photo_bgr)
        cv2.fillConvexPoly(fill, pts, color, lineType=cv2.LINE_AA)
        canvas = cv2.addWeighted(canvas, 1.0, fill, 0.85, 0)
        cv2.polylines(canvas, [pts], True, (255, 255, 255), 1)
    return canvas


def _hud(img: np.ndarray, lines: list[str]) -> None:
    y = 22
    for line in lines:
        for col, thick in (((0, 0, 0), 3), ((240, 240, 240), 1)):
            cv2.putText(
                img,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                col,
                thick,
                cv2.LINE_AA,
            )
        y += 20


def make_overlay_strip(
    photo_bgr: np.ndarray,
    mesh_bgr: np.ndarray,
    *,
    zncc_mean: float,
    edge_mean: float,
    n_quads: int,
    cam_id: str,
) -> np.ndarray:
    """Left photo | middle mesh | right abs-diff+edge HUD."""
    h, w = photo_bgr.shape[:2]
    mesh = mesh_bgr if mesh_bgr.shape[:2] == (h, w) else cv2.resize(mesh_bgr, (w, h))
    mask = (mesh.sum(axis=2) > 8).astype(np.uint8)
    diff = cv2.absdiff(photo_bgr, mesh)
    diff[mask == 0] = photo_bgr[mask == 0] // 3
    edge_panel = photo_bgr.copy()
    edges = cv2.Canny(cv2.cvtColor(mesh, cv2.COLOR_BGR2GRAY), 40, 120)
    edge_panel[edges > 0] = (0, 255, 255)
    right = cv2.addWeighted(diff, 0.7, edge_panel, 0.3, 0)

    target_h = 480
    scale = target_h / float(h)
    tw = max(1, int(round(w * scale)))
    left = cv2.resize(photo_bgr, (tw, target_h))
    mid = cv2.resize(mesh, (tw, target_h))
    right = cv2.resize(right, (tw, target_h))
    strip = np.concatenate([left, mid, right], axis=1)
    z_s = f"{zncc_mean:.3f}" if zncc_mean == zncc_mean else "nan"
    e_s = f"{edge_mean:.3f}" if edge_mean == edge_mean else "nan"
    _hud(
        strip,
        [
            f"{cam_id}  quads={n_quads}  zncc={z_s}  edge={e_s}",
            "photo | mesh reproject | absdiff+edge",
        ],
    )
    return strip


# ---------------------------------------------------------------------------
# Sat footprint Chamfer
# ---------------------------------------------------------------------------


def sat_footprint_edge_mean_m(
    project_root: Path, quads: list[dict[str, Any]]
) -> dict[str, Any]:
    """Project façade/roof XY footprints onto Ortho; Chamfer vs sat Canny (m)."""
    from ps1_hood.reconstruct.sat_roofs import load_ortho_for_run

    try:
        ortho = load_ortho_for_run(Path(project_root))
    except Exception as exc:  # noqa: BLE001 — diagnose soft
        return {"ok": False, "reason": str(exc), "edge_mean_m": float("nan")}

    m_per_px_e = abs(ortho.ee - ortho.sw) / max(ortho.w - 1, 1)
    m_per_px_n = abs(ortho.nn - ortho.sh) / max(ortho.h - 1, 1)
    m_per_px = 0.5 * (m_per_px_e + m_per_px_n)

    sat_edges = ortho_canny(ortho.image)
    if int((sat_edges > 0).sum()) < 40:
        return {"ok": False, "reason": "too few sat edges", "edge_mean_m": float("nan")}

    inv = np.where(sat_edges > 0, 0, 255).astype(np.uint8)
    dt = cv2.distanceTransform(inv, cv2.DIST_L2, 3)

    footprint = np.zeros((ortho.h, ortho.w), dtype=np.uint8)
    n_drawn = 0
    for q in quads:
        corners = np.asarray(q["corners"], dtype=np.float64)
        px = [ortho.enu_to_px(float(c[0]), float(c[1])) for c in corners]
        pts = np.round(np.asarray(px, dtype=np.float64)).astype(np.int32)
        if len(pts) < 3:
            continue
        cv2.polylines(footprint, [pts.reshape(-1, 1, 2)], True, 255, 1)
        n_drawn += 1
    ys, xs = np.where(footprint > 0)
    if len(ys) < 20 or n_drawn == 0:
        return {
            "ok": False,
            "reason": "no footprints projected onto ortho",
            "edge_mean_m": float("nan"),
            "n_quads": n_drawn,
        }
    mean_px = float(dt[ys, xs].mean())
    return {
        "ok": True,
        "edge_mean_m": mean_px * m_per_px,
        "edge_mean_px": mean_px,
        "m_per_px": m_per_px,
        "n_quads": n_drawn,
        "n_samples": int(len(ys)),
    }


# ---------------------------------------------------------------------------
# Per-cam + run
# ---------------------------------------------------------------------------


def make_cam_id(frame: dict[str, Any]) -> str:
    pid = str(frame.get("pano_id") or "cam")
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in pid)
    h = int(round(float(frame.get("heading") or 0.0))) % 360
    return f"{safe}_h{h:03d}"


def compare_camera(
    frame: dict[str, Any],
    quads: list[dict[str, Any]],
    *,
    patch: int = DEFAULT_PATCH,
) -> dict[str, Any] | None:
    view = load_view(frame)
    if view is None:
        return None
    photo = view.image_bgr
    scored: list[dict[str, Any]] = []
    for q in quads:
        r = score_quad_in_view(photo, frame, q, patch=patch)
        if r is not None:
            scored.append(r)
    if not scored:
        return None

    znccs = [s["zncc"] for s in scored if s["zncc"] == s["zncc"]]
    edges = [s["edge"] for s in scored if s["edge"] == s["edge"]]
    zncc_mean = float(np.mean(znccs)) if znccs else float("nan")
    edge_mean = float(np.mean(edges)) if edges else float("nan")
    mesh = render_mesh_panel(photo, quads, scored)
    cid = make_cam_id(frame)
    strip = make_overlay_strip(
        photo,
        mesh,
        zncc_mean=zncc_mean,
        edge_mean=edge_mean,
        n_quads=len(scored),
        cam_id=cid,
    )
    return {
        "id": cid,
        "pano_id": frame.get("pano_id"),
        "heading": float(frame.get("heading") or 0.0),
        "zncc_mean": zncc_mean,
        "edge_mean": edge_mean,
        "n_quads": len(scored),
        "quads": [
            {
                "id": s["id"],
                "kind": s["kind"],
                "zncc": s["zncc"],
                "edge": s["edge"],
                "has_texture": bool(s.get("has_texture")),
            }
            for s in scored
        ],
        "overlay": strip,
    }


def run_compare(
    project_root: Path,
    *,
    max_cams: int = DEFAULT_MAX_CAMS,
    out_dir: Path | None = None,
    patch: int = DEFAULT_PATCH,
) -> dict[str, Any]:
    """Diagnose-only pano↔mesh compare. Writes overlays + summary.json."""
    root = Path(project_root)
    quads = load_product_quads(root)
    frames = subsample_cameras(load_cameras(root), max_cams)
    out = Path(out_dir) if out_dir else root / "recon" / "compare"
    out.mkdir(parents=True, exist_ok=True)

    cam_rows: list[dict[str, Any]] = []
    for fr in frames:
        result = compare_camera(fr, quads, patch=patch)
        if result is None:
            continue
        overlay_name = f"{result['id']}.jpg"
        cv2.imwrite(
            str(out / overlay_name),
            result["overlay"],
            [int(cv2.IMWRITE_JPEG_QUALITY), 88],
        )
        cam_rows.append(
            {
                "id": result["id"],
                "pano_id": result["pano_id"],
                "heading": result["heading"],
                "zncc_mean": result["zncc_mean"],
                "edge_mean": result["edge_mean"],
                "n_quads": result["n_quads"],
                "overlay": overlay_name,
                "quads": result["quads"],
            }
        )

    if not cam_rows:
        raise CompareError(
            "compare produced 0 scored views (no quads visible in any camera); "
            "check poses / planes / crops"
        )

    def _sort_key(r: dict[str, Any]) -> float:
        # Finite ascending (true worst first). NaN → +inf so unscored/no-tex
        # cams do not monopolize worst[:10] (gray-mesh ZNCC was always NaN).
        z = r["zncc_mean"]
        return z if z == z else float("inf")

    ranked = sorted(cam_rows, key=_sort_key)
    finite = [r for r in ranked if r["zncc_mean"] == r["zncc_mean"]]
    worst = finite[:10]
    znccs = [r["zncc_mean"] for r in cam_rows if r["zncc_mean"] == r["zncc_mean"]]
    edges = [r["edge_mean"] for r in cam_rows if r["edge_mean"] == r["edge_mean"]]
    global_zncc = float(np.mean(znccs)) if znccs else float("nan")
    global_edge = float(np.mean(edges)) if edges else float("nan")
    sat = sat_footprint_edge_mean_m(root, quads)
    soft_warn = bool(global_zncc == global_zncc and global_zncc < SOFT_ZNCC_WARN)

    summary: dict[str, Any] = {
        "run": root.name,
        "n_cams_scored": len(cam_rows),
        "n_cams_considered": len(frames),
        "n_quads_product": len(quads),
        "n_facades": sum(1 for q in quads if q.get("kind") == "facade"),
        "n_roofs": sum(1 for q in quads if q.get("kind") != "facade"),
        "cams": ranked,
        "worst": [
            {
                "id": w["id"],
                "zncc_mean": w["zncc_mean"],
                "edge_mean": w["edge_mean"],
                "n_quads": w["n_quads"],
                "overlay": w["overlay"],
            }
            for w in worst
        ],
        "global": {
            "zncc_mean": global_zncc,
            "edge_mean": global_edge,
            "sat_edge_mean_m": sat.get("edge_mean_m"),
            "soft_zncc_warn": soft_warn,
            "soft_zncc_threshold": SOFT_ZNCC_WARN,
            "n_cams_finite_zncc": len(znccs),
            "n_cams_nan_zncc": len(cam_rows) - len(znccs),
        },
        "sat_footprint": sat,
        "diagnose_only": True,
        "out_dir": str(out),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
    )
    log.info(
        "compare: scored %s cams, global zncc=%.3f sat_edge_m=%s → %s",
        len(cam_rows),
        global_zncc if global_zncc == global_zncc else float("nan"),
        sat.get("edge_mean_m"),
        out,
    )
    return summary
