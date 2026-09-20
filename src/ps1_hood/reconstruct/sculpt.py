"""Studio ENU façade sculpt — nudge along normal + resize w×h + bak writeback.

Sacred: sat absolute XY (translate only along plane n), no free-pose,
never wipe product planes without bak. Mapillary garage is out of scope.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

MAX_DELTA_D_M = 3.0
MAX_TANGENT_SLIDE_M = 2.0
BAK_PREFIX_PLANES = "planes.json.bak_sculpt_"
BAK_PREFIX_OBJ = "facades.obj.bak_sculpt_"
BAK_PREFIX_MTL = "facades.mtl.bak_sculpt_"
BAK_TEX_DIR_PREFIX = "bak_sculpt_"


def _unit_horizontal(n: np.ndarray) -> np.ndarray:
    v = np.asarray(n, dtype=np.float64).reshape(-1)
    if v.size < 2:
        raise ValueError("plane normal needs at least nx,ny")
    out = np.array([float(v[0]), float(v[1]), 0.0], dtype=np.float64)
    nrm = float(np.linalg.norm(out[:2]) + 1e-12)
    out[:2] /= nrm
    return out


def facade_frame(n: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (axis_n, axis_t, axis_u) in ENU metres — vertical rect basis."""
    axis_n = _unit_horizontal(n)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    # along wall: normalize(cross(up, n))
    axis_t = np.cross(up, axis_n)
    tn = float(np.linalg.norm(axis_t) + 1e-12)
    axis_t = axis_t / tn
    return axis_n, axis_t, up


def corners_from_center(
    center: np.ndarray,
    n: np.ndarray,
    width_m: float,
    height_m: float,
) -> list[tuple[float, float, float]]:
    """Vertical Manhattan-ish rect: center ± (w/2)·t ± (h/2)·u."""
    axis_n, axis_t, axis_u = facade_frame(n)
    c = np.asarray(center, dtype=np.float64).reshape(3)
    hw = float(width_m) * 0.5
    hh = float(height_m) * 0.5
    # BL, BR, TR, TL (matching existing product quad winding: bottom-left → …)
    bl = c - hw * axis_t - hh * axis_u
    br = c + hw * axis_t - hh * axis_u
    tr = c + hw * axis_t + hh * axis_u
    tl = c - hw * axis_t + hh * axis_u
    return [tuple(map(float, p)) for p in (bl, br, tr, tl)]


def plane_center_from_quad(quad: list | np.ndarray) -> np.ndarray:
    pts = np.asarray(quad, dtype=np.float64)
    return pts.mean(axis=0)


def recompute_d(n: np.ndarray, center: np.ndarray) -> float:
    """planes.json convention: n·X + d = 0."""
    nn = _unit_horizontal(n)
    c = np.asarray(center, dtype=np.float64).reshape(3)
    return float(-float(nn @ c))


def _find_plane_index(planes: list[dict[str, Any]], plane_id: str) -> int:
    for i, pl in enumerate(planes):
        pid = str(pl.get("id") or f"facade_{i:02d}")
        if pid == plane_id or pid.endswith(plane_id) or plane_id.endswith(pid):
            return i
        # allow bare index "3" / "03"
        if plane_id.isdigit() and i == int(plane_id):
            return i
        if plane_id.replace("facade_", "").isdigit() and i == int(
            plane_id.replace("facade_", "")
        ):
            return i
    raise KeyError(f"plane not found: {plane_id}")


def _load_planes_payload(recon_dir: Path) -> dict[str, Any]:
    path = Path(recon_dir) / "planes.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("planes"), list):
        raise RuntimeError(f"invalid planes.json at {path}")
    return payload


def _stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def bak_sculpt(recon_dir: Path, *, stamp: str | None = None) -> dict[str, str]:
    """Snapshot planes.json / facades.obj+mtl / textures → bak_sculpt_<ts>."""
    recon_dir = Path(recon_dir)
    ts = stamp or _stamp()
    backed: dict[str, str] = {"stamp": ts}
    pairs = [
        (recon_dir / "planes.json", recon_dir / f"{BAK_PREFIX_PLANES}{ts}"),
        (recon_dir / "facades.obj", recon_dir / f"{BAK_PREFIX_OBJ}{ts}"),
        (recon_dir / "facades.mtl", recon_dir / f"{BAK_PREFIX_MTL}{ts}"),
    ]
    for src, dst in pairs:
        if src.is_file():
            shutil.copy2(src, dst)
            backed[src.name] = str(dst)
    tex = recon_dir / "textures"
    bak_tex = recon_dir / "textures" / f"{BAK_TEX_DIR_PREFIX}{ts}"
    if tex.is_dir():
        bak_tex.mkdir(parents=True, exist_ok=True)
        for jpg in sorted(tex.glob("facade_*.jpg")):
            shutil.copy2(jpg, bak_tex / jpg.name)
        # also ground if present
        g = tex / "ground.jpg"
        if g.is_file():
            shutil.copy2(g, bak_tex / g.name)
        backed["textures"] = str(bak_tex)
    return backed


def _latest_bak_stamp(recon_dir: Path) -> str | None:
    recon_dir = Path(recon_dir)
    stamps: list[str] = []
    for p in recon_dir.glob(f"{BAK_PREFIX_PLANES}*"):
        stamps.append(p.name[len(BAK_PREFIX_PLANES) :])
    if not stamps:
        return None
    return sorted(stamps)[-1]


def undo_sculpt(recon_dir: Path, *, stamp: str | None = None) -> dict[str, Any]:
    """Restore latest (or given) bak_sculpt_* snapshot. Never deletes other planes."""
    recon_dir = Path(recon_dir)
    ts = stamp or _latest_bak_stamp(recon_dir)
    if not ts:
        raise FileNotFoundError(f"no bak_sculpt_* under {recon_dir}")
    restored: list[str] = []
    pairs = [
        (recon_dir / f"{BAK_PREFIX_PLANES}{ts}", recon_dir / "planes.json"),
        (recon_dir / f"{BAK_PREFIX_OBJ}{ts}", recon_dir / "facades.obj"),
        (recon_dir / f"{BAK_PREFIX_MTL}{ts}", recon_dir / "facades.mtl"),
    ]
    for src, dst in pairs:
        if src.is_file():
            shutil.copy2(src, dst)
            restored.append(dst.name)
    bak_tex = recon_dir / "textures" / f"{BAK_TEX_DIR_PREFIX}{ts}"
    tex = recon_dir / "textures"
    if bak_tex.is_dir() and tex.is_dir():
        for jpg in bak_tex.glob("*.jpg"):
            shutil.copy2(jpg, tex / jpg.name)
            restored.append(f"textures/{jpg.name}")
    return {"ok": True, "stamp": ts, "restored": restored}


def mutate_plane(
    pl: dict[str, Any],
    *,
    delta_d: float | None = None,
    delta_t: float | None = None,
    width_m: float | None = None,
    height_m: float | None = None,
    corners: list | None = None,
) -> dict[str, Any]:
    """Return updated plane dict (n fixed; sat XY seat via n-only translate)."""
    out = dict(pl)
    n = _unit_horizontal(out.get("n") or [out.get("nx"), out.get("ny"), 0.0])
    out["n"] = [float(n[0]), float(n[1]), 0.0]

    if corners is not None:
        quad = [tuple(map(float, c)) for c in corners]
        center = plane_center_from_quad(quad)
        # derive w/h from corners
        _, axis_t, axis_u = facade_frame(n)
        pts = np.asarray(quad, dtype=np.float64)
        # project extents onto t/u
        rel = pts - center
        width_m = float(2.0 * max(abs(float(r @ axis_t)) for r in rel))
        height_m = float(2.0 * max(abs(float(r @ axis_u)) for r in rel))
    else:
        quad = out.get("quad") or out.get("corners")
        if quad is None:
            raise ValueError("plane has no quad/corners")
        pts = np.asarray(quad, dtype=np.float64)
        center = pts.mean(axis=0)
        do_resize = width_m is not None or height_m is not None
        width_m = float(width_m if width_m is not None else out.get("width_m") or 1.0)
        height_m = float(
            height_m if height_m is not None else out.get("height_m") or 1.0
        )
        # Prefer translating existing corners (preserve winding/UVs). Rebuild only on resize.
        shift = np.zeros(3, dtype=np.float64)
        if delta_d is not None:
            dd = float(np.clip(float(delta_d), -MAX_DELTA_D_M, MAX_DELTA_D_M))
            shift = shift + dd * n
        if delta_t is not None:
            _, axis_t, _ = facade_frame(n)
            dt = float(
                np.clip(float(delta_t), -MAX_TANGENT_SLIDE_M, MAX_TANGENT_SLIDE_M)
            )
            shift = shift + dt * axis_t
        if do_resize:
            center = center + shift
            quad = corners_from_center(center, n, width_m, height_m)
        else:
            pts = pts + shift
            quad = [tuple(map(float, p)) for p in pts]

    center = plane_center_from_quad(quad)
    out["quad"] = [list(map(float, c)) for c in quad]
    out["corners"] = out["quad"]
    out["center"] = [float(center[0]), float(center[1]), float(center[2])]
    out["width_m"] = float(width_m)
    out["height_m"] = float(height_m)
    out["d"] = recompute_d(n, center)
    out["sculpt_edit"] = True
    # keep nx/ny consistent if present
    out["nx"] = float(n[0])
    out["ny"] = float(n[1])
    return out


def _patch_mtl_map(mtl_path: Path, mat_name: str, map_rel: str) -> None:
    if not mtl_path.is_file():
        return
    lines = mtl_path.read_text(encoding="ascii", errors="replace").splitlines(keepends=True)
    out: list[str] = []
    in_block = False
    wrote_map = False
    for line in lines:
        s = line.strip()
        if s.startswith("newmtl "):
            if in_block and not wrote_map:
                out.append(f"map_Kd {map_rel}\n")
            in_block = s.split(None, 1)[1].strip() == mat_name
            wrote_map = False
            out.append(line)
            continue
        if in_block and s.lower().startswith("map_kd "):
            out.append(f"map_Kd {map_rel}\n")
            wrote_map = True
            continue
        out.append(line)
    if in_block and not wrote_map:
        out.append(f"map_Kd {map_rel}\n")
    mtl_path.write_text("".join(out), encoding="ascii")


def _pick_bake_frame(
    pl: dict[str, Any],
    frames: list[dict[str, Any]],
    bake_cam: str | None,
) -> dict[str, Any] | None:
    from ps1_hood.reconstruct.facades import _ensure_plane_nxny, _rank_frontal_cameras

    if not frames:
        return None
    if bake_cam:
        for fr in frames:
            if str(fr.get("pano_id") or "") == bake_cam:
                return fr
            if Path(str(fr.get("path") or "")).name == bake_cam:
                return fr
            if str(fr.get("path") or "") == bake_cam:
                return fr
        log.warning("sculpt bake_cam %s not in known frames; falling back to frontal", bake_cam)
    quad = pl.get("quad") or pl.get("corners")
    if not quad:
        return None
    ranked = _rank_frontal_cameras(
        _ensure_plane_nxny(pl),
        [tuple(map(float, c)) for c in quad],
        frames,
        min_frontal=0.05,
        top_k=5,
    )
    return ranked[0] if ranked else None


def apply_sculpt(
    recon_dir: Path,
    plane_id: str,
    *,
    delta_d: float | None = None,
    delta_t: float | None = None,
    width_m: float | None = None,
    height_m: float | None = None,
    corners: list | None = None,
    n: list[float] | None = None,
    d: float | None = None,
    bake: bool = True,
    bake_cam: str | None = None,
    frames: list[dict[str, Any]] | None = None,
    margin_px: float = 120.0,
    ps1_tex_size: int | None = 128,
) -> dict[str, Any]:
    """Bak → mutate one plane → rewrite planes.json + OBJ (+ optional warp bake).

    Does **not** run peel quality-keep. Other planes untouched. Bake fail soft:
    geometry still saved with previous texture.
    """
    from ps1_hood.reconstruct.facades import _warp_facade_texture
    from ps1_hood.reconstruct.ps1_facades import rewrite_facades_obj_quads

    recon_dir = Path(recon_dir)
    payload = _load_planes_payload(recon_dir)
    planes: list[dict[str, Any]] = list(payload["planes"])
    n_before = len(planes)
    idx = _find_plane_index(planes, plane_id)
    pl = dict(planes[idx])
    pid = str(pl.get("id") or f"facade_{idx:02d}")

    # Optional absolute n/d from client — n must stay horizontal unit; ignore free rotate
    if n is not None:
        nn = _unit_horizontal(n)
        pl["n"] = [float(nn[0]), float(nn[1]), 0.0]
    if corners is None and d is not None and delta_d is None:
        # absolute planes.json d (n·X+d=0): center' = center + (d_old - d_new)·n
        cur_c = plane_center_from_quad(pl.get("quad") or pl["corners"])
        cur_d = recompute_d(pl["n"], cur_c)
        delta_d = cur_d - float(d)

    backed = bak_sculpt(recon_dir)
    updated = mutate_plane(
        pl,
        delta_d=delta_d,
        delta_t=delta_t,
        width_m=width_m,
        height_m=height_m,
        corners=corners,
    )
    updated["id"] = pid
    planes[idx] = updated
    if len(planes) != n_before:
        raise RuntimeError("sculpt refused: plane count changed (would wipe product)")

    # Write planes.json preserving top-level meta; only replace planes list
    out_payload = dict(payload)
    out_payload["planes"] = planes
    planes_path = recon_dir / "planes.json"
    planes_path.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")

    obj_path = recon_dir / "facades.obj"
    if obj_path.is_file():
        # rewrite_facades_obj_quads expects nx/d_xy style or quad on each plane
        rewrite_facades_obj_quads(obj_path, planes, backup=False)

    bake_ok = False
    bake_path = None
    bake_error = None
    if bake:
        try:
            fr = _pick_bake_frame(updated, frames or [], bake_cam)
            if fr is None:
                bake_error = "no known frontal camera"
            else:
                tex_rel = f"textures/facade_{idx:02d}.jpg"
                dest = recon_dir / tex_rel
                quad = [tuple(map(float, c)) for c in updated["quad"]]
                ok = _warp_facade_texture(
                    quad,
                    fr,
                    dest,
                    ps1_tex_size=ps1_tex_size,
                    margin_px=float(margin_px),
                    allow_clip=True,
                )
                if ok:
                    bake_ok = True
                    bake_path = tex_rel
                    updated["texture"] = tex_rel
                    planes[idx] = updated
                    out_payload["planes"] = planes
                    planes_path.write_text(
                        json.dumps(out_payload, indent=2), encoding="utf-8"
                    )
                    _patch_mtl_map(
                        recon_dir / "facades.mtl",
                        f"facade_{idx:02d}",
                        tex_rel,
                    )
                else:
                    bake_error = "warp failed (kept prior jpg)"
        except Exception as exc:  # noqa: BLE001
            bake_error = str(exc)
            log.warning("sculpt bake soft-fail: %s", exc)

    return {
        "ok": True,
        "plane_id": pid,
        "index": idx,
        "plane_count": len(planes),
        "bak": backed,
        "plane": {
            "id": pid,
            "n": updated["n"],
            "d": updated["d"],
            "center": updated["center"],
            "corners": updated["quad"],
            "width_m": updated["width_m"],
            "height_m": updated["height_m"],
            "texture": updated.get("texture"),
            "sculpt_edit": True,
        },
        "bake_ok": bake_ok,
        "bake_path": bake_path,
        "bake_error": bake_error,
    }
