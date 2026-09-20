"""Forced SE(2) from red façade edges → yellow Ortho Canny (escape no-op seat).

Sacred: one rigid SE(2) · no free-pose · bak first · prefer sat XY.
Reuse ``fit_se2`` / ``apply_se2_*`` / ``seat_recon_artefacts`` patterns.
Skip roofs/street when already sat-native.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ps1_hood.align.georef import (
    apply_se2_pose,
    apply_se2_to_ascii_obj,
    apply_se2_to_planes_json,
    apply_se2_to_ply,
    apply_se2_xy,
    fit_se2,
    refresh_scene_cameras,
)
from ps1_hood.align.sat_edges import ortho_canny
from ps1_hood.project import Project
from ps1_hood.reconstruct.sat_roofs import load_ortho_for_run, px_to_enu

log = logging.getLogger(__name__)

DEFAULT_SEARCH_R_M = 8.0
DEFAULT_MIN_EDGE_M = 3.0
DEFAULT_MAX_RMS_M = 1.5
DEFAULT_MIN_PAIRS = 4
DEFAULT_MAX_YAW_DEG = 15.0
DEFAULT_MAX_TRANSLATION_M = 10.0
SOURCE_FACADE_SAT_CHAMFER = "facade_sat_chamfer"

DEFAULT_MS_DE_M = 6.0
DEFAULT_MS_DYAW_DEG = 8.0
DEFAULT_MS_STEP_M = 2.0
DEFAULT_MS_STEP_YAW = 4.0


class SatOffsetError(RuntimeError):
    """Fail-loud measure / apply errors."""


def _load_planes(recon_dir: Path) -> list[dict[str, Any]]:
    path = Path(recon_dir) / "planes.json"
    if not path.is_file():
        raise SatOffsetError(f"missing {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    planes = data if isinstance(data, list) else (data.get("planes") or data.get("facades") or [])
    if not isinstance(planes, list) or not planes:
        raise SatOffsetError(f"no planes in {path}")
    return [p for p in planes if isinstance(p, dict)]


def facade_long_edges(
    planes: list[dict[str, Any]],
    *,
    min_len_m: float = DEFAULT_MIN_EDGE_M,
) -> list[dict[str, Any]]:
    """Top-down long edges from plane quads (horizontal world edges in XY).

    Top+bottom of a wall share one XY chord — keep a single deduped segment.
    """
    edges: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int]] = set()
    for pl in planes:
        quad = pl.get("quad") or pl.get("corners") or pl.get("vertices")
        if not isinstance(quad, list) or len(quad) < 2:
            continue
        pid = str(pl.get("id") or "")
        n = len(quad)
        for i in range(n):
            a, b = quad[i], quad[(i + 1) % n]
            if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)):
                continue
            if len(a) < 2 or len(b) < 2:
                continue
            ae, an = float(a[0]), float(a[1])
            be, bn = float(b[0]), float(b[1])
            az = float(a[2]) if len(a) > 2 else 0.0
            bz = float(b[2]) if len(b) > 2 else 0.0
            length_xy = math.hypot(be - ae, bn - an)
            if length_xy < 0.5 and abs(bz - az) > 0.5:
                continue
            if length_xy < float(min_len_m):
                continue
            mid_e, mid_n = 0.5 * (ae + be), 0.5 * (an + bn)
            te, tn = (be - ae) / length_xy, (bn - an) / length_xy
            if te < 0 or (abs(te) < 1e-9 and tn < 0):
                te, tn = -te, -tn
            key = (
                int(round(mid_e * 5)),
                int(round(mid_n * 5)),
                int(round(math.atan2(tn, te) * 20)),
            )
            if key in seen:
                continue
            seen.add(key)
            edges.append(
                {
                    "plane_id": pid,
                    "p0": (ae, an),
                    "p1": (be, bn),
                    "mid": (mid_e, mid_n),
                    "tangent": (te, tn),
                    "length_m": float(length_xy),
                }
            )
    return edges


def _metres_per_px(ortho: Any) -> float:
    m_e = abs(float(ortho.ee) - float(ortho.sw)) / max(int(ortho.w) - 1, 1)
    m_n = abs(float(ortho.nn) - float(ortho.sh)) / max(int(ortho.h) - 1, 1)
    return 0.5 * (m_e + m_n)


def sat_canny_edge_points_enu(
    ortho: Any,
    *,
    max_points: int = 8000,
) -> tuple[np.ndarray, float]:
    """Ortho Canny edge pixels → ENU XY samples + metres/pixel."""
    edges = ortho_canny(ortho.image)
    ys, xs = np.where(edges > 0)
    if len(ys) < 40:
        raise SatOffsetError("too few Ortho Canny edges")
    if len(ys) > max_points:
        idx = np.linspace(0, len(ys) - 1, max_points).astype(np.int64)
        ys, xs = ys[idx], xs[idx]
    pts = np.empty((len(ys), 2), dtype=np.float64)
    for i, (u, v) in enumerate(zip(xs.tolist(), ys.tolist())):
        e, n = px_to_enu(ortho, float(u), float(v))
        pts[i, 0] = e
        pts[i, 1] = n
    return pts, _metres_per_px(ortho)


def sat_canny_dt_metres(ortho: Any) -> tuple[np.ndarray, float]:
    """Distance transform of Ortho Canny in metres (pixel DT × m/px)."""
    edges = ortho_canny(ortho.image)
    if int((edges > 0).sum()) < 40:
        raise SatOffsetError("too few Ortho Canny edges for DT")
    inv = np.where(edges > 0, 0, 255).astype(np.uint8)
    dt_px = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    mpp = _metres_per_px(ortho)
    return dt_px * mpp, mpp


def match_facade_to_sat(
    edges: list[dict[str, Any]],
    sat_pts: np.ndarray,
    *,
    search_r_m: float = DEFAULT_SEARCH_R_M,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[float]]:
    """Nearest sat edge point within ``search_r_m`` for each façade mid (red→yellow)."""
    if sat_pts.size == 0 or not edges:
        return [], [], []
    before: list[dict[str, Any]] = []
    after: list[dict[str, Any]] = []
    dists: list[float] = []
    for ed in edges:
        me, mn = ed["mid"]
        d2 = (sat_pts[:, 0] - me) ** 2 + (sat_pts[:, 1] - mn) ** 2
        j = int(np.argmin(d2))
        dist = float(math.sqrt(float(d2[j])))
        if dist > float(search_r_m):
            continue
        before.append({"e": float(me), "n": float(mn)})
        after.append({"e": float(sat_pts[j, 0]), "n": float(sat_pts[j, 1])})
        dists.append(dist)
    return before, after, dists


def _residual_rms(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
    T: dict[str, float],
) -> float:
    if not before:
        return float("inf")
    errs = []
    for b, a in zip(before, after):
        e2, n2 = apply_se2_xy(float(b["e"]), float(b["n"]), T)
        errs.append(math.hypot(e2 - float(a["e"]), n2 - float(a["n"])))
    return float(math.sqrt(sum(x * x for x in errs) / len(errs)))


def facade_edge_samples(
    edges: list[dict[str, Any]],
    *,
    spacing_m: float = 1.0,
) -> np.ndarray:
    """Sample points along façade long edges for Chamfer cost."""
    pts: list[list[float]] = []
    step = max(float(spacing_m), 0.25)
    for ed in edges:
        e0, n0 = ed["p0"]
        e1, n1 = ed["p1"]
        length = float(ed["length_m"])
        n_steps = max(1, int(math.ceil(length / step)))
        for i in range(n_steps + 1):
            t = i / n_steps
            pts.append([e0 + t * (e1 - e0), n0 + t * (n1 - n0)])
    if not pts:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(pts, dtype=np.float64)


def chamfer_cost_m(
    ortho: Any,
    samples_xy: np.ndarray,
    T: dict[str, float],
    dt_m: np.ndarray | None = None,
) -> float:
    """Mean Ortho-Canny DT (m) at SE(2)-transformed façade samples."""
    if samples_xy.size == 0:
        return float("inf")
    if dt_m is None:
        dt_m, _ = sat_canny_dt_metres(ortho)
    h, w = dt_m.shape[:2]
    vals: list[float] = []
    for e, n in samples_xy:
        e2, n2 = apply_se2_xy(float(e), float(n), T)
        u, v = ortho.enu_to_px(e2, n2)
        ui, vi = int(round(u)), int(round(v))
        if 0 <= ui < w and 0 <= vi < h:
            vals.append(float(dt_m[vi, ui]))
    if len(vals) < 4:
        return float("inf")
    return float(sum(vals) / len(vals))


def multistart_facade_chamfer(
    ortho: Any,
    samples_xy: np.ndarray,
    *,
    init: dict[str, float] | None = None,
    de_span_m: float = DEFAULT_MS_DE_M,
    dyaw_span_deg: float = DEFAULT_MS_DYAW_DEG,
    step_m: float = DEFAULT_MS_STEP_M,
    step_yaw_deg: float = DEFAULT_MS_STEP_YAW,
) -> dict[str, float]:
    """Grid multi-start over SE(2) so auto-align can escape NCC local min.

    Cost = mean DT_sat(T · façade_edge_samples). Optional seed from
    correspondence measure (T_force).
    """
    dt_m, _ = sat_canny_dt_metres(ortho)
    if samples_xy.size == 0:
        raise SatOffsetError("no façade samples for multi-start Chamfer")

    if init is None:
        ce = float(samples_xy[:, 0].mean())
        cn = float(samples_xy[:, 1].mean())
        base = {
            "tx_m": 0.0,
            "ty_m": 0.0,
            "yaw_deg": 0.0,
            "s": 1.0,
            "pivot_e": ce,
            "pivot_n": cn,
        }
    else:
        base = {
            "tx_m": float(init.get("tx_m", 0.0)),
            "ty_m": float(init.get("ty_m", 0.0)),
            "yaw_deg": float(init.get("yaw_deg", 0.0)),
            "s": 1.0,
            "pivot_e": float(init.get("pivot_e", float(samples_xy[:, 0].mean()))),
            "pivot_n": float(init.get("pivot_n", float(samples_xy[:, 1].mean()))),
        }

    best = dict(base)
    best_cost = chamfer_cost_m(ortho, samples_xy, best, dt_m=dt_m)

    des = np.arange(-de_span_m, de_span_m + 1e-9, step_m)
    dns = np.arange(-de_span_m, de_span_m + 1e-9, step_m)
    dyaws = np.arange(-dyaw_span_deg, dyaw_span_deg + 1e-9, step_yaw_deg)
    for de in des:
        for dn in dns:
            for dy in dyaws:
                cand = dict(base)
                cand["tx_m"] = float(base["tx_m"] + de)
                cand["ty_m"] = float(base["ty_m"] + dn)
                cand["yaw_deg"] = float(base["yaw_deg"] + dy)
                cost = chamfer_cost_m(ortho, samples_xy, cand, dt_m=dt_m)
                if cost < best_cost:
                    best_cost = cost
                    best = cand

    for de in (-0.5 * step_m, 0.0, 0.5 * step_m):
        for dn in (-0.5 * step_m, 0.0, 0.5 * step_m):
            for dy in (-0.5 * step_yaw_deg, 0.0, 0.5 * step_yaw_deg):
                cand = dict(best)
                cand["tx_m"] = float(best["tx_m"] + de)
                cand["ty_m"] = float(best["ty_m"] + dn)
                cand["yaw_deg"] = float(best["yaw_deg"] + dy)
                cost = chamfer_cost_m(ortho, samples_xy, cand, dt_m=dt_m)
                if cost < best_cost:
                    best_cost = cost
                    best = cand

    best["chamfer_m"] = float(best_cost)
    best["s"] = 1.0
    best["source"] = "facade_sat_chamfer_multistart"
    return best


def measure_facade_sat_se2(
    project: Project,
    *,
    search_r_m: float = DEFAULT_SEARCH_R_M,
    min_len_m: float = DEFAULT_MIN_EDGE_M,
    max_rms_m: float = DEFAULT_MAX_RMS_M,
    min_pairs: int = DEFAULT_MIN_PAIRS,
    max_yaw_deg: float = DEFAULT_MAX_YAW_DEG,
    max_translation_m: float = DEFAULT_MAX_TRANSLATION_M,
    multistart: bool = False,
) -> dict[str, Any]:
    """Measure yellow ≈ R(yaw) @ red + (tx, ty) from façade edges vs Ortho Canny.

    Does not write files — caller persists ``T_force.json``. Raises when gates fail.
    """
    planes = _load_planes(project.recon_dir)
    edges = facade_long_edges(planes, min_len_m=min_len_m)
    if len(edges) < int(min_pairs):
        raise SatOffsetError(
            f"need ≥{min_pairs} long façade edges, got {len(edges)} (min_len={min_len_m}m)"
        )

    ortho = load_ortho_for_run(project.root)
    sat_pts, mpp = sat_canny_edge_points_enu(ortho)
    before, after, dists = match_facade_to_sat(edges, sat_pts, search_r_m=search_r_m)
    if len(before) < int(min_pairs):
        raise SatOffsetError(
            f"need ≥{min_pairs} façade↔sat pairs within {search_r_m}m, got {len(before)}"
        )

    T = fit_se2(before, after)
    rms = _residual_rms(before, after, T)
    tx = float(T.get("tx_m", 0.0))
    ty = float(T.get("ty_m", 0.0))
    yaw = float(T.get("yaw_deg", 0.0))
    trans = math.hypot(tx, ty)

    if rms > float(max_rms_m):
        raise SatOffsetError(f"rms {rms:.3f}m > gate {max_rms_m}m")
    if abs(yaw) > float(max_yaw_deg):
        raise SatOffsetError(f"|yaw| {abs(yaw):.2f}° > gate {max_yaw_deg}°")
    if trans > float(max_translation_m):
        raise SatOffsetError(f"||t|| {trans:.3f}m > gate {max_translation_m}m")

    if multistart:
        samples = facade_edge_samples(edges, spacing_m=1.0)
        T_ms = multistart_facade_chamfer(ortho, samples, init=T)
        T = {
            "tx_m": float(T_ms["tx_m"]),
            "ty_m": float(T_ms["ty_m"]),
            "yaw_deg": float(T_ms["yaw_deg"]),
            "s": 1.0,
            "pivot_e": float(T_ms["pivot_e"]),
            "pivot_n": float(T_ms["pivot_n"]),
        }
        rms = _residual_rms(before, after, T)

    return {
        "tx_m": float(T.get("tx_m", 0.0)),
        "ty_m": float(T.get("ty_m", 0.0)),
        "yaw_deg": float(T.get("yaw_deg", 0.0)),
        "s": 1.0,
        "pivot_e": float(T.get("pivot_e", 0.0)),
        "pivot_n": float(T.get("pivot_n", 0.0)),
        "source": SOURCE_FACADE_SAT_CHAMFER,
        "rms_m": float(rms),
        "n_pairs": int(len(before)),
        "n_edges": int(len(edges)),
        "search_r_m": float(search_r_m),
        "m_per_px": float(mpp),
        "mean_nn_m": float(sum(dists) / len(dists)) if dists else float("nan"),
    }


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def bak_force(project: Project, *, stamp: str | None = None) -> dict[str, str]:
    """Snapshot align/ + recon product files before forced apply."""
    ts = stamp or _stamp()
    align_bak = project.root / f"align.bak_force_{ts}"
    if project.align_dir.is_dir():
        if align_bak.exists():
            raise SatOffsetError(f"bak already exists: {align_bak}")
        shutil.copytree(project.align_dir, align_bak)

    recon_bak = project.recon_dir / f"bak_force_{ts}"
    recon_bak.mkdir(parents=True, exist_ok=False)
    copied: list[str] = []
    for name in (
        "cloud.ply",
        "cloud_photo.ply",
        "planes.json",
        "facades.obj",
        "facades.mtl",
        "scene.json",
    ):
        src = project.recon_dir / name
        if src.is_file():
            shutil.copy2(src, recon_bak / name)
            copied.append(name)
    return {
        "stamp": ts,
        "align_bak": str(align_bak) if align_bak.exists() else "",
        "recon_bak": str(recon_bak),
        "recon_files": ",".join(copied),
    }


def _parse_csv_set(raw: str | None, default: set[str]) -> set[str]:
    if raw is None or str(raw).strip() == "":
        return set(default)
    return {p.strip().lower() for p in str(raw).split(",") if p.strip()}


def apply_forced_se2(
    project: Project,
    T: dict[str, float],
    *,
    targets: str = "cams,cloud,facades,planes",
    skip: str = "roofs,street",
    bak: bool = True,
) -> dict[str, Any]:
    """Bak then apply one SE(2) to cams + selected recon artefacts.

    Default targets: cams, cloud, facades, planes. Default skip: roofs, street
    (sat-native). No free-pose.
    """
    target_set = _parse_csv_set(targets, {"cams", "cloud", "facades", "planes"})
    skip_set = _parse_csv_set(skip, {"roofs", "street"})
    apply_roofs = "roofs" in target_set and "roofs" not in skip_set
    apply_street = "street" in target_set and "street" not in skip_set

    meta: dict[str, Any] = {
        "targets": sorted(target_set),
        "skip": sorted(skip_set),
        "T": {
            "tx_m": float(T.get("tx_m", 0.0)),
            "ty_m": float(T.get("ty_m", 0.0)),
            "yaw_deg": float(T.get("yaw_deg", 0.0)),
            "s": float(T.get("s", 1.0)),
            "pivot_e": float(T["pivot_e"]) if "pivot_e" in T else None,
            "pivot_n": float(T["pivot_n"]) if "pivot_n" in T else None,
            "source": T.get("source", SOURCE_FACADE_SAT_CHAMFER),
        },
    }

    if bak:
        meta["bak"] = bak_force(project)

    stats: dict[str, Any] = {}
    poses: list[dict[str, Any]] = []
    poses_path = project.align_dir / "poses.json"

    if "cams" in target_set and "cams" not in skip_set:
        if not poses_path.is_file():
            raise SatOffsetError(f"missing {poses_path}")
        raw = json.loads(poses_path.read_text(encoding="utf-8"))
        poses = raw if isinstance(raw, list) else list(raw.get("poses") or raw.get("cameras") or [])
        poses = [apply_se2_pose(p, T) for p in poses]
        poses_path.write_text(json.dumps(poses, indent=2), encoding="utf-8")
        stats["poses"] = len(poses)
        cams_path = project.align_dir / "cameras.json"
        if cams_path.is_file():
            cams_raw = json.loads(cams_path.read_text(encoding="utf-8"))
            if isinstance(cams_raw, list):
                cams_out = [apply_se2_pose(c, T) for c in cams_raw]
                cams_path.write_text(json.dumps(cams_out, indent=2), encoding="utf-8")
                stats["cameras"] = len(cams_out)

    if "cloud" in target_set and "cloud" not in skip_set:
        for name in ("cloud.ply", "cloud_photo.ply"):
            p = project.recon_dir / name
            if p.is_file():
                try:
                    stats[name] = apply_se2_to_ply(p, T)
                except Exception as exc:  # noqa: BLE001
                    log.warning("sat-offset: skip %s (%s)", p, exc)
                    stats[name] = -1

    if "facades" in target_set and "facades" not in skip_set:
        obj = project.recon_dir / "facades.obj"
        if obj.is_file():
            stats["facades.obj"] = apply_se2_to_ascii_obj(obj, T)

    if "planes" in target_set and "planes" not in skip_set:
        planes_path = project.recon_dir / "planes.json"
        if planes_path.is_file():
            stats["planes.json"] = apply_se2_to_planes_json(planes_path, T)

    if apply_roofs:
        roofs = project.recon_dir / "roofs.obj"
        if roofs.is_file():
            stats["roofs.obj"] = apply_se2_to_ascii_obj(roofs, T)
    else:
        stats["roofs.obj"] = "skipped"

    if apply_street:
        street = project.recon_dir / "street.obj"
        if street.is_file():
            stats["street.obj"] = apply_se2_to_ascii_obj(street, T)
    else:
        stats["street.obj"] = "skipped"

    georef_path = project.align_dir / "georef.json"
    georef: dict[str, Any] = {}
    if georef_path.is_file():
        loaded = json.loads(georef_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            georef = loaded
    t_body = {
        "tx_m": float(T.get("tx_m", 0.0)),
        "ty_m": float(T.get("ty_m", 0.0)),
        "yaw_deg": float(T.get("yaw_deg", 0.0)),
        "s": float(T.get("s", 1.0)),
    }
    if "pivot_e" in T:
        t_body["pivot_e"] = float(T["pivot_e"])
    if "pivot_n" in T:
        t_body["pivot_n"] = float(T["pivot_n"])
    georef["T_force"] = dict(t_body)
    georef["T_force"]["source"] = T.get("source", SOURCE_FACADE_SAT_CHAMFER)
    georef["T_applied"] = dict(t_body)
    georef["prior"] = georef.get("prior") or "sat"
    georef["forced_se2"] = True
    project.write_json(georef_path, georef)

    if not poses and poses_path.is_file():
        raw = json.loads(poses_path.read_text(encoding="utf-8"))
        poses = raw if isinstance(raw, list) else list(raw.get("poses") or [])
    scene_path = project.recon_dir / "scene.json"
    if poses and scene_path.is_file():
        refresh_scene_cameras(scene_path, poses, georef)
        stats["scene"] = "refreshed"

    meta["stats"] = stats
    meta["georef"] = str(georef_path)
    return meta


def write_t_force(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_t_force(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise SatOffsetError(f"missing T_force: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SatOffsetError(f"invalid T_force json: {path}")
    return data
