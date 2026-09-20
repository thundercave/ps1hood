"""Forced SE(2) sat-offset: façade Chamfer, Studio picks, or cam→street centerline.

Sacred: one rigid SE(2) · no free-pose · bak first · prefer sat XY.
Reuse ``fit_se2`` / ``apply_se2_*`` / ``seat_recon_artefacts`` patterns.
Skip roofs/street when already sat-native.
Sources:
  - façade↔Ortho Canny Chamfer → ``T_force.json`` (do **not** auto-apply)
  - Studio yellow↔red corner picks → ``fit_pairs_se2`` (preview only; apply on confirm)
  - unique cam XY → Ortho street_mask medial/centerline NN → ``T_cam_road.json``
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
from ps1_hood.reconstruct.sat_roofs import load_cam_xy_enu, load_ortho_for_run, px_to_enu, segment_roof_yard_mask

log = logging.getLogger(__name__)

DEFAULT_SEARCH_R_M = 8.0
DEFAULT_MIN_EDGE_M = 3.0
DEFAULT_MAX_RMS_M = 1.5
DEFAULT_MIN_PAIRS = 4
DEFAULT_MAX_YAW_DEG = 15.0
DEFAULT_MAX_TRANSLATION_M = 10.0
SOURCE_FACADE_SAT_CHAMFER = "facade_sat_chamfer"
SOURCE_STUDIO_PICKS = "studio_corner_picks"
SOURCE_CAM_STREET_CENTERLINE = "cam_street_centerline"
DEFAULT_MIN_PAIRS_PICK = 3

# Cam → street centerline measure gates (pack: n≥4, rms≤2m, |yaw|≤10°, ||t||≤12m)
DEFAULT_CAM_SEARCH_R_M = 15.0
DEFAULT_CAM_MIN_NN_M = 0.5
DEFAULT_CAM_MAX_RMS_M = 2.0
DEFAULT_CAM_MAX_YAW_DEG = 10.0
DEFAULT_CAM_MAX_TRANSLATION_M = 12.0
DEFAULT_CAM_YAW_ZERO_DEG = 2.0
DEFAULT_CENTERLINE_MAX_POINTS = 12000

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




def normalize_corner_pairs(
    pairs: list[Any],
) -> tuple[list[dict[str, float]], list[dict[str, float]], list[dict[str, Any]]]:
    """Parse Studio/CLI pairs → (red_before, yellow_after, audit rows).

    Each pair: ``{yellow:{e,n}, red:{e,n}}`` (aliases: sat/product, after/before).
    """
    before: list[dict[str, float]] = []
    after: list[dict[str, float]] = []
    audit: list[dict[str, Any]] = []
    for i, raw in enumerate(pairs or []):
        if not isinstance(raw, dict):
            raise SatOffsetError(f"pair[{i}] must be an object")
        yel = raw.get("yellow") or raw.get("sat") or raw.get("after")
        red = raw.get("red") or raw.get("product") or raw.get("before")
        if not isinstance(yel, dict) or not isinstance(red, dict):
            raise SatOffsetError(f"pair[{i}] needs yellow{{e,n}} and red{{e,n}}")
        try:
            ye, yn = float(yel["e"]), float(yel["n"])
            re, rn = float(red["e"]), float(red["n"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SatOffsetError(f"pair[{i}] bad e/n: {exc}") from exc
        before.append({"e": re, "n": rn})
        after.append({"e": ye, "n": yn})
        audit.append(
            {
                "i": i,
                "red": {"e": re, "n": rn},
                "yellow": {"e": ye, "n": yn},
                "delta_e": ye - re,
                "delta_n": yn - rn,
                "dist_m": float(math.hypot(ye - re, yn - rn)),
            }
        )
    return before, after, audit


def fit_pairs_se2(
    pairs: list[Any],
    *,
    max_rms_m: float = DEFAULT_MAX_RMS_M,
    min_pairs: int = DEFAULT_MIN_PAIRS_PICK,
    max_yaw_deg: float = DEFAULT_MAX_YAW_DEG,
    max_translation_m: float = DEFAULT_MAX_TRANSLATION_M,
) -> dict[str, Any]:
    """Fit one rigid SE(2) from ≥3 yellow↔red corner pairs (red → yellow).

    Does **not** apply. Preview payload includes transformed red points and
    pair segments for Studio overlay. Gates: n≥min_pairs, rms, |yaw|, ||t||.
    """
    before, after, audit = normalize_corner_pairs(pairs)
    n = len(before)
    if n < int(min_pairs):
        raise SatOffsetError(f"need ≥{min_pairs} yellow↔red pairs, got {n}")

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

    preview_red: list[dict[str, float]] = []
    preview_arrows: list[dict[str, Any]] = []
    for b, a, row in zip(before, after, audit):
        e2, n2 = apply_se2_xy(float(b["e"]), float(b["n"]), T)
        preview_red.append({"e": float(e2), "n": float(n2)})
        preview_arrows.append(
            {
                "from": {"e": float(b["e"]), "n": float(b["n"])},
                "to": {"e": float(a["e"]), "n": float(a["n"])},
                "mapped": {"e": float(e2), "n": float(n2)},
                "residual_m": float(math.hypot(e2 - float(a["e"]), n2 - float(a["n"]))),
                "i": row["i"],
            }
        )

    return {
        "tx_m": tx,
        "ty_m": ty,
        "yaw_deg": yaw,
        "s": 1.0,
        "pivot_e": float(T.get("pivot_e", 0.0)),
        "pivot_n": float(T.get("pivot_n", 0.0)),
        "source": SOURCE_STUDIO_PICKS,
        "rms_m": float(rms),
        "n_pairs": int(n),
        "pairs": audit,
        "preview": {
            "red_mapped": preview_red,
            "arrows": preview_arrows,
        },
        "applied": False,
    }


def persist_t_pick(
    project: Project,
    payload: dict[str, Any],
    *,
    t_path: Path | None = None,
    pairs_path: Path | None = None,
) -> dict[str, str]:
    """Write ``align/T_pick.json`` + ``align/T_pick_pairs.json`` (audit). No apply."""
    dest = Path(t_path) if t_path else (project.align_dir / "T_pick.json")
    if not dest.is_absolute():
        dest = project.root / dest
    pairs_dest = Path(pairs_path) if pairs_path else (project.align_dir / "T_pick_pairs.json")
    if not pairs_dest.is_absolute():
        pairs_dest = project.root / pairs_dest

    t_body = {
        "tx_m": float(payload["tx_m"]),
        "ty_m": float(payload["ty_m"]),
        "yaw_deg": float(payload["yaw_deg"]),
        "s": float(payload.get("s", 1.0)),
        "pivot_e": float(payload.get("pivot_e", 0.0)),
        "pivot_n": float(payload.get("pivot_n", 0.0)),
        "source": payload.get("source", SOURCE_STUDIO_PICKS),
        "rms_m": float(payload.get("rms_m", 0.0)),
        "n_pairs": int(payload.get("n_pairs", 0)),
    }
    write_t_force(dest, t_body)

    audit = {
        "source": SOURCE_STUDIO_PICKS,
        "T": t_body,
        "pairs": payload.get("pairs") or [],
        "preview": payload.get("preview") or {},
        "applied": False,
        "note": "preview only — apply via sat-offset apply --from align/T_pick.json",
    }
    pairs_dest.parent.mkdir(parents=True, exist_ok=True)
    pairs_dest.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return {"T_pick": str(dest), "T_pick_pairs": str(pairs_dest)}


def load_pairs_json(path: Path) -> list[Any]:
    path = Path(path)
    if not path.is_file():
        raise SatOffsetError(f"missing pairs json: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        pairs = data.get("pairs")
    else:
        pairs = data
    if not isinstance(pairs, list):
        raise SatOffsetError(f"pairs json must be a list or {{pairs:[...]}}: {path}")
    return pairs

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



def morph_skeleton(mask_u8: np.ndarray) -> np.ndarray:
    """Binary morphological skeleton (OpenCV; no ximgproc required)."""
    img = (mask_u8 > 0).astype(np.uint8) * 255
    if int(cv2.countNonZero(img)) == 0:
        return img
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(img, opened)
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded
        if cv2.countNonZero(img) == 0:
            break
    return skel


def dt_ridge_skeleton(mask_u8: np.ndarray) -> np.ndarray:
    """Distance-transform ridge: keep street pixels that are local DT maxima.

    Fallback when morphological skeleton is too thin/empty. For each street
    pixel, keep if DT ≥ neighbours along approximate gradient (3×3 local max
    with DT > 0.5 px so edges are dropped).
    """
    bin_m = (mask_u8 > 0).astype(np.uint8)
    if int(bin_m.sum()) < 8:
        return bin_m.astype(np.uint8) * 255
    dt = cv2.distanceTransform(bin_m, cv2.DIST_L2, 5)
    # Local max in 3×3
    k = np.ones((3, 3), dtype=np.uint8)
    local_max = cv2.dilate(dt, k)
    ridge = (dt >= local_max - 1e-6) & (dt > 0.75) & (bin_m > 0)
    # Thin slightly: also require DT greater than mean of 4-neighbours
    out = ridge.astype(np.uint8) * 255
    if int((out > 0).sum()) < 8:
        # Relax: top-percentile DT within street
        vals = dt[bin_m > 0]
        thr = float(np.percentile(vals, 85)) if vals.size else 0.5
        out = ((dt >= thr) & (bin_m > 0)).astype(np.uint8) * 255
    return out


def street_mask_centerline(
    street_m: np.ndarray,
    *,
    max_points: int = DEFAULT_CENTERLINE_MAX_POINTS,
) -> np.ndarray:
    """Street mask → skeleton pixel coordinates as (N,2) int (u=x, v=y).

    Prefers ``cv2.ximgproc.thinning`` when available; else morphological
    skeleton, with DT-ridge fallback if too sparse.
    """
    mask = (street_m > 0).astype(np.uint8) * 255
    if int(cv2.countNonZero(mask)) < 16:
        raise SatOffsetError("street_mask too empty for centerline")

    skel: np.ndarray | None = None
    ximgproc = getattr(cv2, "ximgproc", None)
    if ximgproc is not None and hasattr(ximgproc, "thinning"):
        try:
            skel = ximgproc.thinning(mask)
        except Exception:  # noqa: BLE001
            skel = None
    if skel is None or int(cv2.countNonZero(skel)) < 8:
        skel = morph_skeleton(mask)
    if int(cv2.countNonZero(skel)) < 8:
        skel = dt_ridge_skeleton(mask)
    if int(cv2.countNonZero(skel)) < 8:
        raise SatOffsetError("could not extract street centerline skeleton")

    ys, xs = np.where(skel > 0)
    if len(ys) > int(max_points):
        idx = np.linspace(0, len(ys) - 1, int(max_points)).astype(np.int64)
        ys, xs = ys[idx], xs[idx]
    return np.column_stack([xs, ys]).astype(np.int32)


def street_centerline_enu(
    ortho: Any,
    street_m: np.ndarray | None = None,
    *,
    max_points: int = DEFAULT_CENTERLINE_MAX_POINTS,
) -> tuple[np.ndarray, float]:
    """Ortho street_mask medial → dense ENU XY samples + metres/pixel."""
    if street_m is None:
        _, _, street_m = segment_roof_yard_mask(ortho.image)
    pix = street_mask_centerline(street_m, max_points=max_points)
    pts = np.empty((len(pix), 2), dtype=np.float64)
    for i, (u, v) in enumerate(pix):
        e, n = px_to_enu(ortho, float(u), float(v))
        pts[i, 0] = e
        pts[i, 1] = n
    return pts, _metres_per_px(ortho)


def unique_cam_xy_from_project(project: Project) -> np.ndarray:
    """Unique pano XY from align/poses.json (or cameras.json)."""
    xy, _ = load_cam_xy_enu(project.root)
    if xy is None or len(xy) == 0:
        raise SatOffsetError("no camera XY in align/poses.json or cameras.json")
    return np.asarray(xy, dtype=np.float64)


def match_cams_to_centerline(
    cam_xy: np.ndarray,
    centerline_xy: np.ndarray,
    *,
    search_r_m: float = DEFAULT_CAM_SEARCH_R_M,
    min_nn_m: float = DEFAULT_CAM_MIN_NN_M,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[float]]:
    """Each cam → nearest centerline point (optional dist band).

    Keep pairs with ``min_nn_m < dist ≤ search_r_m``. If that yields nothing,
    fall back to all pairs with ``dist ≤ search_r_m`` (already-on-road cams).
    """
    if cam_xy.size == 0 or centerline_xy.size == 0:
        return [], [], []
    before: list[dict[str, Any]] = []
    after: list[dict[str, Any]] = []
    dists: list[float] = []
    fallback_b: list[dict[str, Any]] = []
    fallback_a: list[dict[str, Any]] = []
    fallback_d: list[float] = []
    for e, n in cam_xy:
        d2 = (centerline_xy[:, 0] - e) ** 2 + (centerline_xy[:, 1] - n) ** 2
        j = int(np.argmin(d2))
        dist = float(math.sqrt(float(d2[j])))
        if dist > float(search_r_m):
            continue
        b = {"e": float(e), "n": float(n)}
        a = {"e": float(centerline_xy[j, 0]), "n": float(centerline_xy[j, 1])}
        fallback_b.append(b)
        fallback_a.append(a)
        fallback_d.append(dist)
        if dist > float(min_nn_m):
            before.append(b)
            after.append(a)
            dists.append(dist)
    if len(before) >= 2:
        return before, after, dists
    return fallback_b, fallback_a, fallback_d


def _fit_translation_only(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
) -> dict[str, float]:
    """Pure translation SE(2) (yaw=0) from mean displacement."""
    be = float(sum(p["e"] for p in before) / len(before))
    bn = float(sum(p["n"] for p in before) / len(before))
    ae = float(sum(p["e"] for p in after) / len(after))
    an = float(sum(p["n"] for p in after) / len(after))
    return {
        "tx_m": ae - be,
        "ty_m": an - bn,
        "yaw_deg": 0.0,
        "s": 1.0,
        "pivot_e": be,
        "pivot_n": bn,
    }


def measure_cam_road_se2(
    project: Project,
    *,
    search_r_m: float = DEFAULT_CAM_SEARCH_R_M,
    min_nn_m: float = DEFAULT_CAM_MIN_NN_M,
    max_rms_m: float = DEFAULT_CAM_MAX_RMS_M,
    min_pairs: int = DEFAULT_MIN_PAIRS,
    max_yaw_deg: float = DEFAULT_CAM_MAX_YAW_DEG,
    max_translation_m: float = DEFAULT_CAM_MAX_TRANSLATION_M,
    yaw_zero_deg: float = DEFAULT_CAM_YAW_ZERO_DEG,
    translation_only: bool | None = None,
) -> dict[str, Any]:
    """Measure SE(2): unique cam XY → Ortho street_mask centerline NN.

    Does not apply. Caller persists ``align/T_cam_road.json``. Gates:
    n_pairs ≥ min_pairs, rms ≤ max_rms_m, |yaw| ≤ max_yaw_deg, ||t|| ≤ max_translation_m.
    Prefer translation-dominant: if |yaw| ≤ yaw_zero_deg (or ``translation_only``),
    fit tx,ty with yaw=0 for stability.
    """
    cam_xy = unique_cam_xy_from_project(project)
    if len(cam_xy) < int(min_pairs):
        raise SatOffsetError(
            f"need ≥{min_pairs} unique cams, got {len(cam_xy)}"
        )

    ortho = load_ortho_for_run(project.root)
    _, _, street_m = segment_roof_yard_mask(ortho.image)
    centerline_xy, mpp = street_centerline_enu(ortho, street_m)
    before, after, dists = match_cams_to_centerline(
        cam_xy,
        centerline_xy,
        search_r_m=search_r_m,
        min_nn_m=min_nn_m,
    )
    if len(before) < int(min_pairs):
        raise SatOffsetError(
            f"need ≥{min_pairs} cam↔centerline pairs within {search_r_m}m, got {len(before)}"
        )

    T = fit_se2(before, after)
    yaw = float(T.get("yaw_deg", 0.0))
    use_t_only = bool(translation_only) if translation_only is not None else (
        abs(yaw) <= float(yaw_zero_deg)
    )
    if use_t_only:
        T = _fit_translation_only(before, after)
        yaw = 0.0

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

    preview_mapped: list[dict[str, float]] = []
    for b in before:
        e2, n2 = apply_se2_xy(float(b["e"]), float(b["n"]), T)
        preview_mapped.append({"e": float(e2), "n": float(n2)})

    return {
        "tx_m": tx,
        "ty_m": ty,
        "yaw_deg": yaw,
        "s": 1.0,
        "pivot_e": float(T.get("pivot_e", 0.0)),
        "pivot_n": float(T.get("pivot_n", 0.0)),
        "source": SOURCE_CAM_STREET_CENTERLINE,
        "rms_m": float(rms),
        "n_pairs": int(len(before)),
        "n_cams": int(len(cam_xy)),
        "n_centerline": int(len(centerline_xy)),
        "search_r_m": float(search_r_m),
        "min_nn_m": float(min_nn_m),
        "m_per_px": float(mpp),
        "mean_nn_m": float(sum(dists) / len(dists)) if dists else float("nan"),
        "translation_only": bool(use_t_only),
        "t_norm_m": float(trans),
        "preview": {
            "cams_before": [{"e": float(b["e"]), "n": float(b["n"])} for b in before],
            "cams_mapped": preview_mapped,
            "centerline_targets": [{"e": float(a["e"]), "n": float(a["n"])} for a in after],
        },
        "applied": False,
    }


def write_cam_road_overlay(
    project: Project,
    payload: dict[str, Any],
    *,
    out_path: Path | None = None,
) -> Path | None:
    """Optional top-down PNG: red cams, yellow centerline targets, cyan mapped."""
    preview = payload.get("preview") or {}
    before = preview.get("cams_before") or []
    mapped = preview.get("cams_mapped") or []
    targets = preview.get("centerline_targets") or []
    if not before:
        return None
    try:
        ortho = load_ortho_for_run(project.root)
    except Exception as exc:  # noqa: BLE001
        log.warning("cam-road overlay: no ortho (%s)", exc)
        return None
    img = ortho.image.copy()
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    def _px(e: float, n: float) -> tuple[int, int]:
        u, v = ortho.enu_to_px(e, n)
        return int(round(u)), int(round(v))

    for t in targets:
        u, v = _px(float(t["e"]), float(t["n"]))
        cv2.circle(img, (u, v), 2, (0, 255, 255), -1)  # yellow-ish
    for b in before:
        u, v = _px(float(b["e"]), float(b["n"]))
        cv2.circle(img, (u, v), 4, (0, 0, 255), -1)  # red
    for m in mapped:
        u, v = _px(float(m["e"]), float(m["n"]))
        cv2.circle(img, (u, v), 3, (255, 255, 0), -1)  # cyan
    for b, m in zip(before, mapped):
        cv2.line(img, _px(float(b["e"]), float(b["n"])), _px(float(m["e"]), float(m["n"])), (255, 128, 0), 1)

    dest = Path(out_path) if out_path else (project.align_dir / "T_cam_road_overlay.png")
    if not dest.is_absolute():
        dest = project.root / dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), img)
    return dest


def persist_t_cam_road(
    project: Project,
    payload: dict[str, Any],
    *,
    t_path: Path | None = None,
    overlay: bool = False,
    overlay_path: Path | None = None,
) -> dict[str, str]:
    """Write ``align/T_cam_road.json`` (measure only — no apply)."""
    dest = Path(t_path) if t_path else (project.align_dir / "T_cam_road.json")
    if not dest.is_absolute():
        dest = project.root / dest
    t_body = {
        "tx_m": float(payload["tx_m"]),
        "ty_m": float(payload["ty_m"]),
        "yaw_deg": float(payload["yaw_deg"]),
        "s": float(payload.get("s", 1.0)),
        "pivot_e": float(payload.get("pivot_e", 0.0)),
        "pivot_n": float(payload.get("pivot_n", 0.0)),
        "source": payload.get("source", SOURCE_CAM_STREET_CENTERLINE),
        "rms_m": float(payload.get("rms_m", 0.0)),
        "n_pairs": int(payload.get("n_pairs", 0)),
        "n_cams": int(payload.get("n_cams", 0)),
        "mean_nn_m": float(payload.get("mean_nn_m", float("nan"))),
        "t_norm_m": float(payload.get("t_norm_m", math.hypot(float(payload["tx_m"]), float(payload["ty_m"])))),
        "translation_only": bool(payload.get("translation_only", False)),
        "applied": False,
        "note": "preview only — apply via sat-offset apply --from align/T_cam_road.json",
    }
    write_t_force(dest, t_body)
    out: dict[str, str] = {"T_cam_road": str(dest)}
    if overlay:
        ov = write_cam_road_overlay(project, payload, out_path=overlay_path)
        if ov is not None:
            out["overlay"] = str(ov)
    return out
