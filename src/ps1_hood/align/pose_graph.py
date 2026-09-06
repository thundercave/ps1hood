"""Build initial poses from GPS/OSM, then refine with views + satellite."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.optimize import least_squares

from ps1_hood.align.features import match_pair
from ps1_hood.align.satellite_align import align_camera_to_satellite
from ps1_hood.capture.satellite import Ortho
from ps1_hood.config import ProjectSpec
from ps1_hood.geo import (
    LocalFrame,
    camera_rotation_cv,
    heading_diff,
    snap_to_polylines_enu,
    wrap_heading,
)
from ps1_hood.overpass import roads_to_enu_lines

log = logging.getLogger(__name__)


def _front_shots(shots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One frame per pano: horizon pitch, looking along the road."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for shot in shots:
        groups.setdefault(shot["pano_id"], []).append(shot)
    fronts: list[dict[str, Any]] = []
    for group in groups.values():

        def score(s: dict[str, Any]) -> tuple[float, float]:
            travel = float(s.get("travel_heading") if s.get("travel_heading") is not None else s.get("heading") or 0.0)
            return (
                abs(float(s.get("pitch") or 0.0)),
                abs(heading_diff(float(s.get("heading") or 0.0), travel)),
            )

        fronts.append(min(group, key=score))
    return fronts


def initial_poses(
    shots: list[dict[str, Any]],
    spec: ProjectSpec,
    frame: LocalFrame,
    osm: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    lines = roads_to_enu_lines(osm, frame) if osm else []
    poses: list[dict[str, Any]] = []
    for shot in _front_shots(shots):
        e, n, _ = frame.to_enu(float(shot["lat"]), float(shot["lon"]))
        e_gps, n_gps = e, n
        heading = float(shot.get("heading") or 0.0)
        snapped = False
        if lines:
            se, sn, sh, dist = snap_to_polylines_enu(e, n, lines)
            if dist < 12.0:
                e, n = se, sn
                # keep the photo heading if it is already close; otherwise use the road
                if abs(heading_diff(heading, sh)) < 40 or abs(heading_diff(heading, sh + 180)) < 40:
                    if abs(heading_diff(heading, sh + 180)) < abs(heading_diff(heading, sh)):
                        heading = wrap_heading(sh + 180.0)
                    else:
                        heading = sh
                snapped = True
        poses.append(
            {
                "pano_id": shot["pano_id"],
                "shot_path": shot["path"],
                "lat_raw": shot["lat"],
                "lon_raw": shot["lon"],
                "e_gps": e_gps,
                "n_gps": n_gps,
                "e": e,
                "n": n,
                "u": spec.camera_height_m,
                "heading": heading,
                "pitch": float(shot.get("pitch") or 0.0),
                "fov": float(shot.get("fov") or spec.fov_deg),
                "width": shot.get("width"),
                "height": shot.get("height"),
                "snapped": snapped,
                "source": shot.get("source"),
                "mask": shot.get("mask"),
                "travel_heading": shot.get("travel_heading", heading),
            }
        )
    return poses


def explode_orbit_cameras(
    pano_poses: list[dict[str, Any]],
    shots: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Copy the refined XYZ onto every heading/pitch grabbed from that pano."""
    by_id = {p["pano_id"]: p for p in pano_poses}
    cameras: list[dict[str, Any]] = []
    for shot in shots:
        base = by_id.get(shot["pano_id"])
        if not base:
            continue
        cam = dict(base)
        cam["shot_path"] = shot["path"]
        cam["heading"] = float(shot["heading"])
        cam["pitch"] = float(shot.get("pitch") or 0.0)
        cam["fov"] = float(shot.get("fov") or base["fov"])
        cam["width"] = shot.get("width") or base.get("width")
        cam["height"] = shot.get("height") or base.get("height")
        cam["mask"] = shot.get("mask")
        cam["rel_heading"] = shot.get("rel_heading")
        cam["R"] = camera_rotation_cv(cam["heading"], cam["pitch"])
        cameras.append(cam)
    return cameras


def _neighbor_pairs(poses: list[dict[str, Any]], max_dist_m: float = 22.0) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for i, a in enumerate(poses):
        best: tuple[float, int] | None = None
        hx = np.sin(np.deg2rad(a["heading"]))
        hy = np.cos(np.deg2rad(a["heading"]))
        for j, b in enumerate(poses):
            if i == j:
                continue
            de, dn = b["e"] - a["e"], b["n"] - a["n"]
            dist = float(np.hypot(de, dn))
            if dist < 1.5 or dist > max_dist_m:
                continue
            ahead = de * hx + dn * hy
            if ahead <= 0:
                continue
            if best is None or dist < best[0]:
                best = (dist, j)
        if best:
            pairs.append((i, best[1]))
    return pairs


def refine_poses(
    poses: list[dict[str, Any]],
    spec: ProjectSpec,
    frame: LocalFrame,
    ortho: Ortho | None,
    *,
    use_satellite: bool = True,
    use_features: bool = True,
) -> list[dict[str, Any]]:
    refined = [dict(p) for p in poses]
    sat_obs: list[dict[str, Any]] = []
    feat_obs: list[dict[str, Any]] = []

    if use_satellite and ortho is not None:
        for i, pose in enumerate(refined):
            photo = cv2.imread(pose["shot_path"], cv2.IMREAD_COLOR)
            if photo is None:
                continue
            log.info("satellite align %s/%s  %s", i + 1, len(refined), pose["pano_id"])
            hit = align_camera_to_satellite(
                photo,
                ortho,
                e=pose["e"],
                n=pose["n"],
                heading=pose["heading"],
                pitch=pose["pitch"],
                height_m=spec.camera_height_m,
                fov_deg=pose["fov"],
                max_shift_m=8.0,
                max_heading_deg=15.0,
            )
            pose["sat_score"] = hit["score"]
            if hit["score"] > 0.08:
                pose["e"] = hit["e"]
                pose["n"] = hit["n"]
                pose["heading"] = hit["heading"]
                sat_obs.append({"i": i, **hit})

    if use_features:
        for i, j in _neighbor_pairs(refined):
            rel = match_pair(refined[i]["shot_path"], refined[j]["shot_path"])
            if not rel:
                continue
            feat_obs.append({"i": i, "j": j, **rel})

    if len(refined) >= 2:
        refined = _bundle_se2(refined, feat_obs)

    for pose in refined:
        lat, lon, _ = frame.to_geodetic(pose["e"], pose["n"], 0.0)
        pose["lat"] = lat
        pose["lon"] = lon
        pose["R"] = camera_rotation_cv(pose["heading"], pose["pitch"])
    return refined


def _bundle_se2(
    poses: list[dict[str, Any]],
    feat_obs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Soft GPS prior + relative heading/translation from matched neighbours."""
    n = len(poses)
    x0 = np.zeros(3 * n, dtype=np.float64)
    for i, p in enumerate(poses):
        x0[3 * i] = p["e"]
        x0[3 * i + 1] = p["n"]
        x0[3 * i + 2] = np.deg2rad(p["heading"])

    prior_e = x0[0::3].copy()
    prior_n = x0[1::3].copy()
    prior_h = x0[2::3].copy()

    def residual(x: np.ndarray) -> np.ndarray:
        rs = []
        e = x[0::3]
        n_ = x[1::3]
        h = x[2::3]
        for i in range(n):
            rs.append(0.15 * (e[i] - prior_e[i]))
            rs.append(0.15 * (n_[i] - prior_n[i]))
            rs.append(0.4 * _wrap_rad(h[i] - prior_h[i]))
        for obs in feat_obs:
            i, j = obs["i"], obs["j"]
            de, dn = e[j] - e[i], n_[j] - n_[i]
            dist = float(np.hypot(de, dn)) + 1e-6
            # predicted heading of translation in world
            trans_h = np.arctan2(de, dn)
            rs.append(0.8 * _wrap_rad(trans_h - h[i]))
            # neighbour should look roughly along the same street
            rs.append(0.3 * _wrap_rad(h[j] - h[i]))
            rs.append(0.05 * (dist - 8.0))  # typical SV spacing
        return np.array(rs, dtype=np.float64)

    try:
        sol = least_squares(residual, x0, method="trf", max_nfev=80)
        x = sol.x
    except Exception as exc:  # noqa: BLE001
        log.warning("pose graph failed (%s); keeping satellite/GPS poses", exc)
        return poses

    out = [dict(p) for p in poses]
    for i, p in enumerate(out):
        p["e"] = float(x[3 * i])
        p["n"] = float(x[3 * i + 1])
        p["heading"] = wrap_heading(float(np.rad2deg(x[3 * i + 2])))
    return out


def _wrap_rad(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def write_debug_overlay(
    poses: list[dict[str, Any]],
    ortho: Ortho,
    dest: Path,
) -> None:
    canvas = ortho.image.copy()
    for pose in poses:
        u, v = ortho.enu_to_px(pose["e"], pose["n"])
        pt = (int(round(u)), int(round(v)))
        h = np.deg2rad(pose["heading"])
        tip = (
            int(round(u + 18 * np.sin(h))),
            int(round(v - 18 * np.cos(h))),
        )
        cv2.circle(canvas, pt, 4, (0, 220, 255), -1)
        cv2.line(canvas, pt, tip, (0, 220, 255), 2)
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), canvas)
