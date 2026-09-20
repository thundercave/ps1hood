"""PR-C — Façade gap fill helpers (ZNCC-gated side/corner seeds).

Keep hybrid 10/8 A core; seed side walls / corner wraps from manhattan ±90°
and sat roof AABB corners as *hypotheses only*; reject road-center peels;
score via existing Path α ZNCC. No BAG hero, no peel-as-hero spam.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from ps1_hood.geo import wrap_heading
from ps1_hood.reconstruct.photo_planes import vertical_plane_from_point_heading

log = logging.getLogger(__name__)

# Recipe knobs
GAP_FILL_DISTANCES_M = (6.0, 10.0, 14.0, 18.0)
GAP_FILL_PEEL_CAP = 32
GAP_FILL_MAX_KEEP = 18
GAP_FILL_MIN_FRONTAL = 0.20  # far-side only (default Path α = 0.25)
GAP_FILL_CAM_MARGIN_M = 3.0
GAP_FILL_CORNER_INSET_M = 1.0
GAP_FILL_WIDTH_M = 8.0
GAP_FILL_HEIGHT_M = 9.0


def estimate_travel_heading_deg(frames: list[dict[str, Any]]) -> float | None:
    """Median travel_heading, or heading of drive from first→last unique pano XY."""
    travels = [
        float(fr["travel_heading"])
        for fr in frames
        if fr.get("travel_heading") is not None
    ]
    if travels:
        return float(np.median(travels))
    # Fallback: vector along unique camera XY sequence
    seen: set[tuple[float, float]] = set()
    pts: list[np.ndarray] = []
    for fr in frames:
        key = (round(float(fr["e"]), 2), round(float(fr["n"]), 2))
        if key in seen:
            continue
        seen.add(key)
        pts.append(np.array([float(fr["e"]), float(fr["n"])], dtype=np.float64))
    if len(pts) < 2:
        return None
    delta = pts[-1] - pts[0]
    if float(np.linalg.norm(delta)) < 1.0:
        return None
    # ENU: heading 0=+N, 90=+E → atan2(e, n)
    return float(math.degrees(math.atan2(float(delta[0]), float(delta[1]))) % 360.0)


def manhattan_side_seeds(
    frames: list[dict[str, Any]],
    *,
    ground_z: float = 0.0,
    distances_m: tuple[float, ...] = GAP_FILL_DISTANCES_M,
    width_m: float = GAP_FILL_WIDTH_M,
    height_m: float = GAP_FILL_HEIGHT_M,
    max_frames: int = 12,
) -> list[dict[str, Any]]:
    """Side / return wall hyps: Manhattan ±90° from street heading @ 6–18 m."""
    if not frames:
        return []
    wall_u = ground_z + height_m * 0.45
    # One seed frame per pano (prefer sideways view if travel known)
    by_pano: dict[str, dict[str, Any]] = {}
    for fr in frames:
        pid = str(fr.get("pano_id") or f"{round(float(fr['e']),2)}_{round(float(fr['n']),2)}")
        travel = fr.get("travel_heading")
        if travel is None:
            by_pano.setdefault(pid, fr)
            continue
        from ps1_hood.geo import heading_diff

        sep = abs(heading_diff(float(fr["heading"]), float(travel)))
        cur = by_pano.get(pid)
        if cur is None:
            by_pano[pid] = fr
        elif 70.0 <= sep <= 110.0:
            by_pano[pid] = fr
    seed_frames = list(by_pano.values())[:max_frames]

    hyps: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int]] = set()

    def _add(n: np.ndarray, d: float, center: np.ndarray, source: str) -> None:
        key = (
            int(round(math.degrees(math.atan2(float(n[0]), float(n[1]))) / 5.0)),
            int(round(d * 2.0)),
            int(round(float(center[0]) + float(center[1]))),
        )
        if key in seen:
            return
        seen.add(key)
        hyps.append(
            {
                "n": n.astype(np.float64),
                "d": float(d),
                "center": center.astype(np.float64),
                "width_m": float(width_m),
                "height_m": float(height_m),
                "source": source,
            }
        )

    for fr in seed_frames:
        C = np.array([float(fr["e"]), float(fr["n"]), wall_u], dtype=np.float64)
        base = float(fr.get("travel_heading") if fr.get("travel_heading") is not None else fr["heading"])
        for yaw_off in (90.0, -90.0):
            heading = wrap_heading(base + yaw_off)
            fwd = np.array(
                [math.sin(math.radians(heading)), math.cos(math.radians(heading)), 0.0],
                dtype=np.float64,
            )
            for dist in distances_m:
                p0 = C + fwd * float(dist)
                p0[2] = wall_u
                n, d = vertical_plane_from_point_heading(p0, heading)
                _add(n, d, p0, "manhattan")
    log.info("gap_fill: %s manhattan side/return seeds", len(hyps))
    return hyps


def corner_seeds_from_roof_aabbs(
    regions: list[dict[str, Any]],
    *,
    ground_z: float = 0.0,
    width_m: float = GAP_FILL_WIDTH_M,
    height_m: float = GAP_FILL_HEIGHT_M,
    inset_m: float = GAP_FILL_CORNER_INSET_M,
    max_corners: int = 24,
) -> list[dict[str, Any]]:
    """Corner wrap hyps: two headings 90° apart near each roof AABB corner (±inset).

    Sat footprints are hypotheses only — never promoted without ZNCC.
    """
    wall_u = ground_z + height_m * 0.45
    hyps: list[dict[str, Any]] = []
    n_corners = 0
    for reg in regions:
        if reg.get("kind") not in {None, "roof", "building"}:
            # Skip yards — gap fill targets building returns
            if reg.get("kind") == "yard":
                continue
        aabb = reg.get("aabb_enu")
        if not aabb or len(aabb) < 4:
            continue
        es = [float(p[0]) for p in aabb]
        ns = [float(p[1]) for p in aabb]
        e0, e1 = min(es), max(es)
        n0, n1 = min(ns), max(ns)
        # Inset corners toward center so centers sit on walls, not mid-street
        ie0, ie1 = e0 + inset_m, e1 - inset_m
        in0, in1 = n0 + inset_m, n1 - inset_m
        if ie1 <= ie0 or in1 <= in0:
            ie0, ie1, in0, in1 = e0, e1, n0, n1
        corners = [
            (ie0, in0),
            (ie1, in0),
            (ie1, in1),
            (ie0, in1),
        ]
        # Outward normals: -E, +E, +N, -N paired with the adjacent side
        # At each corner: two planes with headings along the two AABB edges.
        edge_headings = [
            # SW: +E wall (heading 90) and +N wall (heading 0)
            (90.0, 0.0),
            # SE: -E (270) and +N (0)
            (270.0, 0.0),
            # NE: -E (270) and -N (180)
            (270.0, 180.0),
            # NW: +E (90) and -N (180)
            (90.0, 180.0),
        ]
        for (ce, cn), (h_a, h_b) in zip(corners, edge_headings):
            if n_corners >= max_corners:
                break
            center = np.array([ce, cn, wall_u], dtype=np.float64)
            for heading in (h_a, h_b):
                n, d = vertical_plane_from_point_heading(center, heading)
                hyps.append(
                    {
                        "n": n.astype(np.float64),
                        "d": float(d),
                        "center": center.copy(),
                        "width_m": float(width_m),
                        "height_m": float(height_m),
                        "source": "corner_sat",
                    }
                )
            n_corners += 1
        if n_corners >= max_corners:
            break
    log.info(
        "gap_fill: %s corner_sat seeds from %s roof corners (%s regions)",
        len(hyps),
        n_corners,
        len(regions),
    )
    return hyps


def load_sat_roof_regions(project_root: Path) -> list[dict[str, Any]]:
    """Best-effort sat roof footprints for corner seeds. Empty on miss (no fail)."""
    try:
        from ps1_hood.reconstruct.sat_roofs import (
            extract_roof_yard_polygons,
            load_ortho_for_run,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("gap_fill: sat_roofs import failed (%s)", exc)
        return []
    root = Path(project_root)
    try:
        ortho = load_ortho_for_run(root)
        return extract_roof_yard_polygons(ortho)
    except Exception as exc:  # noqa: BLE001
        log.info("gap_fill: no sat roof footprints (%s) — manhattan seeds only", exc)
        return []


def reject_road_center_hyps(
    hyps: list[dict[str, Any]],
    *,
    cam_xy: np.ndarray | None,
    street_mask: np.ndarray | None = None,
    ortho: Any | None = None,
    margin_m: float = GAP_FILL_CAM_MARGIN_M,
) -> list[dict[str, Any]]:
    """Drop peels / seeds whose XY center sits on the cam corridor or sat road."""
    if not hyps:
        return []
    kept: list[dict[str, Any]] = []
    n_cam = 0
    n_street = 0
    for h in hyps:
        c = np.asarray(h["center"], dtype=np.float64)
        # Camera corridor: center within margin_m of any cam XY → road slab
        if cam_xy is not None and len(cam_xy) > 0:
            dmin = float(np.linalg.norm(cam_xy - c[:2], axis=1).min())
            if dmin < margin_m:
                n_cam += 1
                continue
        # Sat street mask at center pixel
        if street_mask is not None and ortho is not None and hasattr(ortho, "enu_to_px"):
            try:
                u, v = ortho.enu_to_px(float(c[0]), float(c[1]))
                ui, vi = int(round(u)), int(round(v))
                h_m, w_m = street_mask.shape[:2]
                if 0 <= vi < h_m and 0 <= ui < w_m and street_mask[vi, ui] > 0:
                    n_street += 1
                    continue
            except Exception:  # noqa: BLE001
                pass
        kept.append(h)
    if n_cam or n_street:
        log.info(
            "gap_fill: rejected road-center hyps cam=%s street=%s → kept %s/%s",
            n_cam,
            n_street,
            len(kept),
            len(hyps),
        )
    return kept


def prefer_side_wall_order(
    hyps: list[dict[str, Any]],
    travel_heading_deg: float | None,
) -> list[dict[str, Any]]:
    """Stable-sort so normals ⟂ street travel are scored first (gap focus)."""
    if travel_heading_deg is None or not hyps:
        return list(hyps)
    th = math.radians(float(travel_heading_deg))
    travel = np.array([math.sin(th), math.cos(th)], dtype=np.float64)

    def _score(h: dict[str, Any]) -> float:
        n = np.asarray(h["n"], dtype=np.float64)[:2]
        nn = float(np.linalg.norm(n))
        if nn < 1e-9:
            return 0.0
        n = n / nn
        # |n · travel| small ⇒ normal ⟂ travel ⇒ side/return wall
        return 1.0 - abs(float(n @ travel))

    return sorted(hyps, key=_score, reverse=True)


def count_untextured_product(dest_obj: Path) -> tuple[int, int]:
    """Return (n_planes, n_untextured) from live product planes.json / textures."""
    import json

    dest_obj = Path(dest_obj)
    parent = dest_obj.parent
    pj = parent / "planes.json"
    tex_dir = parent / "textures"
    n_planes = 0
    textured_ids: set[str] = set()
    if pj.is_file():
        try:
            payload = json.loads(pj.read_text(encoding="utf-8"))
            planes = payload.get("planes") or []
            n_planes = len(planes)
            for pl in planes:
                tex = pl.get("texture") if isinstance(pl, dict) else None
                if tex:
                    textured_ids.add(str(Path(tex).name))
        except Exception:  # noqa: BLE001
            pass
    if tex_dir.is_dir():
        for p in tex_dir.glob("facade_*.jpg"):
            textured_ids.add(p.name)
    n_tex = len(textured_ids)
    if n_planes <= 0:
        n_planes = n_tex
    n_untex = max(0, n_planes - n_tex)
    return n_planes, n_untex


def build_gap_fill_seeds(
    frames: list[dict[str, Any]],
    project_root: Path,
    *,
    ground_z: float = 0.0,
    cam_xy: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Manhattan side + sat corner seeds, road-rejected, side-wall ordered."""
    man = manhattan_side_seeds(frames, ground_z=ground_z)
    regions = load_sat_roof_regions(project_root)
    corners = corner_seeds_from_roof_aabbs(regions, ground_z=ground_z) if regions else []
    seeds = man + corners

    street_mask = None
    ortho = None
    if regions:
        try:
            from ps1_hood.reconstruct.sat_roofs import (
                load_ortho_for_run,
                segment_roof_yard_mask,
            )

            ortho = load_ortho_for_run(Path(project_root))
            street_mask = getattr(ortho, "_street_mask", None)
            if street_mask is None:
                _, _, street_mask = segment_roof_yard_mask(ortho.image)
        except Exception:  # noqa: BLE001
            pass

    if cam_xy is None and frames:
        xy = np.array(
            [[float(f["e"]), float(f["n"])] for f in frames],
            dtype=np.float64,
        )
        rounded = np.round(xy, 3)
        _, idx = np.unique(rounded, axis=0, return_index=True)
        cam_xy = xy[np.sort(idx)]

    seeds = reject_road_center_hyps(
        seeds, cam_xy=cam_xy, street_mask=street_mask, ortho=ortho
    )
    travel = estimate_travel_heading_deg(frames)
    seeds = prefer_side_wall_order(seeds, travel)
    log.info(
        "gap_fill: built %s seeds (manhattan+corner; travel=%s)",
        len(seeds),
        f"{travel:.1f}" if travel is not None else "None",
    )
    return seeds


def filter_ma_peels_for_gaps(
    hyps: list[dict[str, Any]],
    frames: list[dict[str, Any]],
    project_root: Path | None = None,
    *,
    cam_xy: np.ndarray | None = None,
    peel_cap: int = GAP_FILL_PEEL_CAP,
) -> list[dict[str, Any]]:
    """Road-reject + side-wall prefer + hard peel cap (no peel-as-hero spam)."""
    if cam_xy is None and frames:
        xy = np.array(
            [[float(f["e"]), float(f["n"])] for f in frames],
            dtype=np.float64,
        )
        rounded = np.round(xy, 3)
        _, idx = np.unique(rounded, axis=0, return_index=True)
        cam_xy = xy[np.sort(idx)]

    street_mask = None
    ortho = None
    if project_root is not None:
        try:
            from ps1_hood.reconstruct.sat_roofs import (
                load_ortho_for_run,
                segment_roof_yard_mask,
            )

            ortho = load_ortho_for_run(Path(project_root))
            _, _, street_mask = segment_roof_yard_mask(ortho.image)
        except Exception:  # noqa: BLE001
            pass

    filtered = reject_road_center_hyps(
        hyps, cam_xy=cam_xy, street_mask=street_mask, ortho=ortho
    )
    travel = estimate_travel_heading_deg(frames)
    filtered = prefer_side_wall_order(filtered, travel)
    if len(filtered) > peel_cap:
        log.info(
            "gap_fill: capping MA peels %s → %s (gap focus, not volume)",
            len(filtered),
            peel_cap,
        )
        filtered = filtered[:peel_cap]
    return filtered
