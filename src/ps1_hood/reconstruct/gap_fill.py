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
GAP_FILL_PEEL_CAP = 3  # rethink: cap MA peels; was 32 (spam → clutter)
GAP_FILL_MAX_KEEP = 24
GAP_FILL_MIN_FRONTAL = 0.20  # far-side only (default Path α = 0.25)
GAP_FILL_CAM_MARGIN_M = 3.0
GAP_FILL_CORNER_INSET_M = 1.0
GAP_FILL_WIDTH_M = 8.0
GAP_FILL_HEIGHT_M = 9.0

# Worst-cam targeted seeds (pack: rings × yaw, not global peel spam)
WORST_CAM_DISTANCES_M = (8.0, 12.0, 16.0, 20.0)
WORST_CAM_YAWS_DEG = (0.0, 45.0, -45.0, 90.0, -90.0)
WORST_CAM_MIN_FRONTAL = 0.25
WORST_CAM_HYP_CAP = 40

# Stricter multi-view + sat AABB for *gap adds only* (not product_lock)
GAP_ADD_MIN_ZNCC = 0.35
GAP_ADD_MIN_VIEWS = 2  # ZNCC ≥ min on ≥ this many pair scores / cams
GAP_ADD_MEDIAN_MIN = 0.10  # reject if median ZNCC over scoring views < this
GAP_ADD_MIN_MAX_FRONTAL = 0.4  # reject if max |n·cam_fwd| over scoring cams < this
GAP_ADD_SAT_EDGE_M = 2.0  # center within this of a roof AABB *boundary* edge
GAP_ADD_SAT_EDGE_ALIGN = 0.7  # |n · edge_tangent| ≥ this (plane ∥ edge)
GAP_ADD_MAX_ADDS = 3  # hard cap on new planes after gates
GAP_SEEDS_MODES = ("legacy", "sat-edge", "both")
DEFAULT_GAP_SEEDS = "sat-edge"  # uncovered roof AABB edges → same-side facing cams


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



def parse_cam_id(cam_id: str) -> tuple[str, float]:
    """Parse compare/overlay cam id ``pano_hNNN`` → (pano_id, heading_deg).

    Heading suffix is ``_h`` + integer degrees (e.g. ``_h000``, ``_h120``).
    """
    cid = str(cam_id or "").strip()
    if not cid:
        raise ValueError("empty cam_id")
    # Prefer last ``_hNNN`` segment (pano ids may contain underscores)
    idx = cid.rfind("_h")
    if idx < 0:
        raise ValueError(f"cam_id missing _hNNN suffix: {cam_id!r}")
    pano = cid[:idx]
    rest = cid[idx + 2 :]
    if not rest.isdigit():
        raise ValueError(f"cam_id heading not integer degrees: {cam_id!r}")
    heading = float(int(rest) % 360)
    if not pano:
        raise ValueError(f"cam_id missing pano_id: {cam_id!r}")
    return pano, heading


def make_frame_cam_id(frame: dict[str, Any]) -> str:
    """Match ``compare.make_cam_id`` without importing compare (light helper)."""
    pid = str(frame.get("pano_id") or "cam")
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in pid)
    h = int(round(float(frame.get("heading") or 0.0))) % 360
    return f"{safe}_h{h:03d}"


def find_frame_for_cam_id(
    frames: list[dict[str, Any]], cam_id: str
) -> dict[str, Any] | None:
    """Resolve a worst-cam id to a keyframe dict (exact id, else pano+heading)."""
    from ps1_hood.geo import heading_diff

    want = str(cam_id).strip()
    for fr in frames:
        if make_frame_cam_id(fr) == want:
            return fr
    try:
        pano, heading = parse_cam_id(want)
    except ValueError:
        return None
    best: dict[str, Any] | None = None
    best_sep = 999.0
    for fr in frames:
        if str(fr.get("pano_id") or "") != pano:
            continue
        sep = abs(heading_diff(float(fr.get("heading") or 0.0), heading))
        if sep < best_sep:
            best_sep = sep
            best = fr
    if best is not None and best_sep <= 5.0:
        return best
    return None


def load_worst_cam_ids_from_compare(
    project_root: Path, n: int = 3
) -> list[str]:
    """Read ``recon/compare/summary.json`` finite-worst list (top-N ids)."""
    import json

    root = Path(project_root)
    path = root / "recon" / "compare" / "summary.json"
    if not path.is_file():
        log.warning("gap_fill: no compare summary at %s", path)
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        log.warning("gap_fill: compare summary unreadable (%s)", exc)
        return []
    worst = payload.get("worst") or []
    ids: list[str] = []
    for row in worst:
        if isinstance(row, dict) and row.get("id"):
            ids.append(str(row["id"]))
        elif isinstance(row, str):
            ids.append(row)
        if len(ids) >= int(n):
            break
    return ids


def hyp_frontal_to_frame(hyp: dict[str, Any], frame: dict[str, Any]) -> float:
    """Frontal score of hyp normal vs camera forward (0..1-ish; may be neg)."""
    n = np.asarray(hyp["n"], dtype=np.float64)
    center = np.asarray(hyp["center"], dtype=np.float64)
    n_xy = n[:2] / (np.linalg.norm(n[:2]) + 1e-12)
    C = np.array(
        [float(frame["e"]), float(frame["n"]), float(frame.get("u") or 0.0)],
        dtype=np.float64,
    )
    to_cam = C[:2] - center[:2]
    n_use = n_xy.copy()
    if float(n_use @ to_cam) < 0:
        n_use = -n_use
    h = math.radians(float(frame.get("heading") or 0.0))
    # camera_rotation_cv forward xy = (sin h, cos h)
    fwd = np.array([math.sin(h), math.cos(h)], dtype=np.float64)
    return float((-fwd) @ n_use)


def filter_hyps_visible_in_cams(
    hyps: list[dict[str, Any]],
    cam_frames: list[dict[str, Any]],
    *,
    min_frontal: float = WORST_CAM_MIN_FRONTAL,
) -> list[dict[str, Any]]:
    """Keep hyps with frontal≥min to *at least one* of the given cams."""
    if not hyps:
        return []
    if not cam_frames:
        return list(hyps)
    kept: list[dict[str, Any]] = []
    for h in hyps:
        if any(hyp_frontal_to_frame(h, fr) >= float(min_frontal) for fr in cam_frames):
            kept.append(h)
    return kept


def worst_cam_seeds(
    frames: list[dict[str, Any]],
    cam_ids: list[str],
    *,
    ground_z: float = 0.0,
    distances_m: tuple[float, ...] = WORST_CAM_DISTANCES_M,
    yaws_deg: tuple[float, ...] = WORST_CAM_YAWS_DEG,
    width_m: float = GAP_FILL_WIDTH_M,
    height_m: float = GAP_FILL_HEIGHT_M,
    hyp_cap: int = WORST_CAM_HYP_CAP,
) -> list[dict[str, Any]]:
    """Seed vertical planes in front of worst cams (dist rings × yaw).

    For each cam C: p0 = C_xy + rot(heading+yaw)*dist; plane faces heading+yaw;
    source tagged ``worst_cam`` (+ ``pano_id``). Caps total hyps.
    """
    if not frames or not cam_ids:
        return []
    wall_u = ground_z + height_m * 0.45  # ~3–4 m for default height 9
    hyps: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int]] = set()

    def _add(
        n: np.ndarray,
        d: float,
        center: np.ndarray,
        *,
        pano_id: str,
        cam_id: str,
    ) -> None:
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
                "source": "worst_cam",
                "pano_id": pano_id,
                "cam_id": cam_id,
            }
        )

    for cid in cam_ids:
        fr = find_frame_for_cam_id(frames, cid)
        if fr is None:
            log.warning("gap_fill: worst cam id not in frames: %s", cid)
            continue
        try:
            pano, _heading_parsed = parse_cam_id(cid)
        except ValueError:
            pano = str(fr.get("pano_id") or "")
        C = np.array(
            [float(fr["e"]), float(fr["n"]), wall_u],
            dtype=np.float64,
        )
        base_h = float(fr.get("heading") or 0.0)
        for dist in distances_m:
            for yaw in yaws_deg:
                heading = wrap_heading(base_h + float(yaw))
                fwd = np.array(
                    [
                        math.sin(math.radians(heading)),
                        math.cos(math.radians(heading)),
                        0.0,
                    ],
                    dtype=np.float64,
                )
                p0 = C + fwd * float(dist)
                p0[2] = wall_u
                n, d = vertical_plane_from_point_heading(p0, heading)
                _add(n, d, p0, pano_id=pano, cam_id=str(cid))
                if len(hyps) >= int(hyp_cap):
                    break
            if len(hyps) >= int(hyp_cap):
                break
        if len(hyps) >= int(hyp_cap):
            break

    log.info(
        "gap_fill: %s worst_cam seeds from %s cam ids (cap=%s)",
        len(hyps),
        len(cam_ids),
        hyp_cap,
    )
    return hyps


def frame_indices_for_cam_ids(
    frames: list[dict[str, Any]], cam_ids: list[str]
) -> list[int]:
    """Frame indices matching worst cam ids (for view-picker boost)."""
    idxs: list[int] = []
    for cid in cam_ids:
        fr = find_frame_for_cam_id(frames, cid)
        if fr is None:
            continue
        for i, f in enumerate(frames):
            if f is fr or (
                str(f.get("pano_id")) == str(fr.get("pano_id"))
                and abs(float(f.get("heading") or 0.0) - float(fr.get("heading") or 0.0))
                < 0.5
            ):
                if i not in idxs:
                    idxs.append(i)
                break
    return idxs


# ---------------------------------------------------------------------------
# sat_edge seeds — uncovered roof AABB edges + same-side facing cams
# ---------------------------------------------------------------------------

SAT_EDGE_COVER_ALIGN = 0.7  # |n · n_out| for product plane covering an edge
SAT_EDGE_COVER_DIST_M = 2.0  # |d + n·M| plane-to-midpoint
SAT_EDGE_COVER_CENTER_M = 4.0  # product center within this of segment
SAT_EDGE_CAM_DIST = (6.0, 35.0)
SAT_EDGE_LOOK_AT = 0.5  # (v/dist)·fwd
SAT_EDGE_GRAZE = 0.35  # |n_out · fwd| below this = grazing past wall
SAT_EDGE_WALL_Z = 3.5  # center height above ground
SAT_EDGE_MAX_EDGES = 8


def _roof_aabb_xy(aabb_enu: list | tuple) -> tuple[float, float, float, float] | None:
    pts = [(float(p[0]), float(p[1])) for p in aabb_enu]
    if len(pts) < 2:
        return None
    es = [p[0] for p in pts]
    ns = [p[1] for p in pts]
    return min(es), max(es), min(ns), max(ns)


def aabb_boundary_edges(
    aabb_enu: list | tuple,
    *,
    roof_id: str = "",
) -> list[dict[str, Any]]:
    """Four AABB sides with midpoint, tangent, outward normal (away from center)."""
    box = _roof_aabb_xy(aabb_enu)
    if box is None:
        return []
    e0, e1, n0, n1 = box
    roof_c = np.array([(e0 + e1) * 0.5, (n0 + n1) * 0.5], dtype=np.float64)
    # SW, SE, NE, NW ring
    corners = [
        np.array([e0, n0], dtype=np.float64),
        np.array([e1, n0], dtype=np.float64),
        np.array([e1, n1], dtype=np.float64),
        np.array([e0, n1], dtype=np.float64),
    ]
    edges: list[dict[str, Any]] = []
    for i in range(4):
        p0 = corners[i]
        p1 = corners[(i + 1) % 4]
        t = p1 - p0
        tn = float(np.linalg.norm(t))
        if tn < 1e-6:
            continue
        tang = t / tn
        mid = 0.5 * (p0 + p1)
        # rotate90 CCW / CW; pick outward (away from roof center)
        n_ccw = np.array([-float(tang[1]), float(tang[0])], dtype=np.float64)
        n_cw = np.array([float(tang[1]), -float(tang[0])], dtype=np.float64)
        away = mid - roof_c
        n_out = n_ccw if float(n_ccw @ away) >= float(n_cw @ away) else n_cw
        edges.append(
            {
                "id": f"{roof_id or 'roof'}_e{i}",
                "roof_id": roof_id or "roof",
                "p0": p0,
                "p1": p1,
                "mid": mid,
                "tangent": tang,
                "n_out": n_out,
                "roof_c": roof_c.copy(),
            }
        )
    return edges


def product_plane_covers_edge(
    plane: dict[str, Any],
    edge: dict[str, Any],
    *,
    align_min: float = SAT_EDGE_COVER_ALIGN,
    plane_dist_m: float = SAT_EDGE_COVER_DIST_M,
    center_m: float = SAT_EDGE_COVER_CENTER_M,
) -> bool:
    """True if product plane faces this edge outward and sits on the wall."""
    n = np.asarray(plane.get("n"), dtype=np.float64).reshape(-1)
    if n.size < 2:
        return False
    n_xy = n[:2]
    nn = float(np.linalg.norm(n_xy))
    if nn < 1e-9:
        return False
    n_xy = n_xy / nn
    n_out = np.asarray(edge["n_out"], dtype=np.float64)[:2]
    if abs(float(n_xy @ n_out)) < float(align_min):
        return False
    M = np.asarray(edge["mid"], dtype=np.float64)[:2]
    # |d + n·M| for plane n·X + d = 0 (vertical → z term ~0)
    d = plane.get("d")
    try:
        d_f = float(d) if d is not None else float("nan")
    except (TypeError, ValueError):
        d_f = float("nan")
    if math.isfinite(d_f):
        n3 = np.array(
            [float(n_xy[0]), float(n_xy[1]), float(n[2]) if n.size >= 3 else 0.0],
            dtype=np.float64,
        )
        M3 = np.array([float(M[0]), float(M[1]), 0.0], dtype=np.float64)
        dist = abs(float(d_f) + float(n3 @ M3))
    else:
        c = plane.get("center")
        if c is None:
            return False
        c = np.asarray(c, dtype=np.float64)[:2]
        dist = abs(float(n_xy @ (c - M)))
    if dist > float(plane_dist_m):
        return False
    center = plane.get("center")
    if center is None:
        return False
    cxy = np.asarray(center, dtype=np.float64)[:2]
    if dist_point_to_segment_xy(cxy, edge["p0"], edge["p1"]) > float(center_m):
        return False
    return True


def uncovered_roof_edges(
    regions: list[dict[str, Any]],
    product_planes: list[dict[str, Any]] | None,
    *,
    max_edges: int = SAT_EDGE_MAX_EDGES,
) -> list[dict[str, Any]]:
    """AABB roof/building edges with no covering product plane."""
    planes = list(product_planes or [])
    uncovered: list[dict[str, Any]] = []
    for ri, reg in enumerate(regions):
        kind = reg.get("kind")
        if kind not in {None, "roof", "building"}:
            continue
        aabb = reg.get("aabb_enu")
        if not aabb or len(aabb) < 4:
            continue
        roof_id = str(reg.get("id") or f"roof_{ri}")
        for edge in aabb_boundary_edges(aabb, roof_id=roof_id):
            covered = any(product_plane_covers_edge(pl, edge) for pl in planes)
            if not covered:
                uncovered.append(edge)
            if len(uncovered) >= int(max_edges) * 4:
                # collect generously; caller slices
                pass
    return uncovered[: int(max_edges)] if max_edges else uncovered


def write_gap_needs_json(
    path: Path,
    needs: list[dict[str, Any]],
    *,
    run: str = "smoke-dense",
    product_planes: int = 18,
) -> Path:
    """Append/write recon/gap_needs.json for edges with zero facing cams."""
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run": str(run),
        "product_planes": int(product_planes),
        "needs": list(needs),
        "hint": (
            "fetch far-side SV panos looking at these ENU points; fixed poses only"
        ),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def sat_edge_seeds(
    frames: list[dict[str, Any]],
    regions: list[dict[str, Any]],
    product_planes: list[dict[str, Any]] | None,
    *,
    ground_z: float = 0.0,
    max_edges: int = SAT_EDGE_MAX_EDGES,
    width_m: float = GAP_FILL_WIDTH_M,
    height_m: float = GAP_FILL_HEIGHT_M,
    travel_heading_deg: float | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Uncovered sat roof edges → sat_edge hyps from same-side facing cams.

    Returns ``(hyps, needs)``. ``needs`` entries get no MA peels invented —
    caller writes ``gap_needs.json``.
    """
    hyps: list[dict[str, Any]] = []
    needs: list[dict[str, Any]] = []
    if not regions:
        return hyps, needs
    edges = uncovered_roof_edges(
        regions, product_planes, max_edges=int(max_edges)
    )
    wall_u = float(ground_z) + float(SAT_EDGE_WALL_Z)
    travel = travel_heading_deg
    if travel is None:
        travel = estimate_travel_heading_deg(frames)
    d_lo, d_hi = SAT_EDGE_CAM_DIST

    for edge in edges:
        M = np.asarray(edge["mid"], dtype=np.float64)[:2]
        n_out = np.asarray(edge["n_out"], dtype=np.float64)[:2]
        nn = float(np.linalg.norm(n_out))
        if nn < 1e-9:
            continue
        n_out = n_out / nn
        roof_c = np.asarray(edge["roof_c"], dtype=np.float64)[:2]
        heading = wrap_heading(
            math.degrees(math.atan2(float(n_out[0]), float(n_out[1])))
        )
        cams: list[dict[str, Any]] = []
        for fr in frames:
            C = np.array([float(fr["e"]), float(fr["n"])], dtype=np.float64)
            h = math.radians(float(fr.get("heading") or 0.0))
            fwd = np.array([math.sin(h), math.cos(h)], dtype=np.float64)
            v = M - C
            dist = float(np.linalg.norm(v))
            if dist < d_lo or dist > d_hi:
                continue
            v_hat = v / dist
            if float(v_hat @ fwd) < float(SAT_EDGE_LOOK_AT):
                continue
            if abs(float(n_out @ fwd)) < float(SAT_EDGE_GRAZE):
                continue
            # same side of street as roof: cam outside the building
            if float((C - roof_c) @ n_out) < 0.0:
                continue
            cams.append(fr)
        if not cams:
            needs.append(
                {
                    "e": float(M[0]),
                    "n": float(M[1]),
                    "heading": float(heading),
                    "reason": "no_facing_cam",
                    "roof_id": str(edge.get("roof_id") or ""),
                    "edge_id": str(edge.get("id") or ""),
                }
            )
            continue
        # prefer return walls: |heading − travel| ≈ 90°
        ret_score = 0.0
        if travel is not None:
            from ps1_hood.geo import heading_diff

            sep = abs(heading_diff(float(heading), float(travel)))
            ret_score = 1.0 - abs(sep - 90.0) / 90.0
        n_out_3d = np.array(
            [float(n_out[0]), float(n_out[1]), 0.0], dtype=np.float64
        )
        center = np.array(
            [float(M[0]), float(M[1]), wall_u], dtype=np.float64
        )
        d_plane = -float(n_out_3d @ center)
        seed_cams = [make_frame_cam_id(c) for c in cams[:4]]
        hyps.append(
            {
                "n": n_out_3d,
                "d": d_plane,
                "center": center,
                "width_m": float(width_m),
                "height_m": float(height_m),
                "source": "sat_edge",
                "edge_id": str(edge.get("id") or ""),
                "roof_id": str(edge.get("roof_id") or ""),
                "seed_cams": seed_cams,
                "return_score": float(ret_score),
            }
        )

    # Prefer return walls first
    hyps.sort(key=lambda h: float(h.get("return_score") or 0.0), reverse=True)
    log.info(
        "gap_fill: %s sat_edge seeds from %s uncovered edges (%s needs)",
        len(hyps),
        len(edges),
        len(needs),
    )
    if needs:
        log.info(
            "gap_fill: %s sat_edge needs (no facing cam)",
            len(needs),
        )
    return hyps, needs


def build_gap_fill_seeds(
    frames: list[dict[str, Any]],
    project_root: Path,
    *,
    ground_z: float = 0.0,
    cam_xy: np.ndarray | None = None,
    worst_cam_ids: list[str] | None = None,
    hyp_cap: int = WORST_CAM_HYP_CAP,
    gap_seeds_mode: str = DEFAULT_GAP_SEEDS,
    product_planes: list[dict[str, Any]] | None = None,
    write_needs: bool = True,
    run_name: str | None = None,
) -> list[dict[str, Any]]:
    """Gap seeds by ``gap_seeds_mode``.

    - ``legacy`` — manhattan + corner_sat (+ optional worst-cam)
    - ``sat-edge`` — uncovered roof AABB edges → same-side facing cams only
    - ``both`` — sat_edge first, then AABB-gated legacy

    When sat-edge/both leave edges with zero facing cams, write
    ``recon/gap_needs.json`` (no invented MA peels).
    """
    mode = str(gap_seeds_mode or DEFAULT_GAP_SEEDS).strip().lower()
    if mode not in GAP_SEEDS_MODES:
        log.warning(
            "gap_fill: unknown gap_seeds_mode=%r — using %s",
            gap_seeds_mode,
            DEFAULT_GAP_SEEDS,
        )
        mode = DEFAULT_GAP_SEEDS

    root = Path(project_root)
    regions = load_sat_roof_regions(root)
    travel = estimate_travel_heading_deg(frames)

    street_mask = None
    ortho = None
    if regions:
        try:
            from ps1_hood.reconstruct.sat_roofs import (
                load_ortho_for_run,
                segment_roof_yard_mask,
            )

            ortho = load_ortho_for_run(root)
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

    sat_hyps: list[dict[str, Any]] = []
    needs: list[dict[str, Any]] = []
    if mode in {"sat-edge", "both"}:
        sat_hyps, needs = sat_edge_seeds(
            frames,
            regions,
            product_planes,
            ground_z=ground_z,
            travel_heading_deg=travel,
        )
        sat_hyps = reject_road_center_hyps(
            sat_hyps, cam_xy=cam_xy, street_mask=street_mask, ortho=ortho
        )
        if needs and write_needs:
            n_prod = len(product_planes or [])
            write_gap_needs_json(
                root / "recon" / "gap_needs.json",
                needs,
                run=run_name or root.name or "smoke-dense",
                product_planes=n_prod if n_prod > 0 else 18,
            )

    legacy: list[dict[str, Any]] = []
    if mode in {"legacy", "both"}:
        worst_ids = [
            str(c).strip() for c in (worst_cam_ids or []) if str(c).strip()
        ]
        worst_frames = [
            fr
            for cid in worst_ids
            if (fr := find_frame_for_cam_id(frames, cid)) is not None
        ]
        worst = (
            worst_cam_seeds(frames, worst_ids, ground_z=ground_z, hyp_cap=hyp_cap)
            if worst_ids
            else []
        )
        man = manhattan_side_seeds(frames, ground_z=ground_z)
        corners = (
            corner_seeds_from_roof_aabbs(regions, ground_z=ground_z)
            if regions
            else []
        )
        side_corner = man + corners
        side_corner = reject_road_center_hyps(
            side_corner, cam_xy=cam_xy, street_mask=street_mask, ortho=ortho
        )
        worst = reject_road_center_hyps(
            worst, cam_xy=cam_xy, street_mask=street_mask, ortho=ortho
        )
        if worst_frames:
            before = len(side_corner)
            side_corner = filter_hyps_visible_in_cams(
                side_corner, worst_frames, min_frontal=WORST_CAM_MIN_FRONTAL
            )
            log.info(
                "gap_fill: filtered manhattan/corner to worst-cam visible "
                "%s → %s (min_frontal=%.2f)",
                before,
                len(side_corner),
                WORST_CAM_MIN_FRONTAL,
            )
        side_corner = prefer_side_wall_order(side_corner, travel)
        # both: AABB-gate legacy so they don't spam off-footprint
        if mode == "both" and regions:
            gated: list[dict[str, Any]] = []
            for h in worst + side_corner:
                ok, _why = sat_aabb_edge_ok(
                    np.asarray(h["center"], dtype=np.float64),
                    np.asarray(h["n"], dtype=np.float64),
                    regions,
                )
                if ok:
                    gated.append(h)
            legacy = gated
            log.info(
                "gap_fill: AABB-gated legacy %s → %s (both mode)",
                len(worst) + len(side_corner),
                len(legacy),
            )
        else:
            legacy = list(worst)
            for h in side_corner:
                legacy.append(h)

    # Order: sat_edge first, then legacy; cap
    seeds: list[dict[str, Any]] = []
    for h in sat_hyps + legacy:
        if len(seeds) >= int(hyp_cap):
            break
        seeds.append(h)

    log.info(
        "gap_fill: built %s seeds (mode=%s sat_edge=%s legacy=%s needs=%s "
        "travel=%s)",
        len(seeds),
        mode,
        len(sat_hyps),
        len(legacy),
        len(needs),
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


# ---------------------------------------------------------------------------
# Stricter multi-view + sat AABB gates (gap adds only — never product_lock)
# ---------------------------------------------------------------------------


def _aabb_edges_xy(
    aabb_enu: list | tuple,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return (p0, p1, tangent_unit) for each AABB boundary edge (XY)."""
    pts = [(float(p[0]), float(p[1])) for p in aabb_enu]
    if len(pts) < 2:
        return []
    # Close ring if needed
    if pts[0] != pts[-1]:
        pts = list(pts) + [pts[0]]
    edges: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for i in range(len(pts) - 1):
        a = np.array(pts[i], dtype=np.float64)
        b = np.array(pts[i + 1], dtype=np.float64)
        t = b - a
        tn = float(np.linalg.norm(t))
        if tn < 1e-6:
            continue
        edges.append((a, b, t / tn))
    return edges


def dist_point_to_segment_xy(
    p: np.ndarray, a: np.ndarray, b: np.ndarray
) -> float:
    """Euclidean distance from XY point to segment a→b."""
    p = np.asarray(p, dtype=np.float64)[:2]
    a = np.asarray(a, dtype=np.float64)[:2]
    b = np.asarray(b, dtype=np.float64)[:2]
    ab = b - a
    L2 = float(ab @ ab)
    if L2 < 1e-12:
        return float(np.linalg.norm(p - a))
    t = float(np.clip(((p - a) @ ab) / L2, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def nearest_roof_boundary(
    center_xy: np.ndarray,
    regions: list[dict[str, Any]],
) -> tuple[float, np.ndarray | None, dict[str, Any] | None]:
    """Nearest sat roof AABB boundary edge to center.

    Returns (dist_m, edge_tangent_xy or None, region or None).
    Skips yards. Soft-empty when no roofs.
    """
    c = np.asarray(center_xy, dtype=np.float64)[:2]
    best_d = float("inf")
    best_t: np.ndarray | None = None
    best_reg: dict[str, Any] | None = None
    for reg in regions:
        kind = reg.get("kind")
        if kind == "yard":
            continue
        aabb = reg.get("aabb_enu")
        if not aabb or len(aabb) < 2:
            continue
        for a, b, tang in _aabb_edges_xy(aabb):
            d = dist_point_to_segment_xy(c, a, b)
            if d < best_d:
                best_d = d
                best_t = tang
                best_reg = reg
    if best_t is None:
        return float("inf"), None, None
    return float(best_d), best_t, best_reg


def sat_aabb_edge_ok(
    center: np.ndarray,
    n: np.ndarray,
    regions: list[dict[str, Any]] | None,
    *,
    max_edge_m: float = GAP_ADD_SAT_EDGE_M,
    min_align: float = GAP_ADD_SAT_EDGE_ALIGN,
) -> tuple[bool, str]:
    """Hard gate: center ≤ max_edge_m of a roof boundary + n ∥ edge.

    Soft-pass when no roof regions (can't gate without sat footprints).
    """
    if not regions:
        return True, "no_roofs_soft_pass"
    d, tang, _reg = nearest_roof_boundary(center, regions)
    if not math.isfinite(d) or tang is None:
        return True, "no_roofs_soft_pass"
    if d > float(max_edge_m):
        return False, f"sat_aabb_edge dist={d:.2f}>{max_edge_m}"
    n_xy = np.asarray(n, dtype=np.float64)[:2]
    nn = float(np.linalg.norm(n_xy))
    if nn < 1e-9:
        return False, "sat_aabb_edge degenerate_n"
    n_xy = n_xy / nn
    # Plane ≈ parallel to edge ⇒ normal ⟂ tangent ⇒ |n · tang| small;
    # pack asks |n · edge_tangent| ≥ 0.7 meaning plane normal aligns with…
    # Re-read: "|n · edge_tangent| ≥ 0.7  # plane ≈ parallel to that roof edge"
    # If n is plane normal and edge_tangent is along the wall, parallel plane
    # means n ⟂ tangent → |n·t| ≈ 0. That's the opposite of ≥0.7.
    # They likely meant |n · edge_outward_normal| ≥ 0.7, OR
    # |n × k · tangent| / alignment of plane with edge.
    # For a vertical wall along edge tangent t=(tx,ty), plane normal should be
    # perpendicular to t: |n·t| small. Pack text says ≥0.7 with comment
    # "plane ≈ parallel to that roof edge" — that's inconsistent with ·tangent.
    # Interpret as: rotate tangent 90° → edge outward in XY; |n · n_edge| ≥ 0.7.
    n_edge = np.array([-float(tang[1]), float(tang[0])], dtype=np.float64)
    align = abs(float(n_xy @ n_edge))
    if align < float(min_align):
        return False, f"sat_aabb_align |n·n_edge|={align:.2f}<{min_align}"
    return True, f"sat_aabb_ok d={d:.2f} align={align:.2f}"


def cam_forward_xy(frame: dict[str, Any]) -> np.ndarray:
    h = math.radians(float(frame.get("heading") or 0.0))
    return np.array([math.sin(h), math.cos(h)], dtype=np.float64)


def max_abs_n_dot_cam_fwd(
    n: np.ndarray,
    frames: list[dict[str, Any]],
    view_indices: list[int] | None,
) -> float:
    """Max |n_xy · cam_fwd| over scoring cams (grazing if this is small)."""
    n_xy = np.asarray(n, dtype=np.float64)[:2]
    nn = float(np.linalg.norm(n_xy))
    if nn < 1e-9:
        return 0.0
    n_xy = n_xy / nn
    idxs = list(view_indices) if view_indices else list(range(len(frames)))
    best = 0.0
    for i in idxs:
        if i < 0 or i >= len(frames):
            continue
        fwd = cam_forward_xy(frames[i])
        best = max(best, abs(float(n_xy @ fwd)))
    return float(best)


def gap_add_multiview_ok(
    plane: dict[str, Any],
    frames: list[dict[str, Any]],
    *,
    min_zncc: float = GAP_ADD_MIN_ZNCC,
    min_views: int = GAP_ADD_MIN_VIEWS,
    median_min: float = GAP_ADD_MEDIAN_MIN,
    min_max_frontal: float = GAP_ADD_MIN_MAX_FRONTAL,
) -> tuple[bool, str]:
    """Stricter multi-view accept for gap adds (not locked product).

    - ≥ min_views pairwise/source scores with ZNCC ≥ min_zncc
    - median of finite scores ≥ median_min
    - max |n·cam_fwd| over scoring cams ≥ min_max_frontal (else grazing)
    """
    scores_raw = plane.get("scores")
    scores: list[float] = []
    if isinstance(scores_raw, (list, tuple)):
        for s in scores_raw:
            try:
                v = float(s)
            except (TypeError, ValueError):
                continue
            if math.isfinite(v):
                scores.append(v)
    # Fallback: single aggregate zncc counts as one cam score
    if not scores:
        z = plane.get("zncc")
        try:
            zv = float(z) if z is not None else float("nan")
        except (TypeError, ValueError):
            zv = float("nan")
        if math.isfinite(zv):
            scores = [zv]

    n_good = sum(1 for s in scores if s >= float(min_zncc))
    # Pairwise scores are ref↔source; count ref too when aggregate mean ≥ min
    # so a single strong pair with mean≥min yields 2 cams (ref+source).
    z_mean = float("nan")
    try:
        if plane.get("zncc") is not None:
            z_mean = float(plane["zncc"])
    except (TypeError, ValueError):
        z_mean = float("nan")
    n_cams = int(n_good)
    if math.isfinite(z_mean) and z_mean >= float(min_zncc):
        n_cams = int(n_good) + 1
    if n_cams < int(min_views):
        return (
            False,
            f"multiview n_cams={n_cams}<{min_views} "
            f"(n_good_src={n_good}, need ZNCC≥{min_zncc})",
        )
    med = float(np.median(scores)) if scores else float("nan")
    if not math.isfinite(med) or med < float(median_min):
        return False, f"multiview median={med:.3f}<{median_min}"

    n = np.asarray(plane.get("n"), dtype=np.float64)
    view_idx = plane.get("view_indices")
    idxs = list(view_idx) if isinstance(view_idx, (list, tuple)) else None
    max_fr = max_abs_n_dot_cam_fwd(n, frames, idxs)
    if max_fr < float(min_max_frontal):
        return False, f"grazing max|n·fwd|={max_fr:.3f}<{min_max_frontal}"
    return True, (
        f"multiview ok n_cams={n_cams} n_good_src={n_good} "
        f"med={med:.3f} max_fr={max_fr:.3f}"
    )


def filter_gap_adds(
    planes: list[dict[str, Any]],
    frames: list[dict[str, Any]],
    *,
    roof_regions: list[dict[str, Any]] | None = None,
    sat_aabb_gate_m: float | None = GAP_ADD_SAT_EDGE_M,
    sat_align: float = GAP_ADD_SAT_EDGE_ALIGN,
    min_zncc: float = GAP_ADD_MIN_ZNCC,
    min_views: int = GAP_ADD_MIN_VIEWS,
    median_min: float = GAP_ADD_MEDIAN_MIN,
    min_max_frontal: float = GAP_ADD_MIN_MAX_FRONTAL,
    max_gap_adds: int = GAP_ADD_MAX_ADDS,
) -> list[dict[str, Any]]:
    """Apply sat AABB + multi-view gates to gap-add candidates; cap count.

    Does **not** filter ``product_lock`` (caller should only pass new adds).
    Soft-passes sat AABB when ``sat_aabb_gate_m`` is None/≤0 or no roofs.
    """
    if not planes:
        return []
    regions = list(roof_regions or [])
    kept: list[dict[str, Any]] = []
    n_mv = n_sat = 0
    for p in planes:
        src = str(p.get("source") or "")
        if src == "product_lock":
            kept.append(p)
            continue
        ok_mv, why_mv = gap_add_multiview_ok(
            p,
            frames,
            min_zncc=min_zncc,
            min_views=min_views,
            median_min=median_min,
            min_max_frontal=min_max_frontal,
        )
        if not ok_mv:
            n_mv += 1
            log.info("gap_fill reject multiview [%s]: %s", src, why_mv)
            continue
        if sat_aabb_gate_m is not None and float(sat_aabb_gate_m) > 0:
            center = np.asarray(
                p.get("center") if p.get("center") is not None else p.get("quad_center"),
                dtype=np.float64,
            )
            if center.size < 2 and p.get("corners") is not None:
                center = np.asarray(p["corners"], dtype=np.float64).reshape(-1, 3).mean(
                    axis=0
                )
            n = np.asarray(p.get("n"), dtype=np.float64)
            ok_sat, why_sat = sat_aabb_edge_ok(
                center,
                n,
                regions,
                max_edge_m=float(sat_aabb_gate_m),
                min_align=float(sat_align),
            )
            if not ok_sat:
                n_sat += 1
                log.info("gap_fill reject sat_aabb [%s]: %s", src, why_sat)
                continue
            p = dict(p)
            p["gap_sat_gate"] = why_sat
        p = dict(p)
        p["gap_multiview"] = why_mv
        kept.append(p)

    # Prefer sat_edge > manhattan/worst/corner > ma_segment; then higher ZNCC
    def _rank(pl: dict[str, Any]) -> tuple[int, float]:
        src = str(pl.get("source") or "")
        if "sat_edge" in src:
            tier = 0
        elif "ma_segment" in src or src == "ma_segment":
            tier = 2
        else:
            tier = 1
        z = float(pl.get("zncc") or 0.0)
        return (tier, -z)

    kept.sort(key=_rank)
    if len(kept) > int(max_gap_adds):
        log.info(
            "gap_fill: capping gap adds %s → %s (max_gap_adds)",
            len(kept),
            max_gap_adds,
        )
        kept = kept[: int(max_gap_adds)]
    log.info(
        "gap_fill filter_gap_adds: in=%s out=%s reject_mv=%s reject_sat=%s "
        "sat_gate_m=%s min_views=%s max_adds=%s",
        len(planes),
        len(kept),
        n_mv,
        n_sat,
        sat_aabb_gate_m,
        min_views,
        max_gap_adds,
    )
    return kept
