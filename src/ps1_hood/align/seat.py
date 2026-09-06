"""Keep Street View cameras on the street, outside 3DBAG footprints.

Google GPS + unconstrained Canny matching walks panos through living
rooms. The road and the building outlines are the teacher; photos only
nudge along the street.
"""

from __future__ import annotations

import math
from typing import Any

from shapely.geometry import MultiPoint, Point, Polygon
from shapely.ops import nearest_points

from ps1_hood.geo import heading_diff, wrap_heading


def building_footprints(buildings: list[dict[str, Any]]) -> list[Polygon]:
    """2D lots from ground vertices. Wall triangles project to lines, so we
    hull the points instead of unioning faces."""
    out: list[Polygon] = []
    for bld in buildings:
        verts = bld.get("vertices") or []
        if len(verts) < 3:
            continue
        zs = [v[2] for v in verts]
        zcut = min(zs) + 2.5
        ground = [(v[0], v[1]) for v in verts if v[2] <= zcut]
        if len(ground) < 3:
            ground = [(v[0], v[1]) for v in verts]
        hull = MultiPoint(ground).convex_hull
        if hull.geom_type != "Polygon" or hull.area < 4.0 or hull.area > 2500.0:
            continue
        out.append(hull)
    return out


def _inside_or_close(pt: Point, fp: Polygon, margin_m: float) -> bool:
    if fp.is_empty:
        return False
    if fp.contains(pt) or fp.covers(pt):
        return True
    return fp.distance(pt) < margin_m


def push_out_of_footprints(
    e: float, n: float, footprints: list[Polygon], margin_m: float = 2.2
) -> tuple[float, float]:
    """Slide a camera onto the street if it sits in (or on) a building."""
    pt = Point(e, n)
    for _ in range(6):
        moved = False
        for fp in footprints:
            if not _inside_or_close(pt, fp, margin_m):
                continue
            inside = fp.contains(pt) or fp.covers(pt) or fp.distance(pt) < 1e-6
            _, hit = nearest_points(pt, fp.boundary)
            if inside:
                vx, vy = hit.x - pt.x, hit.y - pt.y
            else:
                vx, vy = pt.x - hit.x, pt.y - hit.y
            ln = math.hypot(vx, vy) or 1.0
            pt = Point(hit.x + margin_m * vx / ln, hit.y + margin_m * vy / ln)
            moved = True
        if not moved:
            break
    return float(pt.x), float(pt.y)


def is_inside_footprint(e: float, n: float, footprints: list[Polygon], margin_m: float = 1.2) -> bool:
    pt = Point(e, n)
    return any(_inside_or_close(pt, fp, margin_m) for fp in footprints)


def look_at_nearest_wall(e: float, n: float, footprints: list[Polygon]) -> tuple[float, float]:
    """Compass heading that faces the nearest façade, plus distance in metres."""
    if not footprints:
        return 0.0, float("inf")
    pt = Point(e, n)
    best_d = float("inf")
    best_h = 0.0
    for fp in footprints:
        if fp.is_empty:
            continue
        _, hit = nearest_points(pt, fp.boundary)
        de, dn = hit.x - e, hit.y - n
        dist = math.hypot(de, dn)
        if dist < best_d:
            best_d = dist
            # heading 0 looks +north; direction (de, dn)
            best_h = wrap_heading(math.degrees(math.atan2(de, dn)))
    return best_h, best_d


def pick_facade_shot(
    shots: list[dict[str, Any]], look_heading: float
) -> dict[str, Any] | None:
    """Horizon frame whose heading is most toward the wall."""
    if not shots:
        return None
    horiz = [s for s in shots if abs(float(s.get("pitch") or 0.0)) < 1.0]
    pool = horiz or shots
    return min(pool, key=lambda s: abs(heading_diff(float(s["heading"]), look_heading)))


def uncollapse_along_gps(
    poses: list[dict[str, Any]],
    *,
    min_frac: float = 0.6,
    min_abs_m: float = 5.0,
) -> int:
    """Spread panos that BAG/Canny stacked on top of each other.

    Street View GPS is usually right *along* the street (order and ~10 m
    spacing) and wrong sideways. If two cameras sat much closer than their
    GPS distance, restore that spacing along the GPS–GPS direction so a
    view further down the road cannot sit next to its neighbour.
    """
    moved = 0
    n = len(poses)
    for _ in range(10):
        changed = False
        for i in range(n):
            gi = poses[i].get("e_gps")
            if gi is None:
                continue
            for j in range(i + 1, n):
                gj = poses[j].get("e_gps")
                if gj is None:
                    continue
                ge = float(poses[i]["e_gps"]) - float(poses[j]["e_gps"])
                gn = float(poses[i]["n_gps"]) - float(poses[j]["n_gps"])
                gdist = math.hypot(ge, gn)
                if gdist < 4.0:
                    continue
                se = float(poses[i]["e"]) - float(poses[j]["e"])
                sn = float(poses[i]["n"]) - float(poses[j]["n"])
                sdist = math.hypot(se, sn)
                target = gdist
                if sdist >= min_frac * target and sdist >= min_abs_m:
                    continue
                ux, uy = ge / gdist, gn / gdist
                mid_e = 0.5 * (float(poses[i]["e"]) + float(poses[j]["e"]))
                mid_n = 0.5 * (float(poses[i]["n"]) + float(poses[j]["n"]))
                poses[i]["e"] = mid_e + 0.5 * target * ux
                poses[i]["n"] = mid_n + 0.5 * target * uy
                poses[j]["e"] = mid_e - 0.5 * target * ux
                poses[j]["n"] = mid_n - 0.5 * target * uy
                moved += 1
                changed = True
        if not changed:
            break
    return moved


def horizon_cardinals(
    pose: dict[str, Any], shots: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Four horizon views around a seated pano, for the live 3D pane."""
    travel = float(pose.get("travel_heading") or pose.get("heading") or 0.0)
    horiz = [s for s in shots if abs(float(s.get("pitch") or 0.0)) < 1.0]
    if not horiz:
        return [pose]
    out: list[dict[str, Any]] = []
    used: set[float] = set()
    for offset in (0.0, 90.0, 180.0, 270.0):
        target = wrap_heading(travel + offset)
        shot = min(horiz, key=lambda s: abs(heading_diff(float(s["heading"]), target)))
        hdg = float(shot["heading"])
        if any(abs(heading_diff(hdg, u)) < 8.0 for u in used):
            continue
        used.add(hdg)
        cam = dict(pose)
        cam["heading"] = hdg
        cam["pitch"] = float(shot.get("pitch") or 0.0)
        cam["fov"] = float(shot.get("fov") or pose.get("fov") or 90.0)
        cam["shot_path"] = shot.get("path") or shot.get("shot_path")
        cam["width"] = shot.get("width") or pose.get("width")
        cam["height"] = shot.get("height") or pose.get("height")
        out.append(cam)
    return out or [pose]
