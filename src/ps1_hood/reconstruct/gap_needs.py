"""Far-side SV fetch for ``recon/gap_needs.json`` (no sat_edge peels, no free-pose).

Read need ENU → probe outside along the outward normal → attach new panos at
fixed poses → sat-align. Prefer existing nearly-facing cams; fetch still runs
for better probes. FILM/lerp spur only when facing count stays < 2.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ps1_hood.geo import LocalFrame, wrap_heading
from ps1_hood.reconstruct.gap_fill import SAT_EDGE_CAM_DIST, SAT_EDGE_LOOK_AT

log = logging.getLogger(__name__)

DEFAULT_RADII_M = (8.0, 12.0, 16.0, 20.0)
FACING_DIST = SAT_EDGE_CAM_DIST  # (6, 35) m
FACING_LOOK_AT = SAT_EDGE_LOOK_AT  # 0.5
# Nearly-facing: same look-at / dist / same-side, but no graze gate (pack + steering).
PROBE_BBOX_PAD_M = 20.0


class GapNeedsMissing(FileNotFoundError):
    """``recon/gap_needs.json`` is required and absent."""


def need_id_of(need: dict[str, Any]) -> str:
    """Stable id: prefer ``id``, else ``edge_id``, else synthesised."""
    for key in ("id", "edge_id"):
        v = need.get(key)
        if v is not None and str(v).strip():
            return str(v).strip()
    roof = str(need.get("roof_id") or "need").strip() or "need"
    return f"{roof}_{float(need.get('e', 0)):.1f}_{float(need.get('n', 0)):.1f}"


def load_gap_needs(path: Path) -> dict[str, Any]:
    """Load ``gap_needs.json``. Fail loud if missing or empty needs."""
    path = Path(path)
    if not path.is_file():
        raise GapNeedsMissing(
            f"gap_needs missing: {path} — run facades with --gap-fill "
            "(sat-edge) first, or write recon/gap_needs.json by hand"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    needs = payload.get("needs")
    if not isinstance(needs, list) or not needs:
        raise RuntimeError(f"gap_needs has no needs entries: {path}")
    # Normalise id field
    for n in needs:
        if isinstance(n, dict) and not n.get("id"):
            n["id"] = need_id_of(n)
    payload["needs"] = needs
    return payload


def select_need(
    payload: dict[str, Any], need_id: str | None
) -> list[dict[str, Any]]:
    """All needs, or the single need matching ``need_id`` (fail loud)."""
    needs = list(payload.get("needs") or [])
    if need_id is None or not str(need_id).strip():
        return needs
    nid = str(need_id).strip()
    matched = [n for n in needs if need_id_of(n) == nid]
    if not matched:
        known = [need_id_of(n) for n in needs]
        raise RuntimeError(
            f"need-id {nid!r} not in gap_needs (have: {known})"
        )
    return matched


def local_frame_for_run(project_root: Path) -> LocalFrame:
    """LocalFrame for ENU↔ll: georef/scene origin if present, else bbox centre.

    Need ENU is sat-seated in the run's LocalFrame; georef.json holds SE(2)
    only — origin comes from scene/live/project bbox.
    """
    root = Path(project_root)
    for rel in ("recon/scene.json", "live.json", "align/georef.json"):
        p = root / rel
        if not p.is_file():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        origin = data.get("origin") if isinstance(data, dict) else None
        if isinstance(origin, dict) and "lat" in origin and "lon" in origin:
            return LocalFrame(float(origin["lat"]), float(origin["lon"]))
        # georef may stash lat0/lon0 under frame/origin extras
        for key in ("lat0", "lon0"):
            pass
        if data.get("lat0") is not None and data.get("lon0") is not None:
            return LocalFrame(float(data["lat0"]), float(data["lon0"]))
    # project.yaml bbox
    spec_path = root / "project.yaml"
    if spec_path.is_file():
        import yaml

        spec = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
        bbox = spec.get("bbox") or {}
        if all(k in bbox for k in ("south", "west", "north", "east")):
            from ps1_hood.geo import BBox

            return LocalFrame.from_bbox(
                BBox(
                    south=float(bbox["south"]),
                    west=float(bbox["west"]),
                    north=float(bbox["north"]),
                    east=float(bbox["east"]),
                )
            )
    raise RuntimeError(
        f"cannot build LocalFrame for {root}: need scene/live origin or project.yaml bbox"
    )


def enu_to_ll(
    frame: LocalFrame, e: float, n: float, u: float = 0.0
) -> tuple[float, float]:
    """ENU metres → WGS84 (lat, lon)."""
    lat, lon, _alt = frame.to_geodetic(float(e), float(n), float(u))
    return float(lat), float(lon)


def heading_unit(heading_deg: float) -> np.ndarray:
    """ENU unit vector for compass heading (0=+N, 90=+E)."""
    h = math.radians(float(heading_deg))
    return np.array([math.sin(h), math.cos(h)], dtype=np.float64)


def probe_enu_points(
    e: float,
    n: float,
    look_heading_deg: float,
    radii_m: tuple[float, ...] | list[float] = DEFAULT_RADII_M,
) -> list[tuple[float, float, float]]:
    """Probe XY outside the wall along the outward normal.

    Pack text said ``look_heading+180``; that lands inside the footprint.
    Cameras that *see* the wall sit outside along ``+look_heading`` (outward).
    Returns list of ``(e, n, radius_m)``.
    """
    fwd = heading_unit(look_heading_deg)  # outward
    out: list[tuple[float, float, float]] = []
    for r in radii_m:
        rr = float(r)
        if rr <= 0:
            continue
        pe = float(e) + fwd[0] * rr
        pn = float(n) + fwd[1] * rr
        out.append((pe, pn, rr))
    return out


def roof_center_xy(need: dict[str, Any]) -> np.ndarray:
    """Best-effort roof centre for same-side test."""
    if need.get("roof_c") is not None:
        rc = need["roof_c"]
        return np.array([float(rc[0]), float(rc[1])], dtype=np.float64)
    # Fall back: step 4 m inward from need mid along −n_out
    n_out = heading_unit(float(need.get("heading") or 0.0))
    return np.array(
        [float(need["e"]) - 4.0 * n_out[0], float(need["n"]) - 4.0 * n_out[1]],
        dtype=np.float64,
    )


def is_facing_cam(
    cam: dict[str, Any],
    need: dict[str, Any],
    *,
    dist_lo: float = FACING_DIST[0],
    dist_hi: float = FACING_DIST[1],
    look_at: float = FACING_LOOK_AT,
    require_same_side: bool = True,
) -> bool:
    """True if cam looks at need mid (pack facing filter, no graze gate)."""
    try:
        C = np.array([float(cam["e"]), float(cam["n"])], dtype=np.float64)
        M = np.array([float(need["e"]), float(need["n"])], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return False
    h = math.radians(float(cam.get("heading") or 0.0))
    fwd = np.array([math.sin(h), math.cos(h)], dtype=np.float64)
    v = M - C
    dist = float(np.linalg.norm(v))
    if dist < float(dist_lo) or dist > float(dist_hi):
        return False
    v_hat = v / dist
    if float(v_hat @ fwd) < float(look_at):
        return False
    if require_same_side:
        n_out = heading_unit(float(need.get("heading") or 0.0))
        roof_c = roof_center_xy(need)
        if float((C - roof_c) @ n_out) < 0.0:
            return False
    return True


def facing_score(cam: dict[str, Any], need: dict[str, Any]) -> float:
    """Higher = more frontal / nearer ideal ~12–16 m. For ranking existing."""
    C = np.array([float(cam["e"]), float(cam["n"])], dtype=np.float64)
    M = np.array([float(need["e"]), float(need["n"])], dtype=np.float64)
    h = math.radians(float(cam.get("heading") or 0.0))
    fwd = np.array([math.sin(h), math.cos(h)], dtype=np.float64)
    v = M - C
    dist = float(np.linalg.norm(v))
    if dist < 1e-6:
        return -1e9
    look = float((v / dist) @ fwd)
    # prefer ~14 m and high look-at
    return look * 2.0 - abs(dist - 14.0) / 20.0


def find_existing_facing(
    cameras: list[dict[str, Any]],
    need: dict[str, Any],
    *,
    dist_hi: float = FACING_DIST[1],
) -> list[dict[str, Any]]:
    """Prefer existing nearly-facing frames (unique pano_id, best heading)."""
    hits: list[dict[str, Any]] = []
    for cam in cameras:
        if is_facing_cam(cam, need, dist_hi=dist_hi):
            hits.append(cam)
    # one row per pano_id — keep best facing_score
    best: dict[str, dict[str, Any]] = {}
    for cam in hits:
        pid = str(cam.get("pano_id") or cam.get("id") or "")
        if not pid:
            continue
        prev = best.get(pid)
        if prev is None or facing_score(cam, need) > facing_score(prev, need):
            best[pid] = cam
    ranked = sorted(best.values(), key=lambda c: facing_score(c, need), reverse=True)
    return ranked


def existing_pano_ids(discover_panos: dict[str, Any] | list) -> set[str]:
    if isinstance(discover_panos, dict):
        panos = discover_panos.get("panos") or []
    else:
        panos = discover_panos
    out: set[str] = set()
    for p in panos:
        if not isinstance(p, dict):
            continue
        pid = p.get("pano_id")
        if pid and not str(pid).startswith("seed-"):
            out.add(str(pid))
    return out


def build_probe_seeds(
    need: dict[str, Any],
    frame: LocalFrame,
    *,
    radii_m: tuple[float, ...] | list[float] = DEFAULT_RADII_M,
    need_id: str | None = None,
) -> list[dict[str, Any]]:
    """Provisional google_web-style seeds at probe lat/lon facing the wall."""
    nid = need_id or need_id_of(need)
    look = float(need.get("heading") or 0.0)
    # travel so heading 0 (drive) looks at the wall (= look + 180)
    travel = wrap_heading(look + 180.0)
    seeds: list[dict[str, Any]] = []
    for i, (pe, pn, r) in enumerate(probe_enu_points(float(need["e"]), float(need["n"]), look, radii_m)):
        lat, lon = enu_to_ll(frame, pe, pn)
        seeds.append(
            {
                "pano_id": f"gap-{nid}-r{int(round(r)):02d}-{i:02d}",
                "lat": lat,
                "lon": lon,
                "travel_heading": travel,
                "provider": "gap_needs",
                "provisional": True,
                "source": "gap_needs",
                "need_id": nid,
                "probe_radius_m": float(r),
                "probe_e": pe,
                "probe_n": pn,
            }
        )
    return seeds


def lookup_new_panos(
    seeds: list[dict[str, Any]],
    *,
    existing_ids: set[str],
    lookup_fn: Callable[[float, float], dict[str, Any] | None] | None,
    max_panos: int = 6,
) -> list[dict[str, Any]]:
    """Resolve seeds via lookup_fn (or keep provisional). Skip known pano_ids."""
    found: list[dict[str, Any]] = []
    seen: set[str] = set(existing_ids)
    for seed in seeds:
        if len(found) >= int(max_panos):
            break
        lat, lon = float(seed["lat"]), float(seed["lon"])
        meta: dict[str, Any] | None = None
        if lookup_fn is not None:
            try:
                meta = lookup_fn(lat, lon)
            except Exception as exc:  # noqa: BLE001
                log.warning("gap_needs lookup failed at %.6f,%.6f: %s", lat, lon, exc)
                meta = None
        if meta and meta.get("pano_id"):
            pid = str(meta["pano_id"])
            if pid in seen:
                log.info("gap_needs: skip existing pano %s", pid)
                continue
            seen.add(pid)
            row = {
                **seed,
                "pano_id": pid,
                "lat": float(meta.get("lat") or lat),
                "lon": float(meta.get("lon") or lon),
                "provisional": False,
                "provider": meta.get("provider") or seed.get("provider") or "google",
                "date": meta.get("date"),
            }
            found.append(row)
        else:
            # Keep provisional seed for google_web capture path
            pid = str(seed["pano_id"])
            if pid in seen:
                continue
            seen.add(pid)
            found.append(dict(seed))
    return found


def filter_facing_panos_enu(
    panos: list[dict[str, Any]],
    need: dict[str, Any],
    frame: LocalFrame,
    *,
    dist_lo: float = FACING_DIST[0],
    dist_hi: float = FACING_DIST[1],
    look_at: float = FACING_LOOK_AT,
) -> list[dict[str, Any]]:
    """Keep panos whose GPS/ENU position can face the wall (travel toward need)."""
    kept: list[dict[str, Any]] = []
    for p in panos:
        lat = float(p["lat"])
        lon = float(p["lon"])
        e, n, _u = frame.to_enu(lat, lon, 0.0)
        travel = float(p.get("travel_heading") or wrap_heading(float(need["heading"]) + 180.0))
        cam = {"e": e, "n": n, "heading": travel, "pano_id": p.get("pano_id")}
        if is_facing_cam(
            cam, need, dist_lo=dist_lo, dist_hi=dist_hi, look_at=look_at
        ):
            kept.append(p)
    return kept


def merge_discover_panos(
    discover: dict[str, Any],
    new_panos: list[dict[str, Any]],
) -> dict[str, Any]:
    """Append new panos to discover payload (dedupe by pano_id)."""
    out = dict(discover)
    existing = list(out.get("panos") or [])
    have = {str(p.get("pano_id")) for p in existing if isinstance(p, dict)}
    added = 0
    for p in new_panos:
        pid = str(p.get("pano_id") or "")
        if not pid or pid in have:
            continue
        existing.append(p)
        have.add(pid)
        added += 1
    out["panos"] = existing
    out["gap_needs_added"] = int(out.get("gap_needs_added") or 0) + added
    return out


def merge_shots(
    old_shots: list[dict[str, Any]],
    new_shots: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Union shots by (pano_id, heading, pitch)."""
    key = lambda s: (
        str(s.get("pano_id")),
        round(float(s.get("heading") or 0.0), 1),
        round(float(s.get("pitch") or 0.0), 1),
    )
    seen = {key(s) for s in old_shots}
    out = list(old_shots)
    for s in new_shots:
        k = key(s)
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


def unique_facing_pano_count(facing_cams: list[dict[str, Any]]) -> int:
    return len({str(c.get("pano_id")) for c in facing_cams if c.get("pano_id")})


def summarize_need(
    need: dict[str, Any],
    frame: LocalFrame,
    cameras: list[dict[str, Any]],
) -> dict[str, Any]:
    """Show helpers: ll, probes, existing facing."""
    nid = need_id_of(need)
    lat, lon = enu_to_ll(frame, float(need["e"]), float(need["n"]))
    probes = probe_enu_points(float(need["e"]), float(need["n"]), float(need["heading"]))
    probe_ll = [
        {
            "radius_m": r,
            "e": pe,
            "n": pn,
            "lat": enu_to_ll(frame, pe, pn)[0],
            "lon": enu_to_ll(frame, pe, pn)[1],
        }
        for pe, pn, r in probes
    ]
    existing = find_existing_facing(cameras, need)
    return {
        "id": nid,
        "e": float(need["e"]),
        "n": float(need["n"]),
        "heading": float(need.get("heading") or 0.0),
        "reason": need.get("reason"),
        "roof_id": need.get("roof_id"),
        "lat": lat,
        "lon": lon,
        "probes": probe_ll,
        "existing_facing": [
            {
                "pano_id": c.get("pano_id"),
                "heading": c.get("heading"),
                "e": c.get("e"),
                "n": c.get("n"),
                "dist_m": float(
                    math.hypot(
                        float(need["e"]) - float(c["e"]),
                        float(need["n"]) - float(c["n"]),
                    )
                ),
                "score": facing_score(c, need),
            }
            for c in existing
        ],
        "n_existing_facing": unique_facing_pano_count(existing),
    }


def capture_new_panos_only(
    project: Any,
    settings: Any,
    new_panos: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run existing capture stack on *new* panos only; merge into raw/shots.json."""
    from ps1_hood.capture.google_static import capture_panos as capture_static
    from ps1_hood.capture.google_web import capture_panos as capture_web
    from ps1_hood.capture.google_js import capture_panos as capture_js
    from ps1_hood.capture.mapillary import capture_panos as capture_mapillary

    if not new_panos:
        return []
    spec = project.load_spec()
    dest = project.raw_dir
    if spec.source == "google_web":
        shots = capture_web(new_panos, spec, settings, dest)
    elif spec.source == "google_static":
        shots = capture_static(new_panos, spec, settings, dest)
    elif spec.source == "google_js":
        shots = capture_js(new_panos, spec, settings, dest, project.root / "tmp")
    elif spec.source == "mapillary":
        shots = capture_mapillary(new_panos, spec, dest, settings.mapillary_token)
    else:
        raise RuntimeError(f"unknown source {spec.source}")
    shots_path = project.raw_dir / "shots.json"
    old: list[dict[str, Any]] = []
    if shots_path.is_file():
        old = project.read_json(shots_path)
    merged = merge_shots(old, shots)
    project.write_json(shots_path, merged)
    return shots


def crop_new_shots_only(project: Any, settings: Any, new_shots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Crop only newly captured shots; merge into cropped/shots.json."""
    from ps1_hood.capture.crop import crop_shots

    if not new_shots:
        return []
    spec = project.load_spec()
    js_mode = spec.source in {"google_js", "google_web"}
    cropped_new = crop_shots(
        new_shots,
        project.cropped_dir,
        top_frac=settings.crop_top_frac if js_mode else 0.0,
        bottom_frac=settings.crop_bottom_frac if js_mode else 0.0,
        auto=js_mode,
    )
    path = project.cropped_dir / "shots.json"
    old: list[dict[str, Any]] = []
    if path.is_file():
        old = project.read_json(path)
    merged = merge_shots(old, cropped_new)
    project.write_json(path, merged)
    return cropped_new


def interpolate_facing_spur(
    project: Any,
    facing_poses: list[dict[str, Any]],
    *,
    steps: int | None = None,
) -> list[dict[str, Any]]:
    """PR-B ENU ``lerp_pose`` midframes between facing/new poses only (no free-pose).

    Writes spur frames under ``interp/gap_needs_spur/`` and merges into
    ``interp/frames.json`` when present. Skips if <2 poses.
    """
    from ps1_hood.interpolate.flow import interpolate_pair
    from ps1_hood.interpolate.sequence import enu_baseline_m, lerp_pose, MIN_LERP_BASELINE_M

    if len(facing_poses) < 2:
        return []
    spec = project.load_spec()
    n_steps = int(steps if steps is not None else getattr(spec, "interp_steps", 3) or 3)
    # Order by score proxy: south-west → greedy nearest
    poses = sorted(
        facing_poses,
        key=lambda p: (float(p.get("n") or 0.0), float(p.get("e") or 0.0)),
    )
    spur_dir = project.interp_dir / "gap_needs_spur"
    spur_dir.mkdir(parents=True, exist_ok=True)
    frames: list[dict[str, Any]] = []
    for a, b in zip(poses, poses[1:]):
        if enu_baseline_m(a, b) < MIN_LERP_BASELINE_M:
            continue
        path_a = a.get("shot_path") or a.get("path")
        path_b = b.get("shot_path") or b.get("path")
        imgs: list[Any] = []
        if path_a and path_b and Path(path_a).is_file() and Path(path_b).is_file():
            try:
                import cv2

                ia = cv2.imread(str(path_a))
                ib = cv2.imread(str(path_b))
                if ia is not None and ib is not None:
                    imgs = interpolate_pair(ia, ib, n_steps)
            except Exception as exc:  # noqa: BLE001
                log.warning("gap_needs spur FILM failed (%s) — pose-only lerp", exc)
                imgs = []
        for i in range(1, n_steps + 1):
            t = i / (n_steps + 1)
            pose = lerp_pose(a, b, t)
            pose["interpolated"] = True
            pose["source"] = "gap_needs_spur"
            name = (
                f"spur_{a.get('pano_id', 'a')}_{b.get('pano_id', 'b')}"
                f"_t{i:02d}.jpg"
            )
            out_path = spur_dir / name
            if imgs and i - 1 < len(imgs):
                try:
                    import cv2

                    cv2.imwrite(str(out_path), imgs[i - 1])
                    pose["path"] = str(out_path)
                    pose["shot_path"] = str(out_path)
                except Exception:  # noqa: BLE001
                    pass
            frames.append(pose)
    if not frames:
        return []
    frames_path = project.interp_dir / "frames.json"
    existing: list[dict[str, Any]] = []
    if frames_path.is_file():
        existing = project.read_json(frames_path)
    # drop prior spur frames, keep others
    kept = [f for f in existing if f.get("source") != "gap_needs_spur"]
    project.write_json(frames_path, kept + frames)
    project.write_json(spur_dir / "frames.json", frames)
    log.info("gap_needs: wrote %s ENU-lerp spur midframes", len(frames))
    return frames
