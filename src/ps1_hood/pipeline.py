"""Run the reconstruction stages in order."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from ps1_hood.align.bag_edges import snap_camera_to_bag
from ps1_hood.align.georef import (
    ALIGN_PRIORS,
    clip_recon_clouds_to_ortho,
    fit_se2,
    georef_payload,
    refresh_scene_cameras,
    seat_recon_artefacts,
    summarize_se2,
)
from ps1_hood.align.pose_graph import (
    explode_orbit_cameras,
    initial_poses,
    refine_poses,
    write_debug_overlay,
)
from ps1_hood.align.seat import (
    building_footprints,
    horizon_cardinals,
    look_at_nearest_wall,
    pick_facade_shot,
    push_out_of_footprints,
    uncollapse_along_gps,
)
from ps1_hood.capture.bag import buildings_enu, fetch_bag, live_buildings
from ps1_hood.capture.crop import crop_shots
from ps1_hood.capture.discover import cap_panos, discover
from ps1_hood.capture.google_js import capture_panos as capture_js
from ps1_hood.capture.google_static import capture_panos as capture_static
from ps1_hood.capture.google_web import capture_panos as capture_web
from ps1_hood.capture.mapillary import capture_panos as capture_mapillary
from ps1_hood.capture.satellite import Ortho, fetch_satellite
from ps1_hood.config import Settings
from ps1_hood.geo import BBox, LocalFrame
from ps1_hood.interpolate.flow import interpolate_track
from ps1_hood.interpolate.sequence import assert_interp_frame_poses, select_densify_frames, select_posed_sparse_frames
from ps1_hood.interpolate.video import write_video
from ps1_hood.overpass import fetch_roads
from ps1_hood.progress import emit as _emit
from ps1_hood.project import Project
from ps1_hood.reconstruct.colmap import (
    COLMAP_HINT,
    export_colmap_images,
    frames_have_known_poses,
    run_colmap,
    run_colmap_posed,
)
from ps1_hood.reconstruct.unproject import triangulate_frames, triangulate_sift_frames
from ps1_hood.reconstruct.export import scene_payload, write_scene
from ps1_hood.reconstruct.facades import extract_facades
from ps1_hood.reconstruct.keyframes import load_keyframes

log = logging.getLogger(__name__)

Progress = Any

# Google snaps panos onto nearby streets; keep the kerb, drop the next block.
PANO_BBOX_SLOP_M = 25.0


def _shots_in_bbox(
    shots: list[dict[str, Any]], bbox: BBox, slop_m: float = PANO_BBOX_SLOP_M
) -> tuple[list[dict[str, Any]], int]:
    """Keep frames whose snapped GPS is inside the drawn block (+ slop)."""
    keep_ids = {
        s["pano_id"]
        for s in shots
        if s.get("lat") is not None and bbox.contains_m(float(s["lat"]), float(s["lon"]), slop_m)
    }
    kept = [s for s in shots if s["pano_id"] in keep_ids]
    dropped_panos = len({s["pano_id"] for s in shots}) - len(keep_ids)
    if not kept:
        log.warning("every pano is outside the bbox; keeping all so align can still run")
        return shots, 0
    return kept, dropped_panos


def _bag_query_bbox(spec_bbox: BBox, shots: list[dict[str, Any]] | None) -> BBox:
    """Query 3DBAG for the drawn block plus in-block camera GPS."""
    box = spec_bbox
    for s in shots or []:
        if s.get("lat") is None:
            continue
        lat, lon = float(s["lat"]), float(s["lon"])
        if spec_bbox.contains_m(lat, lon, slop_m=PANO_BBOX_SLOP_M):
            box = box.union_point(lat, lon)
    return box.padded(8.0)


def _rel_image(project: Project, path: str | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    try:
        return str(p.resolve().relative_to(project.root.resolve()))
    except ValueError:
        return str(p)


def _write_live(
    project: Project,
    frame: LocalFrame,
    buildings: list[dict[str, Any]] | None,
    cameras: list[dict[str, Any]] | None,
) -> None:
    payload = {
        "origin": {"lat": frame.lat0, "lon": frame.lon0},
        "buildings": live_buildings(buildings or []),
        "cameras": [
            {
                "pano_id": c.get("pano_id"),
                "e": c.get("e"),
                "n": c.get("n"),
                "u": c.get("u"),
                "heading": c.get("heading"),
                "pitch": c.get("pitch") or 0,
                "fov": c.get("fov") or 90,
                "image": _rel_image(project, c.get("shot_path") or c.get("path")),
                "residual_m": c.get("residual_m"),
            }
            for c in (cameras or [])
            if c.get("e") is not None
        ],
    }
    project.write_json(project.live_path, payload)


def stage_discover(project: Project, settings: Settings, progress: Progress | None = None) -> dict:
    spec = project.load_spec()
    _emit(progress, "discover", f"sampling {spec.source} coverage in bbox")
    payload = discover(spec.bbox, spec.source, settings, spec.spacing_m)
    before = len(payload["panos"])
    capped, max_n = cap_panos(payload["panos"], getattr(spec, "max_panos", None))
    payload["panos"] = capped
    if max_n is not None:
        log.info("discover: capped to %s / max %s", len(capped), max_n)
        if before > len(capped):
            _emit(
                progress,
                "discover",
                f"capped to {len(capped)} / max {max_n} (from {before})",
                queued=len(capped),
                captured=0,
                skipped=0,
            )
    project.write_json(project.discover_dir / "panos.json", payload)
    osm = fetch_roads(spec.bbox)
    project.write_json(project.osm_dir / "roads.json", osm)
    n = len(payload["panos"])
    _emit(progress, "discover", f"{n} panos queued", queued=n, captured=0, skipped=0)
    return payload


def stage_capture(project: Project, settings: Settings, progress: Progress | None = None) -> list:
    spec = project.load_spec()
    discovered = project.read_json(project.discover_dir / "panos.json")
    panos = discovered["panos"]
    _emit(progress, "capture", f"{len(panos)} panos via {spec.source}", queued=len(panos), captured=0)

    def on_event(ev: dict[str, Any]) -> None:
        _emit(
            progress,
            "capture",
            ev.get("message") or f"{ev.get('captured', 0)}/{ev.get('queued', 0)}",
            queued=ev.get("queued", len(panos)),
            captured=ev.get("captured", 0),
            skipped=ev.get("skipped", 0),
            current=ev.get("current", ""),
        )

    if spec.source == "google_web":
        shots = capture_web(panos, spec, settings, project.raw_dir, on_event=on_event)
    elif spec.source == "google_static":
        shots = capture_static(panos, spec, settings, project.raw_dir)
    elif spec.source == "google_js":
        shots = capture_js(panos, spec, settings, project.raw_dir, project.root / "tmp")
    elif spec.source == "mapillary":
        shots = capture_mapillary(panos, spec, project.raw_dir, settings.mapillary_token)
    else:
        raise RuntimeError(f"unknown source {spec.source}")
    project.write_json(project.raw_dir / "shots.json", shots)
    n_pano = len({s["pano_id"] for s in shots})
    _emit(progress, "capture", f"wrote {len(shots)} frames from {n_pano} panos", captured=n_pano)
    return shots


def stage_crop(project: Project, settings: Settings, progress: Progress | None = None) -> list:
    spec = project.load_spec()
    shots = project.read_json(project.raw_dir / "shots.json")
    # Static API images already omit the Maps chrome; only the JS screenshot
    # path needs the original "crop the UI" pass.
    js_mode = spec.source in {"google_js", "google_web"}
    _emit(
        progress,
        "crop",
        f"stripping UI from {len(shots)} screengrabs" if js_mode else "copying frames",
    )
    cropped = crop_shots(
        shots,
        project.cropped_dir,
        top_frac=settings.crop_top_frac if js_mode else 0.0,
        bottom_frac=settings.crop_bottom_frac if js_mode else 0.0,
        auto=js_mode,
    )
    project.write_json(project.cropped_dir / "shots.json", cropped)
    return cropped


def stage_satellite(project: Project, progress: Progress | None = None) -> dict:
    spec = project.load_spec()
    _emit(progress, "satellite", "fetching orthoimagery")
    meta = fetch_satellite(spec.bbox, project.satellite_dir)
    project.write_json(project.satellite_dir / "ortho.json", meta)
    return meta


def stage_bag(project: Project, progress: Progress | None = None) -> list:
    spec = project.load_spec()
    frame = LocalFrame.from_bbox(spec.bbox)
    shots: list[dict[str, Any]] = []
    shots_path = project.cropped_dir / "shots.json"
    if shots_path.is_file():
        shots = project.read_json(shots_path)
    query_bbox = _bag_query_bbox(spec.bbox, shots)
    _emit(progress, "bag", "querying 3DBAG GeoPackage dump (bbox only, no full unzip)")
    try:
        raw = fetch_bag(query_bbox, dest=project.bag_dir)
        buildings = buildings_enu(raw, frame, clip=spec.bbox)
    except Exception as exc:  # noqa: BLE001
        log.warning("3DBAG fetch failed (%s) — edge snap will be skipped", exc)
        buildings = []
        raw = {"features": [], "error": str(exc)}
    project.write_json(project.bag_dir / "raw.json", {"count": raw.get("count"), "error": raw.get("error")})
    project.write_json(
        project.bag_dir / "buildings.json",
        [
            {
                "id": b["id"],
                "dak_type": b.get("dak_type"),
                "bbox": b["bbox"],
                "n_faces": len(b["faces"]),
                "vertices": b["vertices"],
                "faces": b["faces"],
                "edges": b["edges"],
            }
            for b in buildings
        ],
    )
    _write_live(project, frame, buildings, [])
    _emit(progress, "bag", f"{len(buildings)} buildings", bag_buildings=len(buildings))
    return buildings


def stage_align(
    project: Project,
    progress: Progress | None = None,
    *,
    align_prior: str | None = None,
    sat_edge_weight: float | None = None,
    cloud_clip_sat: bool | None = None,
    sat_cloud_margin_m: float | None = None,
) -> list:
    """GPS/OSM prior → sat or BAG absolute seat.

    Default ``align_prior=sat``: Ortho edge+NCC fuse + feature SE(2) bundle;
    **never** ``snap_camera_to_bag``; skip footprint push that fights sat.
    Cloud clip is **opt-in** (``cloud_clip_sat=False`` by default): prefer the
    unclipped product cloud; use ``--cloud-clip-sat`` only as a floater tool.
    Writes ``align/georef.json`` and seats existing recon artefacts with one SE(2).
    ``align_prior=bag`` keeps legacy BAG-first behaviour for debug.
    """
    spec = project.load_spec()
    prior = (align_prior or getattr(spec, "align_prior", None) or "sat").strip().lower()
    if prior not in ALIGN_PRIORS:
        raise ValueError(f"align_prior must be one of {ALIGN_PRIORS}, got {prior!r}")
    if align_prior is not None and getattr(spec, "align_prior", None) != prior:
        spec.align_prior = prior
        project.save_spec(spec)

    w_edge = float(0.65 if sat_edge_weight is None else sat_edge_weight)
    w_edge = min(1.0, max(0.0, w_edge))
    w_ncc = 1.0 - w_edge
    do_clip = False if cloud_clip_sat is None else bool(cloud_clip_sat)
    clip_margin = float(2.0 if sat_cloud_margin_m is None else sat_cloud_margin_m)

    shots = project.read_json(project.cropped_dir / "shots.json")
    shots, dropped = _shots_in_bbox(shots, spec.bbox)
    if dropped:
        log.info("dropped %s panos whose Google GPS sits outside the bbox", dropped)
        _emit(progress, "align", f"dropped {dropped} panos outside the bbox")
    osm = project.read_json(project.osm_dir / "roads.json")
    sat = project.read_json(project.satellite_dir / "ortho.json")
    frame = LocalFrame.from_bbox(spec.bbox)
    ortho = Ortho.load(sat, frame)
    buildings = (
        project.read_json(project.bag_dir / "buildings.json")
        if (project.bag_dir / "buildings.json").is_file()
        else []
    )
    footprints = building_footprints(buildings) if buildings else []
    shots_by: dict[str, list[dict[str, Any]]] = {}
    for shot in shots:
        shots_by.setdefault(shot["pano_id"], []).append(shot)

    old_poses = None
    poses_path = project.align_dir / "poses.json"
    if poses_path.is_file():
        try:
            old_poses = project.read_json(poses_path)
        except Exception:  # noqa: BLE001
            old_poses = None

    if prior == "sat":
        _emit(progress, "align", f"sat prior: Ortho edge+NCC (w_edge={w_edge:.2f}) + feature SE(2); BAG snap off")
        use_satellite = True
        use_features = True
    else:
        _emit(progress, "align", "bag prior: GPS/OSM then 3DBAG façade snap (debug)")
        use_satellite = not buildings
        use_features = not bool(buildings)

    poses = initial_poses(shots, spec, frame, osm)
    poses_before = [dict(p) for p in poses]
    poses = refine_poses(
        poses,
        spec,
        frame,
        ortho,
        use_satellite=use_satellite,
        use_features=use_features,
        w_ncc=w_ncc,
        w_edge=w_edge,
    )

    if prior == "bag":
        for pose in poses:
            pose["e"], pose["n"] = push_out_of_footprints(pose["e"], pose["n"], footprints)

    for i, pose in enumerate(poses):
        photo = None
        look_h, _wall_d = look_at_nearest_wall(pose["e"], pose["n"], footprints)
        facade = pick_facade_shot(shots_by.get(pose["pano_id"]) or [], look_h)
        photo_path = (facade or {}).get("path") or pose["shot_path"]
        photo_heading = float((facade or {}).get("heading") or pose["heading"])
        photo_pitch = float((facade or {}).get("pitch") or pose.get("pitch") or 0.0)
        try:
            import cv2

            photo = cv2.imread(photo_path, cv2.IMREAD_COLOR)
        except Exception:
            photo = None
        # BAG snap only under bag prior (sacred: sat ortho = absolute XY)
        if prior == "bag" and photo is not None and buildings:
            hit = snap_camera_to_bag(
                photo,
                buildings,
                e=pose["e"],
                n=pose["n"],
                u=pose["u"],
                heading=photo_heading,
                pitch=photo_pitch,
                fov=float((facade or {}).get("fov") or pose.get("fov") or spec.fov_deg),
                travel_heading=float(pose.get("travel_heading") or pose["heading"]),
                footprints=footprints,
            )
            pose["e"] = hit["e"]
            pose["n"] = hit["n"]
            pose["u"] = hit["u"]
            pose["e"], pose["n"] = push_out_of_footprints(pose["e"], pose["n"], footprints)
            pose["residual_m"] = hit["residual_m"]
            pose["bag_score"] = hit["score"]
            pose["bag_snapped"] = hit["snapped"]
            lat, lon, _ = frame.to_geodetic(pose["e"], pose["n"], 0.0)
            pose["lat"] = lat
            pose["lon"] = lon
        live_cams = []
        for p in poses[: i + 1]:
            live_cams.extend(horizon_cardinals(p, shots_by.get(p["pano_id"]) or []))
        _emit(
            progress,
            "align",
            f"{'snap' if prior == 'bag' else 'seat'} {i + 1}/{len(poses)}  residual={pose.get('residual_m')} m",
            aligned=i + 1,
            queued=len(poses),
        )
        _write_live(project, frame, buildings, live_cams)

    n_spread = uncollapse_along_gps(poses)
    if n_spread:
        log.info("restored GPS spacing for %s collapsed pano pairs", n_spread)
        _emit(progress, "align", f"spread {n_spread} stacked panos back along the street")
    for pose in poses:
        if prior == "bag":
            pose["e"], pose["n"] = push_out_of_footprints(pose["e"], pose["n"], footprints)
        lat, lon, _ = frame.to_geodetic(pose["e"], pose["n"], 0.0)
        pose["lat"] = lat
        pose["lon"] = lon

    T_sat = summarize_se2(poses_before, poses)
    scores = [float(p["sat_score"]) for p in poses if p.get("sat_score") is not None]
    sat_mean = sum(scores) / len(scores) if scores else None
    ncc_vals = [float(p["sat_ncc"]) for p in poses if p.get("sat_ncc") is not None and float(p["sat_ncc"]) > -0.5]
    edge_vals = [float(p["sat_edge"]) for p in poses if p.get("sat_edge") is not None and float(p["sat_edge"]) > -0.5]
    sat_ncc_mean = sum(ncc_vals) / len(ncc_vals) if ncc_vals else None
    sat_edge_mean = sum(edge_vals) / len(edge_vals) if edge_vals else None
    snapped_n = sum(1 for p in poses if p.get("bag_snapped"))
    georef = georef_payload(
        prior=prior,
        T_sat=T_sat,
        sat_score_mean=sat_mean,
        bag_snapped=snapped_n,
        n_poses=len(poses),
        extra={
            "sat_ncc_mean": sat_ncc_mean,
            "sat_edge_mean": sat_edge_mean,
            "w_ncc": w_ncc,
            "w_edge": w_edge,
        },
    )

    # Seat existing product artefacts with one SE(2) (old poses → new) so Studio
    # agrees without a full densify re-run.
    if prior == "sat" and old_poses and len(old_poses) == len(poses):
        T_applied = fit_se2(old_poses, poses)
        stats = seat_recon_artefacts(project.recon_dir, T_applied)
        georef["T_applied"] = {
            "tx_m": T_applied["tx_m"],
            "ty_m": T_applied["ty_m"],
            "yaw_deg": T_applied["yaw_deg"],
            "s": T_applied.get("s", 1.0),
            "pivot_e": T_applied.get("pivot_e"),
            "pivot_n": T_applied.get("pivot_n"),
        }
        georef["artefacts_seated"] = stats
        if stats:
            log.info("applied product SE(2) to recon artefacts: %s", stats)
            _emit(progress, "align", f"seated recon artefacts with T_applied {stats}")
    elif prior == "sat" and (project.recon_dir / "cloud.ply").is_file():
        # No prior poses.json — apply this-run T_sat (init→sat) as best effort.
        stats = seat_recon_artefacts(project.recon_dir, T_sat)
        georef["artefacts_seated"] = stats
        if stats:
            log.info("applied T_sat to recon artefacts (no prior poses): %s", stats)

    # Opt-in floater gate: drop MA pts outside Ortho ENU ± margin (façades untouched)
    if prior == "sat" and do_clip and (project.recon_dir / "cloud.ply").is_file():
        clip_stats = clip_recon_clouds_to_ortho(
            project.recon_dir,
            ortho.sw,
            ortho.sh,
            ortho.ee,
            ortho.nn,
            margin_m=clip_margin,
        )
        georef["cloud_clipped"] = clip_stats
        log.info(
            "sat cloud clip kept=%s dropped=%s margin=%.1fm",
            clip_stats.get("kept"),
            clip_stats.get("dropped"),
            clip_margin,
        )
        _emit(
            progress,
            "align",
            f"clipped MA cloud: dropped {clip_stats.get('dropped', 0)} outside Ortho ±{clip_margin:.0f}m",
        )

    cameras = explode_orbit_cameras(poses, shots)
    project.write_json(project.align_dir / "poses.json", poses)
    project.write_json(project.align_dir / "cameras.json", cameras)
    project.write_json(project.align_dir / "georef.json", georef)
    write_debug_overlay(poses, ortho, project.align_dir / "overlay.jpg")
    refresh_scene_cameras(project.recon_dir / "scene.json", poses, georef)
    live_cams = []
    for pose in poses:
        live_cams.extend(horizon_cardinals(pose, shots_by.get(pose["pano_id"]) or []))
    _write_live(project, frame, buildings, live_cams)
    if poses and buildings and prior == "bag":
        ce = sum(p["e"] for p in poses) / len(poses)
        cn = sum(p["n"] for p in poses) / len(poses)
        be = sum(0.5 * (b["bbox"][0] + b["bbox"][3]) for b in buildings) / len(buildings)
        bn = sum(0.5 * (b["bbox"][1] + b["bbox"][4]) for b in buildings) / len(buildings)
        offset = ((ce - be) ** 2 + (cn - bn) ** 2) ** 0.5
        log.info(
            "camera vs 3DBAG centroid %.1f m  (%s/%s panos edge-snapped)",
            offset,
            snapped_n,
            len(poses),
        )
        _emit(
            progress,
            "align",
            f"SV vs 3DBAG {offset:.1f} m apart, {snapped_n}/{len(poses)} snapped",
        )
    if prior == "sat":
        log.info(
            "sat georef T_sat tx=%.2f ty=%.2f yaw=%.2f°  sat_score_mean=%s ncc=%s edge=%s",
            T_sat["tx_m"],
            T_sat["ty_m"],
            T_sat["yaw_deg"],
            f"{sat_mean:.3f}" if sat_mean is not None else "n/a",
            f"{sat_ncc_mean:.3f}" if sat_ncc_mean is not None else "n/a",
            f"{sat_edge_mean:.3f}" if sat_edge_mean is not None else "n/a",
        )
    _emit(
        progress,
        "align",
        f"refined {len(poses)} panos / {len(cameras)} views (prior={prior})",
        aligned=len(poses),
    )
    return poses



def stage_interpolate(project: Project, progress: Progress | None = None) -> list:
    spec = project.load_spec()
    poses = project.read_json(project.align_dir / "poses.json")
    _emit(progress, "interpolate", f"{spec.interp_backend}  steps={spec.interp_steps}")
    if spec.interp_backend != "flow":
        _emit(
            progress,
            "interpolate",
            f"{spec.interp_backend} is not bundled; using optical-flow fallback. "
            "See README for FILM / RIFE.",
        )
    frames = interpolate_track(poses, project.interp_dir / "frames", spec.interp_steps)
    assert_interp_frame_poses(frames)  # full FILM rate: ENU + fov/width/height
    project.write_json(project.interp_dir / "frames.json", frames)
    video = write_video(frames, project.interp_dir / "drive.mp4")
    if video:
        _emit(progress, "interpolate", f"drive video → {video}")
    return frames


def stage_reconstruct(project: Project, progress: Progress | None = None) -> dict[str, Any]:
    spec = project.load_spec()
    poses = project.read_json(project.align_dir / "poses.json")
    sat = project.read_json(project.satellite_dir / "ortho.json")
    frame = LocalFrame.from_bbox(spec.bbox)
    keyframes = load_keyframes(project)
    interp_path = project.interp_dir / "frames.json"
    interp_frames: list[dict[str, Any]] = (
        project.read_json(interp_path) if interp_path.is_file() else []
    )
    if interp_frames:
        # Fail loud before any densify/recon consumes FILM midframes.
        assert_interp_frame_poses(interp_frames)

    # Real aligned shots for SfM / MVS; DIS midframes only as flow fallback.
    backend = spec.recon_backend
    # Prefer known-pose triangulator when poses exist — never retry mapper on SV.
    if backend == "colmap" and frames_have_known_poses(
        keyframes if len(keyframes) >= 2 else (interp_frames if interp_frames else [])
    ):
        backend = "colmap_posed"
        log.info("colmap → colmap_posed (known ENU poses present; skip mapper)")
    matcher = getattr(spec, "recon_matcher", "sift") or "sift"
    if backend == "mast3r":
        # Product path: MASt3R = matcher only → colmap_posed with ENU lock.
        backend = "colmap_posed"
        matcher = "mast3r"
        log.info("mast3r → colmap_posed matcher=mast3r (ENU locked; no free-pose)")
    if backend in {"colmap", "colmap_posed", "sift", "mast3r", "export"}:
        frames = keyframes
        source = "keyframes"
        # Grow posed sparse/tracks: orbit keyframes + subsampled FILM midframes
        # that already carry lerp_pose ENU (assert on full FILM rate first).
        if backend in {"colmap_posed", "sift", "export"} and interp_frames:
            frames = select_posed_sparse_frames(keyframes, interp_frames)
            n_mids = sum(1 for f in frames if f.get("interpolated"))
            source = "keyframes+midframes" if n_mids else "keyframes"
            log.info(
                "posed sparse frames: %s keyframes + %s strided midframes → %s views",
                len(keyframes),
                n_mids,
                len(frames),
            )
        if len(frames) < 2:
            raise RuntimeError(
                f"recon backend {backend} needs ≥2 aligned keyframes "
                f"(align/cameras.json with existing shot_path); got {len(frames)}"
            )
    else:
        # flow triangulation
        if len(keyframes) >= 2:
            frames = keyframes
            source = "keyframes"
        elif len(interp_frames) >= 2:
            # Assert every midframe pose; densify/recon only every Nth midframe + panos.
            frames = select_densify_frames(interp_frames)
            source = "interp"
            log.warning(
                "only %s keyframes — falling back to %s interpolated frames "
                "(densify subset %s after stride)",
                len(keyframes),
                len(interp_frames),
                len(frames),
            )
        else:
            raise RuntimeError(
                "need ≥2 keyframes (or interp frames) to triangulate; "
                f"got keyframes={len(keyframes)} interp={len(interp_frames)}"
            )

    log.info("reconstruct: using %s (%s views) backend=%s", source, len(frames), backend)
    _emit(progress, "reconstruct", f"{backend} from {source} ({len(frames)} views)")

    cloud = None

    def _facade_pass(cloud_ply: Path, meta: dict) -> None:
        try:
            # Dual-source: A xyz stays on the triangulated flow cloud;
            # MA peels use MapAnything ENU when present. Do not swap A onto MA.
            ma_ply = None
            for cand in (
                project.root / "mapanything" / "cloud.ply",
                project.recon_dir / "cloud_mapanything.ply",
            ):
                if cand.is_file():
                    ma_ply = cand
                    break
            fac = extract_facades(
                ma_ply or cloud_ply,
                project.recon_dir / "facades.obj",
                frames=frames,
                satellite=sat,
                local_frame=frame,
                planarize=None,  # auto when dense ≳50k
                a_ply_path=cloud_ply,
                a_source="flow",
            )
            meta["facades"] = fac
            if int(fac.get("planes") or 0) == 0:
                log.error(
                    "facade pass: 0 photo-consistent planes (source=%s) — "
                    "Studio shows ground only, not RANSAC invention",
                    fac.get("source"),
                )
            else:
                log.info(
                    "facade pass: %s photo-consistent planes (%s textured, mean ZNCC=%s)",
                    fac.get("planes"),
                    fac.get("textured"),
                    fac.get("mean_zncc"),
                )
        except RuntimeError as exc:
            log.error("facade pass failed loud: %s", exc)

    if backend == "export":
        export_colmap_images(frames, project.recon_dir / "colmap")
        _emit(progress, "reconstruct", "exported keyframes for an external reconstructor")
        _emit(progress, "reconstruct", COLMAP_HINT.strip())
    elif backend == "colmap_posed":
        _emit(
            progress,
            "reconstruct",
            f"COLMAP known-pose triangulator (cross-pano, matcher={matcher})",
        )
        ws, names = export_colmap_images(frames, project.recon_dir / "colmap")
        ply_photo = project.recon_dir / "cloud_photo.ply"
        try:
            ply_path = run_colmap_posed(
                ws, frames, names, ply_out=ply_photo, matcher=matcher
            )
            cloud_ply = project.recon_dir / "cloud.ply"
            shutil.copy2(ply_path, cloud_ply)
            n_pts = 0
            for line in cloud_ply.read_text(encoding="ascii", errors="ignore").splitlines():
                if line.startswith("element vertex"):
                    n_pts = int(line.split()[-1])
                    break
            cloud = {
                "path": str(cloud_ply),
                "source": "colmap_posed",
                "matcher": matcher,
                "photo_ply": str(ply_path),
                "points": n_pts,
                "pose_lock": True,
            }
            _facade_pass(cloud_ply, cloud)
        except Exception as exc:  # noqa: BLE001
            # Mast3r missing/GPU: fail loud (no silent SIFT swap) — user asked for mast3r.
            if matcher == "mast3r":
                raise
            log.warning("colmap_posed failed (%s) — falling back to OpenCV SIFT stereo", exc)
            _emit(progress, "reconstruct", f"posed COLMAP failed; OpenCV SIFT fallback: {exc}")
            cloud = triangulate_sift_frames(frames, project.recon_dir / "cloud.ply")
            sift_copy = project.recon_dir / "cloud_sift.ply"
            shutil.copy2(project.recon_dir / "cloud.ply", sift_copy)
            cloud["sift_ply"] = str(sift_copy)
            cloud["colmap_posed_error"] = str(exc)
            _facade_pass(project.recon_dir / "cloud.ply", cloud)
    elif backend == "colmap":
        # Explicit classic mapper — only when user asked and poses missing.
        _emit(progress, "reconstruct", "COLMAP classic mapper (often fails on SV orbits)")
        ws, _names = export_colmap_images(frames, project.recon_dir / "colmap")
        ply_colmap = project.recon_dir / "cloud_colmap.ply"
        ply_path = run_colmap(ws, ply_out=ply_colmap)
        cloud_ply = project.recon_dir / "cloud.ply"
        shutil.copy2(ply_path, cloud_ply)
        cloud = {"path": str(cloud_ply), "source": "colmap", "colmap_ply": str(ply_path)}
        _facade_pass(cloud_ply, cloud)
    elif backend == "sift":
        _emit(progress, "reconstruct", "OpenCV SIFT stereo (cross-pano known poses)")
        cloud = triangulate_sift_frames(frames, project.recon_dir / "cloud.ply")
        _facade_pass(project.recon_dir / "cloud.ply", cloud)
    else:
        _emit(progress, "reconstruct", "triangulating flow correspondences (cross-pano pairs)")
        cloud = triangulate_frames(frames, project.recon_dir / "cloud.ply")
        _facade_pass(project.recon_dir / "cloud.ply", cloud)
    buildings = []
    if (project.bag_dir / "buildings.json").is_file():
        buildings = project.read_json(project.bag_dir / "buildings.json")
    georef = None
    georef_path = project.align_dir / "georef.json"
    if georef_path.is_file():
        try:
            georef = project.read_json(georef_path)
        except Exception:  # noqa: BLE001
            georef = None
    scene = scene_payload(
        spec_name=spec.name,
        bbox=spec.bbox,
        frame=frame,
        poses=poses,
        frames=frames,
        satellite=sat,
        cloud=cloud,
        buildings=live_buildings(buildings) if buildings else [],
        georef=georef,
    )
    _write_live(project, frame, buildings, poses)
    write_scene(project.recon_dir / "scene.json", scene)
    return scene


STAGES = (
    "discover",
    "capture",
    "crop",
    "satellite",
    "bag",
    "align",
    "interpolate",
    "reconstruct",
)


def run_all(
    project: Project,
    settings: Settings,
    *,
    from_stage: str = "discover",
    progress: Progress | None = None,
    align_prior: str | None = None,
    sat_edge_weight: float | None = None,
    cloud_clip_sat: bool | None = None,
    sat_cloud_margin_m: float | None = None,
) -> None:
    if from_stage not in STAGES:
        raise ValueError(f"unknown stage {from_stage}")
    start = STAGES.index(from_stage)
    table = {
        "discover": lambda: stage_discover(project, settings, progress),
        "capture": lambda: stage_capture(project, settings, progress),
        "crop": lambda: stage_crop(project, settings, progress),
        "satellite": lambda: stage_satellite(project, progress),
        "bag": lambda: stage_bag(project, progress),
        "align": lambda: stage_align(
            project,
            progress,
            align_prior=align_prior,
            sat_edge_weight=sat_edge_weight,
            cloud_clip_sat=cloud_clip_sat,
            sat_cloud_margin_m=sat_cloud_margin_m,
        ),
        "interpolate": lambda: stage_interpolate(project, progress),
        "reconstruct": lambda: stage_reconstruct(project, progress),
    }
    for name in STAGES[start:]:
        table[name]()
