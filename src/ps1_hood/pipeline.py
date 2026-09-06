"""Run the reconstruction stages in order."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from ps1_hood.align.bag_edges import snap_camera_to_bag
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
from ps1_hood.interpolate.video import write_video
from ps1_hood.overpass import fetch_roads
from ps1_hood.progress import emit as _emit
from ps1_hood.project import Project
from ps1_hood.reconstruct.colmap import COLMAP_HINT, export_colmap_images, run_colmap
from ps1_hood.reconstruct.export import scene_payload, write_scene
from ps1_hood.reconstruct.facades import extract_facades
from ps1_hood.reconstruct.keyframes import load_keyframes
from ps1_hood.reconstruct.mast3r import run_mast3r
from ps1_hood.reconstruct.unproject import triangulate_frames

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


def stage_align(project: Project, progress: Progress | None = None) -> list:
    spec = project.load_spec()
    shots = project.read_json(project.cropped_dir / "shots.json")
    shots, dropped = _shots_in_bbox(shots, spec.bbox)
    if dropped:
        log.info("dropped %s panos whose Google GPS sits outside the bbox", dropped)
        _emit(progress, "align", f"dropped {dropped} panos outside the bbox")
    osm = project.read_json(project.osm_dir / "roads.json")
    sat = project.read_json(project.satellite_dir / "ortho.json")
    frame = LocalFrame.from_bbox(spec.bbox)
    ortho = Ortho.load(sat, frame)
    buildings = project.read_json(project.bag_dir / "buildings.json") if (project.bag_dir / "buildings.json").is_file() else []
    footprints = building_footprints(buildings) if buildings else []
    shots_by: dict[str, list[dict[str, Any]]] = {}
    for shot in shots:
        shots_by.setdefault(shot["pano_id"], []).append(shot)
    _emit(progress, "align", "GPS prior, street constraint, then 3DBAG façade snap")
    poses = initial_poses(shots, spec, frame, osm)
    poses = refine_poses(
        poses,
        spec,
        frame,
        ortho,
        use_satellite=not buildings,
        # similar street photos match each other and collapse XY
        use_features=not bool(buildings),
    )
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
        if photo is not None and buildings:
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
            f"snap {i + 1}/{len(poses)}  residual={pose.get('residual_m')} m",
            aligned=i + 1,
            queued=len(poses),
        )
        _write_live(project, frame, buildings, live_cams)
    n_spread = uncollapse_along_gps(poses)
    if n_spread:
        log.info("restored GPS spacing for %s collapsed pano pairs", n_spread)
        _emit(progress, "align", f"spread {n_spread} stacked panos back along the street")
    for pose in poses:
        pose["e"], pose["n"] = push_out_of_footprints(pose["e"], pose["n"], footprints)
        lat, lon, _ = frame.to_geodetic(pose["e"], pose["n"], 0.0)
        pose["lat"] = lat
        pose["lon"] = lon

    cameras = explode_orbit_cameras(poses, shots)
    project.write_json(project.align_dir / "poses.json", poses)
    project.write_json(project.align_dir / "cameras.json", cameras)
    write_debug_overlay(poses, ortho, project.align_dir / "overlay.jpg")
    live_cams = []
    for pose in poses:
        live_cams.extend(horizon_cardinals(pose, shots_by.get(pose["pano_id"]) or []))
    _write_live(project, frame, buildings, live_cams)
    if poses and buildings:
        ce = sum(p["e"] for p in poses) / len(poses)
        cn = sum(p["n"] for p in poses) / len(poses)
        be = sum(0.5 * (b["bbox"][0] + b["bbox"][3]) for b in buildings) / len(buildings)
        bn = sum(0.5 * (b["bbox"][1] + b["bbox"][4]) for b in buildings) / len(buildings)
        offset = ((ce - be) ** 2 + (cn - bn) ** 2) ** 0.5
        snapped_n = sum(1 for p in poses if p.get("bag_snapped"))
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
    _emit(progress, "align", f"refined {len(poses)} panos / {len(cameras)} views", aligned=len(poses))
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

    # Real aligned shots for SfM / MVS; DIS midframes only as flow fallback.
    backend = spec.recon_backend
    if backend in {"colmap", "mast3r", "export"}:
        frames = keyframes
        source = "keyframes"
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
            frames = interp_frames
            source = "interp"
            log.warning(
                "only %s keyframes — falling back to %s interpolated frames",
                len(keyframes),
                len(interp_frames),
            )
        else:
            raise RuntimeError(
                "need ≥2 keyframes (or interp frames) to triangulate; "
                f"got keyframes={len(keyframes)} interp={len(interp_frames)}"
            )

    log.info("reconstruct: using %s (%s views) backend=%s", source, len(frames), backend)
    _emit(progress, "reconstruct", f"{backend} from {source} ({len(frames)} views)")

    cloud = None
    if backend == "export":
        export_colmap_images(frames, project.recon_dir / "colmap")
        _emit(progress, "reconstruct", "exported keyframes for an external reconstructor")
        _emit(progress, "reconstruct", COLMAP_HINT.strip())
    elif backend == "colmap":
        ws = export_colmap_images(frames, project.recon_dir / "colmap")
        ply_colmap = project.recon_dir / "cloud_colmap.ply"
        ply_path = run_colmap(ws, ply_out=ply_colmap)
        # Prefer the COLMAP cloud as cloud.ply when it has real geometry.
        cloud_ply = project.recon_dir / "cloud.ply"
        shutil.copy2(ply_path, cloud_ply)
        cloud = {
            "path": str(cloud_ply),
            "source": "colmap",
            "colmap_ply": str(ply_path),
        }
        try:
            fac = extract_facades(
                cloud_ply,
                project.recon_dir / "facades.obj",
                frames=frames,
                satellite=sat,
                local_frame=frame,
            )
            cloud["facades"] = fac
        except RuntimeError as exc:
            log.warning("facade pass skipped: %s", exc)
    elif backend == "mast3r":
        cloud = run_mast3r(frames, project.recon_dir)
    else:
        _emit(progress, "reconstruct", "triangulating flow correspondences (known-pose pairs)")
        cloud = triangulate_frames(frames, project.recon_dir / "cloud.ply")
        try:
            fac = extract_facades(
                project.recon_dir / "cloud.ply",
                project.recon_dir / "facades.obj",
                frames=frames,
                satellite=sat,
                local_frame=frame,
            )
            cloud["facades"] = fac
        except RuntimeError as exc:
            log.warning("facade pass skipped: %s", exc)
    buildings = []
    if (project.bag_dir / "buildings.json").is_file():
        buildings = project.read_json(project.bag_dir / "buildings.json")
    scene = scene_payload(
        spec_name=spec.name,
        bbox=spec.bbox,
        frame=frame,
        poses=poses,
        frames=frames,
        satellite=sat,
        cloud=cloud,
        buildings=live_buildings(buildings) if buildings else [],
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
        "align": lambda: stage_align(project, progress),
        "interpolate": lambda: stage_interpolate(project, progress),
        "reconstruct": lambda: stage_reconstruct(project, progress),
    }
    for name in STAGES[start:]:
        table[name]()
