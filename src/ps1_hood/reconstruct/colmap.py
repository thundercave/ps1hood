"""COLMAP export + known-pose triangulator (primary) / classic mapper (legacy).

Known-pose path for Street View orbits:
  export images → write cameras.txt/images.txt/empty points3D.txt →
  feature_extractor → **remap IMAGE_ID to database** → cross-pano matches →
  point_triangulator (NOT mapper).

See https://colmap.github.io/faq.html#reconstruct-sparse-dense-model-from-known-camera-poses
and colmap#497 (IMAGE_ID must match database or you get empty clouds).
"""

from __future__ import annotations

import functools
import logging
import math
import shutil
import sqlite3
import struct
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ps1_hood.geo import camera_rotation_cv
from ps1_hood.interpolate.sequence import order_track
from ps1_hood.reconstruct.known_pose import forward_drive_pairs, select_stereo_pairs

log = logging.getLogger(__name__)

COLMAP_HINT = """
Reconstruction prefers aligned Street View keyframes (align/cameras.json).

  COLMAP known poses (recommended for SV orbits — uses point_triangulator):
    ps1hood reconstruct <run> --backend colmap_posed
    (--backend colmap also prefers posed when cameras.json poses exist)

  Classic mapper often fails to init on SV orbits; do not retry it.

  MASt3R-as-matcher (learned matches → same posed triangulator):
    ps1hood reconstruct <run> --backend mast3r
    # or: --backend colmap_posed --matcher mast3r
    # NEVER free-pose MASt3R / GLOMAP as product — ENU poses stay fixed.

  Flow / OpenCV SIFT stereo are CPU photo fallbacks.
"""

_MIN_POINTS3D_BYTES = 1024
_MIN_PHOTO_POINTS = 200


def export_colmap_images(frames: list[dict[str, Any]], dest: Path) -> tuple[Path, list[str]]:
    """Copy frames into ``dest/images``. Returns ``(workspace, image_names)``."""
    dest.mkdir(parents=True, exist_ok=True)
    images = dest / "images"
    images.mkdir(exist_ok=True)
    used: set[str] = set()
    names: list[str] = []
    for i, frame in enumerate(frames):
        src = Path(frame["path"])
        pano = frame.get("pano_id") or src.parent.name
        candidate = f"{pano}_{src.name}"
        if candidate in used:
            candidate = f"{i:05d}_{pano}_{src.name}"
        used.add(candidate)
        target = images / candidate
        if not target.exists():
            shutil.copy2(src, target)
        names.append(candidate)
    return dest, names


def _points3d_count(path: Path) -> int:
    raw = path.read_bytes()
    if len(raw) < 8:
        return 0
    return int(struct.unpack("<Q", raw[:8])[0])


def _points3d_count_txt(path: Path) -> int:
    n = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        n += 1
    return n


def find_sparse_model(sparse: Path) -> Path | None:
    if not sparse.is_dir():
        return None
    children = sorted(
        (p for p in sparse.iterdir() if p.is_dir()),
        key=lambda p: (not p.name.isdigit(), int(p.name) if p.name.isdigit() else p.name),
    )
    for child in children:
        if (child / "points3D.bin").is_file() or (child / "points3D.txt").is_file():
            return child
    if (sparse / "points3D.bin").is_file() or (sparse / "points3D.txt").is_file():
        return sparse
    return None


def _count_model_points(model: Path) -> int:
    bin_path = model / "points3D.bin"
    txt_path = model / "points3D.txt"
    if bin_path.is_file() and bin_path.stat().st_size >= 8:
        return _points3d_count(bin_path)
    if txt_path.is_file():
        return _points3d_count_txt(txt_path)
    return 0


def track_length_stats(model: Path) -> dict[str, float | int] | None:
    """Track-length stats from points3D.bin (None if unavailable).

    Returns ``n_points``, ``mean_track_length``, ``n_ge3`` (points with ≥3 views),
    and ``frac_ge3``. Mean ≈ 2.0 means the match graph is a matching of edges,
    not multi-view tracks — OpenMVS neighbor select starves.
    """
    bin_path = model / "points3D.bin"
    if not bin_path.is_file() or bin_path.stat().st_size < 8:
        return None
    raw = bin_path.read_bytes()
    n = int(struct.unpack("<Q", raw[:8])[0])
    if n < 1:
        return {
            "n_points": 0,
            "mean_track_length": 0.0,
            "n_ge3": 0,
            "frac_ge3": 0.0,
        }
    off = 8
    total = 0
    n_ge3 = 0
    for _ in range(n):
        if off + 8 + 24 + 3 + 8 + 8 > len(raw):
            return None
        off += 8  # point3D_id
        off += 24  # xyz
        off += 3  # rgb
        off += 8  # error
        track_len = int(struct.unpack_from("<Q", raw, off)[0])
        off += 8
        total += track_len
        if track_len >= 3:
            n_ge3 += 1
        off += track_len * 8  # (image_id, point2D_idx) pairs
        if off > len(raw) + 1:
            return None
    return {
        "n_points": n,
        "mean_track_length": total / float(n),
        "n_ge3": n_ge3,
        "frac_ge3": n_ge3 / float(n),
    }


def mean_track_length(model: Path) -> float | None:
    """Mean observations/point from points3D.bin (None if unavailable)."""
    stats = track_length_stats(model)
    if stats is None:
        return None
    return float(stats["mean_track_length"])


def _validate_sparse_model(model: Path, *, min_points: int = 1, label: str = "COLMAP") -> int:
    points_bin = model / "points3D.bin"
    points_txt = model / "points3D.txt"
    if not points_bin.is_file() and not points_txt.is_file():
        raise RuntimeError(
            f"{label} wrote no points3D under {model}. "
            "Use --backend colmap_posed / flow / OpenCV SIFT — do not retry mapper on SV orbits."
        )
    n = _count_model_points(model)
    size = points_bin.stat().st_size if points_bin.is_file() else points_txt.stat().st_size
    if size < _MIN_POINTS3D_BYTES or n < min_points:
        raise RuntimeError(
            f"{label} sparse model too thin ({n} points, {size} bytes at {model}). "
            f"Need ≥{min_points}. Next photo path: denser capture or OpenCV SIFT stereo."
        )
    return n


def _rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    """COLMAP quaternion (qw, qx, qy, qz) from 3×3 world→camera rotation."""
    qvec = np.empty(4, dtype=np.float64)
    trace = float(R[0, 0] + R[1, 1] + R[2, 2])
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qvec[0] = 0.25 * s
        qvec[1] = (R[2, 1] - R[1, 2]) / s
        qvec[2] = (R[0, 2] - R[2, 0]) / s
        qvec[3] = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qvec[0] = (R[2, 1] - R[1, 2]) / s
        qvec[1] = 0.25 * s
        qvec[2] = (R[0, 1] + R[1, 0]) / s
        qvec[3] = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qvec[0] = (R[0, 2] - R[2, 0]) / s
        qvec[1] = (R[0, 1] + R[1, 0]) / s
        qvec[2] = 0.25 * s
        qvec[3] = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qvec[0] = (R[1, 0] - R[0, 1]) / s
        qvec[1] = (R[0, 2] + R[2, 0]) / s
        qvec[2] = (R[1, 2] + R[2, 1]) / s
        qvec[3] = 0.25 * s
    if qvec[0] < 0:
        qvec *= -1.0
    return qvec


def _focal_px(width: int, fov_deg: float) -> float:
    """fx = (W/2) / tan(hfov/2) from crop horizontal FoV."""
    return (0.5 * float(width)) / math.tan(math.radians(float(fov_deg)) / 2.0)


def _frame_size(frame: dict[str, Any]) -> tuple[int, int]:
    w = frame.get("width")
    h = frame.get("height")
    if w and h:
        return int(w), int(h)
    img = cv2.imread(str(frame["path"]), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"cannot read image for COLMAP pose export: {frame['path']}")
    hh, ww = img.shape[:2]
    return int(ww), int(hh)


def _pose_has_enu(frame: dict[str, Any]) -> bool:
    return all(k in frame for k in ("e", "n", "u", "heading"))


def frames_have_known_poses(frames: list[dict[str, Any]]) -> bool:
    return len(frames) >= 2 and all(_pose_has_enu(f) for f in frames)


def _pano_key(frame: dict[str, Any]) -> str:
    if frame.get("pano_id"):
        return str(frame["pano_id"])
    return f"{round(float(frame['e']), 2)}_{round(float(frame['n']), 2)}"


def _is_cross_pano_baseline(
    frames: list[dict[str, Any]], i: int, j: int, *, min_xy_m: float = 1.5
) -> bool:
    if _pano_key(frames[i]) == _pano_key(frames[j]):
        return False
    be = abs(float(frames[i]["e"]) - float(frames[j]["e"]))
    bn = abs(float(frames[i]["n"]) - float(frames[j]["n"]))
    return math.hypot(be, bn) >= min_xy_m


def cross_pano_pair_indices(
    frames: list[dict[str, Any]],
    *,
    max_pairs_per_frame: int = 8,
    min_baseline_m: float = 2.0,
) -> list[tuple[int, int]]:
    """Stereo pairs with real baseline — never same-center / same-pano orbit mates.

    Defaults prefer **more** cross-pano matches (``max_pairs_per_frame=8``) so
    posed sparse covisibility is denser for OpenMVS neighbor selection.
    Prefer ``cross_pano_forward_pairs`` for multi-view track growth (3 forward
    mates along the drive); this helper remains for stereo / OpenMVS neighbors.
    """
    pairs = select_stereo_pairs(
        frames,
        min_baseline_m=min_baseline_m,
        max_pairs_per_frame=max_pairs_per_frame,
    )
    out: list[tuple[int, int]] = []
    for i, j in pairs:
        if _is_cross_pano_baseline(frames, i, j):
            out.append((i, j))
    return out


def cross_pano_forward_pairs(
    frames: list[dict[str, Any]],
    *,
    n_forward: int = 3,
    min_baseline_m: float = 2.0,
    max_baseline_m: float = 25.0,
) -> list[tuple[int, int]]:
    """Match pairs: each frame → up to ``n_forward`` later drive mates (cross-pano).

    Orders frames with ``order_track`` so sequential + skip-1/skip-2 edges form
    3-cycles after triangulation (mean track ≫ 2.0). Same-pano orbit mates are
    dropped. Falls back to ``cross_pano_pair_indices`` if the drive order yields
    no usable pairs.
    """
    if len(frames) < 2:
        return []
    # order_track returns pose dicts; map back to indices by identity.
    ordered = order_track(frames)
    if len(ordered) < 2:
        # Degenerate path — keep list order.
        order_indices = list(range(len(frames)))
    else:
        # Match by object identity first, then by (e,n,heading,pano) for copies.
        id_map = {id(f): i for i, f in enumerate(frames)}
        used: set[int] = set()
        order_indices: list[int] = []
        for fr in ordered:
            idx = id_map.get(id(fr))
            if idx is None:
                for i, f in enumerate(frames):
                    if i in used:
                        continue
                    if (
                        abs(float(f["e"]) - float(fr["e"])) < 1e-6
                        and abs(float(f["n"]) - float(fr["n"])) < 1e-6
                        and abs(float(f["heading"]) - float(fr["heading"])) < 1e-3
                        and _pano_key(f) == _pano_key(fr)
                    ):
                        idx = i
                        break
            if idx is None or idx in used:
                continue
            used.add(idx)
            order_indices.append(idx)
        # Append any frames order_track dropped (orbit leftovers).
        for i in range(len(frames)):
            if i not in used:
                order_indices.append(i)

    pairs = forward_drive_pairs(
        frames,
        n_forward=n_forward,
        min_baseline_m=min_baseline_m,
        max_baseline_m=max_baseline_m,
        order_indices=order_indices,
        quadratic_overlap=True,
    )
    out_set: set[tuple[int, int]] = {
        (i, j) for i, j in pairs if _is_cross_pano_baseline(frames, i, j)
    }
    # Union preferred-baseline stereo pairs for extra redundant edges.
    out_set.update(cross_pano_pair_indices(frames, min_baseline_m=min_baseline_m))
    out = sorted(out_set)
    if not out:
        log.warning(
            "forward drive pairs empty after cross-pano filter — "
            "falling back to select_stereo_pairs"
        )
        return cross_pano_pair_indices(frames, min_baseline_m=min_baseline_m)
    return out


def write_known_pose_model(
    frames: list[dict[str, Any]],
    image_names: list[str],
    model_dir: Path,
    *,
    image_ids: list[int] | None = None,
    camera_ids: list[int] | None = None,
    db_cameras: dict[int, tuple[str, int, int, list[float]]] | None = None,
) -> Path:
    """Write COLMAP text model; empty points3D.txt. PINHOLE, t = -R @ C.

    When remapping after feature_extractor, pass ``camera_ids`` + ``db_cameras``
    from the COLMAP SQLite DB so cameras.txt WIDTH/HEIGHT/PARAMS match the DB
    for every CAMERA_ID (avoids point_triangulator SIGABRT on size mismatches).
    """
    if len(frames) != len(image_names):
        raise ValueError("frames and image_names length mismatch")
    if not frames_have_known_poses(frames):
        raise RuntimeError("known-pose COLMAP needs e/n/u/heading on every frame")

    model_dir.mkdir(parents=True, exist_ok=True)

    cameras_txt = model_dir / "cameras.txt"
    if db_cameras is not None:
        if camera_ids is None:
            raise ValueError("db_cameras requires camera_ids")
        frame_cam_ids = [int(c) for c in camera_ids]
        used = sorted(set(frame_cam_ids))
        missing_cams = [cid for cid in used if cid not in db_cameras]
        if missing_cams:
            raise RuntimeError(
                f"COLMAP database missing cameras rows for CAMERA_ID(s) {missing_cams}"
            )
        with cameras_txt.open("w", encoding="ascii") as fh:
            fh.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            fh.write(f"# Number of cameras: {len(used)}\n")
            for cid in used:
                model_name, w, h, params = db_cameras[cid]
                params_s = " ".join(f"{p:.6f}" for p in params)
                fh.write(f"{cid} {model_name} {int(w)} {int(h)} {params_s}\n")
    else:
        # One PINHOLE camera per (w, h, fov); used for the pre-extractor prior.
        cam_key_to_id: dict[tuple[int, int, float], int] = {}
        cam_defs: list[tuple[int, int, int, float, float, float, float]] = []
        # id, w, h, fx, fy, cx, cy
        frame_cam_ids = []
        for idx, fr in enumerate(frames):
            w, h = _frame_size(fr)
            fov = float(fr.get("fov") or 90.0)
            key = (w, h, round(fov, 3))
            if key not in cam_key_to_id:
                if camera_ids is not None:
                    cid = int(camera_ids[idx])
                else:
                    cid = len(cam_key_to_id) + 1
                cam_key_to_id[key] = cid
                fx = _focal_px(w, fov)
                fy = fx  # square pixels
                cam_defs.append((cid, w, h, fx, fy, w / 2.0, h / 2.0))
            frame_cam_ids.append(cam_key_to_id[key])

        if camera_ids is not None:
            # Legacy path without db_cameras: pin ids, rebuild from first frame.
            frame_cam_ids = [int(c) for c in camera_ids]
            seen: dict[int, tuple[int, int, int, float, float, float, float]] = {}
            for fr, cid in zip(frames, frame_cam_ids, strict=True):
                if cid in seen:
                    continue
                w, h = _frame_size(fr)
                fov = float(fr.get("fov") or 90.0)
                fx = _focal_px(w, fov)
                seen[cid] = (cid, w, h, fx, fx, w / 2.0, h / 2.0)
            cam_defs = list(seen.values())

        with cameras_txt.open("w", encoding="ascii") as fh:
            fh.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            fh.write(f"# Number of cameras: {len(cam_defs)}\n")
            for cid, w, h, fx, fy, cx, cy in cam_defs:
                fh.write(f"{cid} PINHOLE {w} {h} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")

    ids = image_ids if image_ids is not None else list(range(1, len(frames) + 1))
    if len(ids) != len(frames):
        raise ValueError("image_ids length mismatch")

    images_txt = model_dir / "images.txt"
    with images_txt.open("w", encoding="ascii") as fh:
        fh.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        fh.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        fh.write(f"# Number of images: {len(frames)}\n")
        for img_id, fr, name, cid in zip(ids, frames, image_names, frame_cam_ids, strict=True):
            R_wc = np.array(
                camera_rotation_cv(float(fr["heading"]), float(fr.get("pitch") or 0.0)),
                dtype=np.float64,
            )
            R_cw = R_wc.T  # world → camera
            C = np.array([float(fr["e"]), float(fr["n"]), float(fr["u"])], dtype=np.float64)
            t = -R_cw @ C  # NOT the ENU center
            q = _rotmat_to_qvec(R_cw)
            fh.write(
                f"{int(img_id)} {q[0]:.10f} {q[1]:.10f} {q[2]:.10f} {q[3]:.10f} "
                f"{t[0]:.10f} {t[1]:.10f} {t[2]:.10f} {int(cid)} {name}\n"
            )
            fh.write("\n")

    (model_dir / "points3D.txt").write_text(
        "# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n# Number of points: 0\n",
        encoding="ascii",
    )
    return model_dir


# COLMAP CameraModelId → cameras.txt model name (src/colmap/sensor/models.h).
_CAMERA_MODEL_NAMES: dict[int, str] = {
    0: "SIMPLE_PINHOLE",
    1: "PINHOLE",
    2: "SIMPLE_RADIAL",
    3: "RADIAL",
    4: "OPENCV",
    5: "OPENCV_FISHEYE",
    6: "FULL_OPENCV",
    7: "FOV",
    8: "SIMPLE_RADIAL_FISHEYE",
    9: "RADIAL_FISHEYE",
    10: "THIN_PRISM_FISHEYE",
}


def read_db_image_ids(database: Path) -> dict[str, tuple[int, int]]:
    """Map image basename → (image_id, camera_id) from COLMAP SQLite DB."""
    con = sqlite3.connect(str(database))
    try:
        rows = con.execute("SELECT image_id, name, camera_id FROM images").fetchall()
    finally:
        con.close()
    out: dict[str, tuple[int, int]] = {}
    for image_id, name, camera_id in rows:
        out[str(name)] = (int(image_id), int(camera_id))
    return out


def read_db_cameras(database: Path) -> dict[int, tuple[str, int, int, list[float]]]:
    """Map camera_id → (model_name, width, height, params) from COLMAP SQLite DB.

    ``params`` is the float64 blob decoded to a Python list (PINHOLE: fx,fy,cx,cy).
    """
    con = sqlite3.connect(str(database))
    try:
        rows = con.execute(
            "SELECT camera_id, model, width, height, params FROM cameras"
        ).fetchall()
    finally:
        con.close()
    out: dict[int, tuple[str, int, int, list[float]]] = {}
    for camera_id, model, width, height, params_blob in rows:
        model_id = int(model)
        model_name = _CAMERA_MODEL_NAMES.get(model_id)
        if model_name is None:
            raise RuntimeError(f"unsupported COLMAP camera model id {model_id}")
        params = list(np.frombuffer(params_blob, dtype=np.float64)) if params_blob else []
        out[int(camera_id)] = (model_name, int(width), int(height), params)
    return out


def remap_known_pose_model_to_db(
    model_dir: Path,
    frames: list[dict[str, Any]],
    image_names: list[str],
    database: Path,
    *,
    max_missing_frac: float = 0.10,
    min_images: int = 4,
) -> tuple[Path, list[dict[str, Any]], list[str]]:
    """Rewrite images.txt IMAGE_IDs/CAMERA_IDs to match feature_extractor DB.

    cameras.txt is rewritten from the DB ``cameras`` table (not frame metadata)
    so WIDTH/HEIGHT/PARAMS match for every CAMERA_ID used in images.txt.

    If a small minority of ``image_names`` are absent from the DB (e.g. COLMAP
    silently skipped odd-sized crops under ``single_camera``), drop those frames,
    rewrite the prior with survivors, and return the filtered lists. Hard-fail
    when too many are missing, too few remain, or no cross-pano pairs survive.
    """
    if len(frames) != len(image_names):
        raise ValueError("frames / image_names mismatch")
    db_map = read_db_image_ids(database)
    missing = [n for n in image_names if n not in db_map]
    kept_frames = frames
    kept_names = image_names
    if missing:
        frac = len(missing) / max(len(image_names), 1)
        survivors = [(f, n) for f, n in zip(frames, image_names, strict=True) if n in db_map]
        if frac > max_missing_frac or len(survivors) < min_images:
            raise RuntimeError(
                f"COLMAP database missing {len(missing)}/{len(image_names)} images "
                f"after feature_extractor (e.g. {missing[0]}). Cannot remap IMAGE_IDs "
                f"(frac={frac:.1%} > {max_missing_frac:.0%} or survivors={len(survivors)} "
                f"< {min_images})."
            )
        kept_frames = [f for f, _ in survivors]
        kept_names = [n for _, n in survivors]
        if not cross_pano_pair_indices(kept_frames):
            raise RuntimeError(
                f"COLMAP database missing {len(missing)} images (e.g. {missing[0]}); "
                "after filtering survivors there are no cross-pano stereo pairs left."
            )
        log.warning(
            "COLMAP DB missing %s/%s images after feature_extractor (e.g. %s); "
            "continuing with %s survivors",
            len(missing),
            len(image_names),
            missing[0],
            len(kept_names),
        )
    image_ids = [db_map[n][0] for n in kept_names]
    camera_ids = [db_map[n][1] for n in kept_names]
    # Always write cameras.txt from the DB so WIDTH/HEIGHT/PARAMS match the
    # cameras table for every CAMERA_ID (size-grouping from frames can collide
    # with DB camera_ids at different dimensions → triangulator SIGABRT).
    db_cameras = read_db_cameras(database)
    path = write_known_pose_model(
        kept_frames,
        kept_names,
        model_dir,
        image_ids=image_ids,
        camera_ids=camera_ids,
        db_cameras=db_cameras,
    )
    return path, kept_frames, kept_names


def write_cross_pano_match_list(
    frames: list[dict[str, Any]],
    image_names: list[str],
    path: Path,
    *,
    n_forward: int = 3,
) -> int:
    """Write image-name pairs for matches_importer (cross-pano forward mates).

    Default: each frame ↔ up to ``n_forward`` later drive neighbors so tracks
    can chain across 3+ views. Same-pano orbit mates stay excluded.
    """
    pairs = cross_pano_forward_pairs(frames, n_forward=n_forward)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as fh:
        for i, j in pairs:
            fh.write(f"{image_names[i]} {image_names[j]}\n")
    log.info(
        "COLMAP match list: %s cross-pano forward pairs (n_forward=%s) → %s",
        len(pairs),
        n_forward,
        path,
    )
    return len(pairs)


def _pair_id_to_image_ids(pair_id: int) -> tuple[int, int]:
    """Decode COLMAP ``pair_id`` → ``(image_id1, image_id2)`` with id1 < id2."""
    # Database::PairIdToImagePair — kMaxNumImages = 2^31 - 1
    max_images = 2147483647
    id2 = int(pair_id % max_images)
    id1 = int((pair_id - id2) // max_images)
    return id1, id2


def image_ids_with_two_view_tracks(database: Path) -> set[int]:
    """Image IDs that appear in at least one non-empty ``two_view_geometries`` row."""
    con = sqlite3.connect(str(database))
    try:
        rows = con.execute(
            "SELECT pair_id, rows FROM two_view_geometries WHERE rows IS NOT NULL AND rows > 0"
        ).fetchall()
    finally:
        con.close()
    out: set[int] = set()
    for pair_id, _n in rows:
        a, b = _pair_id_to_image_ids(int(pair_id))
        out.add(a)
        out.add(b)
    return out


def filter_frames_registered_in_matches(
    frames: list[dict[str, Any]],
    image_names: list[str],
    database: Path,
    *,
    model_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Drop images with no successful two-view geometry (COLMAP may SIGABRT on them).

    If ``model_dir`` is set, rewrite the known-pose prior for survivors (DB IMAGE_IDs).
    """
    if len(frames) != len(image_names):
        raise ValueError("frames / image_names mismatch")
    db_map = read_db_image_ids(database)
    tracked = image_ids_with_two_view_tracks(database)
    kept_frames: list[dict[str, Any]] = []
    kept_names: list[str] = []
    dropped: list[str] = []
    for fr, name in zip(frames, image_names, strict=True):
        if name not in db_map:
            dropped.append(name)
            continue
        iid = db_map[name][0]
        if iid not in tracked:
            dropped.append(name)
            continue
        kept_frames.append(fr)
        kept_names.append(name)
    if dropped:
        log.warning(
            "COLMAP: dropping %s/%s images with no verified two-view tracks "
            "(e.g. %s); continuing with %s",
            len(dropped),
            len(image_names),
            dropped[0],
            len(kept_names),
        )
    if len(kept_names) < 4:
        raise RuntimeError(
            f"Too few images with verified matches after filtering "
            f"({len(kept_names)} left; dropped {len(dropped)})."
        )
    if not cross_pano_pair_indices(kept_frames):
        raise RuntimeError(
            "No cross-pano pairs left after dropping images without two-view tracks."
        )
    if model_dir is not None:
        image_ids = [db_map[n][0] for n in kept_names]
        camera_ids = [db_map[n][1] for n in kept_names]
        db_cameras = read_db_cameras(database)
        write_known_pose_model(
            kept_frames,
            kept_names,
            model_dir,
            image_ids=image_ids,
            camera_ids=camera_ids,
            db_cameras=db_cameras,
        )
    return kept_frames, kept_names



def choose_colmap_cpu_gpu_flags(
    feature_help: str,
    matching_help: str = "",
) -> tuple[list[str], list[str]]:
    """Pick ``use_gpu=0`` argv for feature extraction and matching.

    COLMAP ≤3.12 uses ``--SiftExtraction.use_gpu`` / ``--SiftMatching.use_gpu``.
    COLMAP 3.13+ renamed these to ``--FeatureExtraction.use_gpu`` /
    ``--FeatureMatching.use_gpu``. Prefer the names present in ``*-h`` output;
    if help is empty/unknown, keep the legacy Sift* names.
    """
    extract_key = (
        "FeatureExtraction"
        if "FeatureExtraction.use_gpu" in feature_help
        else "SiftExtraction"
    )
    match_src = matching_help if matching_help else feature_help
    match_key = (
        "FeatureMatching"
        if "FeatureMatching.use_gpu" in match_src
        else "SiftMatching"
    )
    return (
        [f"--{extract_key}.use_gpu", "0"],
        [f"--{match_key}.use_gpu", "0"],
    )


def _colmap_subcommand_help(colmap: str, subcommand: str) -> str:
    try:
        proc = subprocess.run(
            [colmap, subcommand, "-h"],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return (proc.stdout or "") + (proc.stderr or "")


@functools.lru_cache(maxsize=8)
def colmap_cpu_gpu_flags(colmap: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Probe COLMAP help and return cached CPU ``use_gpu=0`` flag tuples.

    Returns ``(extract_flags, match_flags)`` suitable for ``list.extend`` /
    splat into feature_extractor / matcher argv. Always forces GPU off (AMD /
    no-CUDA boxes).
    """
    feat_help = _colmap_subcommand_help(colmap, "feature_extractor")
    match_help = _colmap_subcommand_help(colmap, "exhaustive_matcher")
    extract, matching = choose_colmap_cpu_gpu_flags(feat_help, match_help)
    return tuple(extract), tuple(matching)


def clear_db_matches(database: Path) -> None:
    """Drop matcher output so a retry does not see a half-written DB."""
    con = sqlite3.connect(str(database))
    try:
        con.execute("DELETE FROM two_view_geometries")
        con.execute("DELETE FROM matches")
        con.commit()
    except sqlite3.OperationalError:
        # Tables may not exist yet — nothing to clear.
        pass
    finally:
        con.close()


def matches_importer_argv(
    colmap: str,
    database: Path,
    match_list: Path,
    *,
    guided_matching: bool = False,
    max_num_matches: int | None = None,
) -> list[str]:
    """Build ``matches_importer`` argv for cross-pano pair lists.

    Guided matching is **off by default** (OOM risk on dense FILM sets). When
    enabled, constrain with ``max_num_matches`` (2048–4096) and keep matching
    on the custom pair list only — never exhaustive+guided.
    """
    _extract_gpu, match_gpu = colmap_cpu_gpu_flags(colmap)
    cmd = [
        colmap,
        "matches_importer",
        "--database_path",
        str(database),
        "--match_list_path",
        str(match_list),
        "--match_type",
        "pairs",
        *match_gpu,
    ]
    help_txt = ""
    if guided_matching or max_num_matches is not None:
        help_txt = _colmap_subcommand_help(colmap, "matches_importer")
    if guided_matching:
        if "guided_matching" in help_txt or not help_txt:
            # Empty help → still try the flag (COLMAP 3.10+ has it).
            cmd.extend(["--SiftMatching.guided_matching", "1"])
        # Cap matches when guiding to reduce OOM (pair list already scoped).
        cap = 4096 if max_num_matches is None else int(max_num_matches)
        if "max_num_matches" in help_txt or not help_txt:
            cmd.extend(["--SiftMatching.max_num_matches", str(cap)])
    elif max_num_matches is not None:
        if "max_num_matches" in help_txt or not help_txt:
            cmd.extend(["--SiftMatching.max_num_matches", str(int(max_num_matches))])
    return cmd


def _colmap_bin() -> str:
    colmap = shutil.which("colmap")
    if not colmap:
        raise RuntimeError("colmap is not on PATH." + COLMAP_HINT)
    return colmap


def _run(cmd: list[str]) -> None:
    log.info("COLMAP: %s", " ".join(cmd[1:4]))
    subprocess.run(cmd, check=True)


def _convert_model_to_ply(colmap: str, model: Path, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            colmap,
            "model_converter",
            "--input_path",
            str(model),
            "--output_path",
            str(out),
            "--output_type",
            "PLY",
        ]
    )
    if not out.is_file() or out.stat().st_size < 64:
        raise RuntimeError(f"COLMAP model_converter failed to write a usable PLY at {out}")
    return out


def run_colmap(workspace: Path, *, ply_out: Path | None = None) -> Path:
    """Legacy classic mapper — expect failure on SV orbits; prefer run_colmap_posed."""
    colmap = _colmap_bin()
    db = workspace / "database.db"
    if db.exists():
        db.unlink()
    images = workspace / "images"
    sparse = workspace / "sparse"
    if sparse.exists():
        shutil.rmtree(sparse)
    sparse.mkdir(parents=True, exist_ok=True)
    extract_gpu, match_gpu = colmap_cpu_gpu_flags(colmap)
    _run(
        [
            colmap,
            "feature_extractor",
            "--database_path",
            str(db),
            "--image_path",
            str(images),
            "--ImageReader.single_camera",
            "1",
            "--ImageReader.camera_model",
            "PINHOLE",
            *extract_gpu,
        ]
    )
    _run(
        [
            colmap,
            "exhaustive_matcher",
            "--database_path",
            str(db),
            *match_gpu,
        ]
    )
    _run(
        [
            colmap,
            "mapper",
            "--database_path",
            str(db),
            "--image_path",
            str(images),
            "--output_path",
            str(sparse),
        ]
    )
    model = find_sparse_model(sparse)
    if model is None:
        raise RuntimeError(
            f"COLMAP mapper produced no sparse model under {sparse}. "
            "Expected on SV orbits — use --backend colmap_posed (no mapper retry)."
        )
    n = _validate_sparse_model(model, label="COLMAP mapper")
    out = ply_out if ply_out is not None else workspace.parent / "cloud_colmap.ply"
    _convert_model_to_ply(colmap, model, out)
    log.info("COLMAP mapper %s → %s (%s points)", model, out, n)
    return out



def frame_sizes_uniform(frames: list[dict[str, Any]]) -> bool:
    """True when every frame reports the same (width, height)."""
    if not frames:
        return True
    sizes = {_frame_size(f) for f in frames}
    return len(sizes) <= 1


def feature_extractor_argv(
    colmap: str,
    database: Path,
    image_path: Path,
    frames: list[dict[str, Any]],
    *,
    max_image_size: int | None = None,
) -> list[str]:
    """Build ``colmap feature_extractor`` argv for the posed path.

    When crop sizes differ, do **not** force ``single_camera`` + global
    ``camera_params`` (COLMAP then silently omits odd-sized images). Prefer
    per-image cameras so intrinsics stay honest for known-pose triangulation;
    ``write_known_pose_model`` already emits one PINHOLE per (w, h, fov).

    Optional ``max_image_size`` (1200–1600) downscales before SIFT — use when
    enabling guided matching to avoid OOM.
    """
    extract_gpu, _match_gpu = colmap_cpu_gpu_flags(colmap)
    cmd = [
        colmap,
        "feature_extractor",
        "--database_path",
        str(database),
        "--image_path",
        str(image_path),
        "--ImageReader.camera_model",
        "PINHOLE",
        *extract_gpu,
    ]
    if max_image_size is not None:
        cmd.extend(["--SiftExtraction.max_image_size", str(int(max_image_size))])
    if frame_sizes_uniform(frames):
        first_w, first_h = _frame_size(frames[0])
        first_fov = float(frames[0].get("fov") or 90.0)
        fx = _focal_px(first_w, first_fov)
        cam_params = f"{fx:.6f},{fx:.6f},{first_w / 2.0:.6f},{first_h / 2.0:.6f}"
        cmd.extend(
            [
                "--ImageReader.single_camera",
                "1",
                "--ImageReader.camera_params",
                cam_params,
            ]
        )
    else:
        # Multi-camera: let COLMAP register each image with its own size.
        cmd.extend(["--ImageReader.single_camera", "0"])
        log.info(
            "COLMAP posed: frame sizes differ — using single_camera=0 (no global camera_params)"
        )
    return cmd


def run_colmap_posed(
    workspace: Path,
    frames: list[dict[str, Any]],
    image_names: list[str],
    *,
    ply_out: Path | None = None,
    min_points: int = _MIN_PHOTO_POINTS,
    guided_matching: bool = False,
    max_image_size: int | None = None,
    max_num_matches: int | None = None,
    n_forward: int = 3,
    matcher: str = "sift",
) -> Path:
    """Triangulate with known poses via point_triangulator (never mapper).

    Critical steps for SV orbits:
    - PINHOLE from crop FoV; t = -R @ C
    - ``matcher=sift`` (default): feature_extractor → remap IMAGE_ID (colmap#497)
      → matches_importer on cross-pano forward mates
    - ``matcher=mast3r``: bootstrap DB from ENU priors → MASt3R pairwise matches
      into keypoints/matches (no free-pose SfM / GLOMAP)
    - Triangulator: clear_points, allow two-view tracks, low min tri angle
    - No PatchMatch / dense (needs GPU)
    """
    matcher = (matcher or "sift").strip().lower()
    if matcher not in MATCHERS:
        raise ValueError(f"unknown matcher {matcher!r}; expected one of {MATCHERS}")

    colmap = _colmap_bin()
    images = workspace / "images"
    if not images.is_dir() or not any(images.iterdir()):
        raise RuntimeError(f"no images under {images}; export_colmap_images first")
    if len(frames) != len(image_names):
        raise ValueError("frames / image_names mismatch")

    pair_count = len(cross_pano_forward_pairs(frames, n_forward=n_forward))
    if pair_count < 1:
        raise RuntimeError(
            "no cross-pano stereo pairs (need drive-adjacent views with baseline). "
            "Same-pano orbit headings are pure rotation and cannot triangulate. "
            "Capture denser along the street."
        )

    db = workspace / "database.db"
    if db.exists():
        db.unlink()

    sparse_prior = workspace / "sparse_prior"
    if sparse_prior.exists():
        shutil.rmtree(sparse_prior)
    write_known_pose_model(frames, image_names, sparse_prior)

    match_list = workspace / "cross_pano_pairs.txt"
    n_pairs = write_cross_pano_match_list(
        frames, image_names, match_list, n_forward=n_forward
    )
    if n_pairs < 1:
        raise RuntimeError("cross-pano match list empty")

    if matcher == "mast3r":
        from ps1_hood.reconstruct.mast3r import fill_database_with_mast3r_matches

        log.info(
            "COLMAP posed matcher=mast3r (%s cross-pano pairs) — ENU priors locked",
            n_pairs,
        )
        # IMAGE_IDs 1..N match write_known_pose_model; no feature_extractor remap.
        bootstrap_posed_database(db, frames, image_names)
        fill_database_with_mast3r_matches(
            db,
            images,
            frames,
            image_names,
            pair_indices=cross_pano_forward_pairs(frames, n_forward=n_forward),
        )
    else:
        # When guiding, downscale SIFT (1200–1600) to reduce OOM risk.
        extract_size = max_image_size
        if guided_matching and extract_size is None:
            extract_size = 1600
        match_cap = max_num_matches
        if guided_matching and match_cap is None:
            match_cap = 4096
        _run(
            feature_extractor_argv(
                colmap, db, images, frames, max_image_size=extract_size
            )
        )

        # colmap#497: IMAGE_ID in images.txt MUST match database after extraction.
        # May drop a minority of images COLMAP omitted (odd crop sizes).
        _, frames, image_names = remap_known_pose_model_to_db(
            sparse_prior, frames, image_names, db
        )

        # Rewrite match list after possible survivor filter.
        n_pairs = write_cross_pano_match_list(
            frames, image_names, match_list, n_forward=n_forward
        )
        if n_pairs < 1:
            raise RuntimeError("cross-pano match list empty after DB remap")

        # Guided matching off by default. If enabled and it OOMs, clear + retry plain.
        try:
            _run(
                matches_importer_argv(
                    colmap,
                    db,
                    match_list,
                    guided_matching=guided_matching,
                    max_num_matches=match_cap,
                )
            )
        except (subprocess.CalledProcessError, OSError) as exc:
            if not guided_matching:
                raise
            log.warning(
                "COLMAP matches_importer with guided_matching failed (%s); "
                "retrying without guided matching",
                exc,
            )
            clear_db_matches(db)
            _run(
                matches_importer_argv(
                    colmap, db, match_list, guided_matching=False
                )
            )

    # Images with only failed geometric verification can SIGABRT point_triangulator.
    frames, image_names = filter_frames_registered_in_matches(
        frames, image_names, db, model_dir=sparse_prior
    )

    sparse_out = workspace / "sparse_posed"
    if sparse_out.exists():
        shutil.rmtree(sparse_out)
    sparse_out.mkdir(parents=True, exist_ok=True)

    _run(point_triangulator_argv(colmap, db, images, sparse_prior, sparse_out))

    model = find_sparse_model(sparse_out)
    if model is None and (
        (sparse_out / "points3D.bin").is_file() or (sparse_out / "points3D.txt").is_file()
    ):
        model = sparse_out
    if model is None:
        raise RuntimeError(
            f"COLMAP point_triangulator produced no points3D under {sparse_out}. "
            "Posed path failed on this SV orbit. Next photo-only step: denser "
            "drive spacing / OpenCV SIFT stereo on cross-pano pairs."
        )

    out = ply_out if ply_out is not None else workspace.parent / "cloud_photo.ply"
    _convert_model_to_ply(colmap, model, out)
    n = _count_model_points(model)
    stats = track_length_stats(model)
    if stats is not None:
        mtl = float(stats["mean_track_length"])
        frac_ge3 = float(stats["frac_ge3"])
        n_ge3 = int(stats["n_ge3"])
        log.info(
            "COLMAP posed → %s (%s points, mean track length %.2f, "
            "%.1f%% ≥3 views (%s pts), %s cross-pano forward pairs)",
            out,
            n,
            mtl,
            100.0 * frac_ge3,
            n_ge3,
            n_pairs,
        )
        if mtl < 2.2:
            log.warning(
                "posed sparse covisibility thin (mean track length %.2f < 2.2; "
                "%.1f%% ≥3-view) — OpenMVS SelectNeighborViews may still fail; "
                "grow midframes / forward matches (target mean ≳ 2.5–3)",
                mtl,
                100.0 * frac_ge3,
            )
    else:
        log.info("COLMAP posed → %s (%s points, %s cross-pano pairs)", out, n, n_pairs)
    # Fail loud AFTER writing PLY so Studio/debug still has the thin cloud.
    if n < min_points:
        raise RuntimeError(
            f"COLMAP point_triangulator too thin ({n} points at {out}; need ≥{min_points}). "
            "SV orbit known poses + cross-pano matches under-constrain geometry. "
            "Next photo-only step: denser drive spacing / more overlap, or OpenCV SIFT/flow stereo."
        )
    return out


# ---------------------------------------------------------------------------
# External matcher support (MASt3R etc.) — keypoints/matches into COLMAP DB
# ---------------------------------------------------------------------------

MATCHERS = ("sift", "mast3r")
_MAX_NUM_IMAGES = 2147483647


def image_ids_to_pair_id(image_id1: int, image_id2: int) -> int:
    """COLMAP ``Database::ImagePairToPairId`` (id1 < id2)."""
    i1, i2 = int(image_id1), int(image_id2)
    if i1 > i2:
        i1, i2 = i2, i1
    return i1 * _MAX_NUM_IMAGES + i2


def create_empty_database(database: Path) -> Path:
    """Create an empty COLMAP SQLite DB via ``database_creator``."""
    colmap = _colmap_bin()
    database = Path(database)
    if database.exists():
        database.unlink()
    database.parent.mkdir(parents=True, exist_ok=True)
    _run([colmap, "database_creator", "--database_path", str(database)])
    return database


def bootstrap_posed_database(
    database: Path,
    frames: list[dict[str, Any]],
    image_names: list[str],
) -> dict[str, tuple[int, int]]:
    """Insert PINHOLE cameras + images so IMAGE_IDs match ``write_known_pose_model``.

    Returns ``{name: (image_id, camera_id)}`` with image_id = 1..N in list order
    (same convention as the pre-remap known-pose text model).
    """
    if len(frames) != len(image_names):
        raise ValueError("frames / image_names mismatch")
    if not frames_have_known_poses(frames):
        raise RuntimeError("posed DB bootstrap needs e/n/u/heading on every frame")

    create_empty_database(database)
    cam_key_to_id: dict[tuple[int, int, float], int] = {}
    frame_cam_ids: list[int] = []
    cam_rows: list[tuple[int, int, int, int, bytes]] = []
    for fr in frames:
        w, h = _frame_size(fr)
        fov = float(fr.get("fov") or 90.0)
        key = (w, h, round(fov, 3))
        if key not in cam_key_to_id:
            cid = len(cam_key_to_id) + 1
            cam_key_to_id[key] = cid
            fx = _focal_px(w, fov)
            params = np.asarray([fx, fx, w / 2.0, h / 2.0], dtype=np.float64)
            cam_rows.append((cid, 1, w, h, params.tobytes()))  # model 1 = PINHOLE
        frame_cam_ids.append(cam_key_to_id[key])

    con = sqlite3.connect(str(database))
    try:
        for cid, model, w, h, blob in cam_rows:
            con.execute(
                "INSERT INTO cameras(camera_id, model, width, height, params, prior_focal_length) "
                "VALUES (?, ?, ?, ?, ?, 1)",
                (cid, model, w, h, blob),
            )
        name_to_ids: dict[str, tuple[int, int]] = {}
        for idx, (name, cid) in enumerate(zip(image_names, frame_cam_ids, strict=True), start=1):
            con.execute(
                "INSERT INTO images(image_id, name, camera_id) VALUES (?, ?, ?)",
                (idx, name, cid),
            )
            name_to_ids[name] = (idx, cid)
        # Keep AUTOINCREMENT sequences consistent for later inserts.
        con.execute("DELETE FROM sqlite_sequence WHERE name='cameras'")
        con.execute(
            "INSERT INTO sqlite_sequence(name, seq) VALUES ('cameras', ?)",
            (max(cid for cid, *_ in cam_rows),),
        )
        con.execute("DELETE FROM sqlite_sequence WHERE name='images'")
        con.execute(
            "INSERT INTO sqlite_sequence(name, seq) VALUES ('images', ?)",
            (len(image_names),),
        )
        con.commit()
    finally:
        con.close()
    return name_to_ids


def write_keypoints_blob(xy: np.ndarray) -> tuple[int, int, bytes]:
    """Pack Nx2 float32 keypoints for the COLMAP ``keypoints`` table."""
    arr = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
    return int(arr.shape[0]), 2, np.ascontiguousarray(arr).tobytes()


def write_matches_blob(matches: np.ndarray) -> tuple[int, int, bytes]:
    """Pack Nx2 uint32 match indices for ``matches`` / ``two_view_geometries``."""
    arr = np.asarray(matches, dtype=np.uint32).reshape(-1, 2)
    return int(arr.shape[0]), 2, np.ascontiguousarray(arr).tobytes()


def import_keypoints_and_matches(
    database: Path,
    keypoints: dict[int, np.ndarray],
    pair_matches: dict[tuple[int, int], np.ndarray],
    *,
    skip_geometric_verification: bool = True,
) -> int:
    """Write keypoints + matches (+ optional two_view_geometries) into ``database``.

    ``keypoints`` maps image_id → Nx2 float32 xy in original image pixels.
    ``pair_matches`` maps (image_id1, image_id2) with id1 < id2 → Mx2 uint32
    keypoint indices. When ``skip_geometric_verification`` is True, also write
    ``two_view_geometries`` with config=2 (CALIBRATED) so point_triangulator
    can consume matches without a SIFT verify pass.
    """
    con = sqlite3.connect(str(database))
    n_pairs = 0
    try:
        con.execute("DELETE FROM keypoints")
        con.execute("DELETE FROM matches")
        con.execute("DELETE FROM two_view_geometries")
        for image_id, xy in keypoints.items():
            rows, cols, blob = write_keypoints_blob(xy)
            if rows == 0:
                continue
            con.execute(
                "INSERT INTO keypoints(image_id, rows, cols, data) VALUES (?, ?, ?, ?)",
                (int(image_id), rows, cols, blob),
            )
        # Match naver/mast3r COLMAPDatabase.add_two_view_geometry defaults:
        # identity F/E/H + unit qvec. All-zero blobs make COLMAP ignore pairs
        # ("connected 0" / SIGABRT on point_triangulator).
        eye3 = np.eye(3, dtype=np.float64).tobytes()
        qvec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64).tobytes()
        tvec = np.zeros(3, dtype=np.float64).tobytes()
        for (id1, id2), matches in pair_matches.items():
            i1, i2 = int(id1), int(id2)
            if i1 > i2:
                i1, i2 = i2, i1
                matches = np.asarray(matches)[:, ::-1]
            rows, cols, blob = write_matches_blob(matches)
            if rows == 0:
                continue
            pair_id = image_ids_to_pair_id(i1, i2)
            con.execute(
                "INSERT INTO matches(pair_id, rows, cols, data) VALUES (?, ?, ?, ?)",
                (pair_id, rows, cols, blob),
            )
            if skip_geometric_verification:
                con.execute(
                    "INSERT INTO two_view_geometries("
                    "pair_id, rows, cols, data, config, F, E, H, qvec, tvec) "
                    "VALUES (?, ?, ?, ?, 2, ?, ?, ?, ?, ?)",
                    (pair_id, rows, cols, blob, eye3, eye3, eye3, qvec, tvec),
                )
            n_pairs += 1
        con.commit()
    finally:
        con.close()
    return n_pairs


def point_triangulator_argv(
    colmap: str,
    database: Path,
    image_path: Path,
    input_path: Path,
    output_path: Path,
) -> list[str]:
    """Build ``point_triangulator`` argv with extrinsics locked (no BA refine)."""
    return [
        colmap,
        "point_triangulator",
        "--database_path",
        str(database),
        "--image_path",
        str(image_path),
        "--input_path",
        str(input_path),
        "--output_path",
        str(output_path),
        "--clear_points",
        "1",
        "--Mapper.tri_ignore_two_view_tracks",
        "0",
        "--Mapper.filter_min_tri_angle",
        "0.5",
        "--Mapper.ba_refine_focal_length",
        "0",
        "--Mapper.ba_refine_extra_params",
        "0",
        "--Mapper.ba_refine_principal_point",
        "0",
    ]
