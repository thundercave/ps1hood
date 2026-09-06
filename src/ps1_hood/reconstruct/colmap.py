"""COLMAP export + known-pose triangulator (primary) / classic mapper (legacy).

Known-pose path for Street View orbits:
  export images → write cameras.txt/images.txt/empty points3D.txt →
  feature_extractor → **remap IMAGE_ID to database** → cross-pano matches →
  point_triangulator (NOT mapper).

See https://colmap.github.io/faq.html#reconstruct-sparse-dense-model-from-known-camera-poses
and colmap#497 (IMAGE_ID must match database or you get empty clouds).
"""

from __future__ import annotations

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
from ps1_hood.reconstruct.known_pose import select_stereo_pairs

log = logging.getLogger(__name__)

COLMAP_HINT = """
Reconstruction prefers aligned Street View keyframes (align/cameras.json).

  COLMAP known poses (recommended for SV orbits — uses point_triangulator):
    ps1hood reconstruct <run> --backend colmap_posed
    (--backend colmap also prefers posed when cameras.json poses exist)

  Classic mapper often fails to init on SV orbits; do not retry it.

  MASt3R / flow / OpenCV SIFT stereo are the photo fallbacks.
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


def cross_pano_pair_indices(frames: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """Stereo pairs with real baseline — never same-center / same-pano orbit mates."""
    pairs = select_stereo_pairs(frames)
    out: list[tuple[int, int]] = []
    for i, j in pairs:
        if _pano_key(frames[i]) == _pano_key(frames[j]):
            continue
        # Extra guard: near-zero XY baseline is pure rotation.
        be = abs(float(frames[i]["e"]) - float(frames[j]["e"]))
        bn = abs(float(frames[i]["n"]) - float(frames[j]["n"]))
        if math.hypot(be, bn) < 1.5:
            continue
        out.append((i, j))
    return out


def write_known_pose_model(
    frames: list[dict[str, Any]],
    image_names: list[str],
    model_dir: Path,
    *,
    image_ids: list[int] | None = None,
    camera_ids: list[int] | None = None,
) -> Path:
    """Write COLMAP text model; empty points3D.txt. PINHOLE, t = -R @ C."""
    if len(frames) != len(image_names):
        raise ValueError("frames and image_names length mismatch")
    if not frames_have_known_poses(frames):
        raise RuntimeError("known-pose COLMAP needs e/n/u/heading on every frame")

    model_dir.mkdir(parents=True, exist_ok=True)

    # One PINHOLE camera per (w, h, fov); override ids when remapping to DB.
    cam_key_to_id: dict[tuple[int, int, float], int] = {}
    cam_defs: list[tuple[int, int, int, float, float, float, float]] = []
    # id, w, h, fx, fy, cx, cy
    frame_cam_ids: list[int] = []
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
        # Prefer per-image DB camera_id (usually all 1 with single_camera).
        frame_cam_ids = [int(c) for c in camera_ids]
        # Rebuild cam_defs from unique DB camera ids using first matching frame.
        seen: dict[int, tuple[int, int, int, float, float, float, float]] = {}
        for fr, cid in zip(frames, frame_cam_ids, strict=True):
            if cid in seen:
                continue
            w, h = _frame_size(fr)
            fov = float(fr.get("fov") or 90.0)
            fx = _focal_px(w, fov)
            seen[cid] = (cid, w, h, fx, fx, w / 2.0, h / 2.0)
        cam_defs = list(seen.values())

    cameras_txt = model_dir / "cameras.txt"
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


def remap_known_pose_model_to_db(
    model_dir: Path,
    frames: list[dict[str, Any]],
    image_names: list[str],
    database: Path,
    *,
    max_missing_frac: float = 0.10,
    min_images: int = 4,
) -> tuple[Path, list[dict[str, Any]], list[str]]:
    """Rewrite images.txt IMAGE_IDs to match feature_extractor DB (colmap#497).

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
    camera_ids_raw = [db_map[n][1] for n in kept_names]
    # One PINHOLE per (w,h,fov) keeps intrinsics honest when sizes differ.
    # Only pin DB camera_ids when feature_extractor used a single shared camera.
    camera_ids = camera_ids_raw if len(set(camera_ids_raw)) == 1 else None
    path = write_known_pose_model(
        kept_frames, kept_names, model_dir, image_ids=image_ids, camera_ids=camera_ids
    )
    return path, kept_frames, kept_names


def write_cross_pano_match_list(
    frames: list[dict[str, Any]],
    image_names: list[str],
    path: Path,
) -> int:
    """Write image-name pairs for matches_importer (cross-pano only)."""
    pairs = cross_pano_pair_indices(frames)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as fh:
        for i, j in pairs:
            fh.write(f"{image_names[i]} {image_names[j]}\n")
    log.info("COLMAP match list: %s cross-pano pairs → %s", len(pairs), path)
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
        camera_ids_raw = [db_map[n][1] for n in kept_names]
        camera_ids = camera_ids_raw if len(set(camera_ids_raw)) == 1 else None
        write_known_pose_model(
            kept_frames, kept_names, model_dir, image_ids=image_ids, camera_ids=camera_ids
        )
    return kept_frames, kept_names


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
            "--SiftExtraction.use_gpu",
            "0",
        ]
    )
    _run(
        [
            colmap,
            "exhaustive_matcher",
            "--database_path",
            str(db),
            "--SiftMatching.use_gpu",
            "0",
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
) -> list[str]:
    """Build ``colmap feature_extractor`` argv for the posed path.

    When crop sizes differ, do **not** force ``single_camera`` + global
    ``camera_params`` (COLMAP then silently omits odd-sized images). Prefer
    per-image cameras so intrinsics stay honest for known-pose triangulation;
    ``write_known_pose_model`` already emits one PINHOLE per (w, h, fov).
    """
    cmd = [
        colmap,
        "feature_extractor",
        "--database_path",
        str(database),
        "--image_path",
        str(image_path),
        "--ImageReader.camera_model",
        "PINHOLE",
        "--SiftExtraction.use_gpu",
        "0",
    ]
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
) -> Path:
    """Triangulate with known poses via point_triangulator (never mapper).

    Critical steps for SV orbits:
    - PINHOLE from crop FoV; t = -R @ C
    - Remap IMAGE_ID to database after feature_extractor (colmap#497)
    - Match cross-pano pairs only (same-center headings = pure rotation)
    - Triangulator: clear_points, allow two-view tracks, low min tri angle
    - No PatchMatch / dense (needs GPU)
    """
    colmap = _colmap_bin()
    images = workspace / "images"
    if not images.is_dir() or not any(images.iterdir()):
        raise RuntimeError(f"no images under {images}; export_colmap_images first")
    if len(frames) != len(image_names):
        raise ValueError("frames / image_names mismatch")

    pair_count = len(cross_pano_pair_indices(frames))
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

    _run(feature_extractor_argv(colmap, db, images, frames))

    # colmap#497: IMAGE_ID in images.txt MUST match database after extraction.
    # May drop a minority of images COLMAP omitted (odd crop sizes).
    _, frames, image_names = remap_known_pose_model_to_db(
        sparse_prior, frames, image_names, db
    )

    match_list = workspace / "cross_pano_pairs.txt"
    n_pairs = write_cross_pano_match_list(frames, image_names, match_list)
    if n_pairs < 1:
        raise RuntimeError("cross-pano match list empty")

    _run(
        [
            colmap,
            "matches_importer",
            "--database_path",
            str(db),
            "--match_list_path",
            str(match_list),
            "--match_type",
            "pairs",
            "--SiftMatching.use_gpu",
            "0",
        ]
    )

    # Images with only failed geometric verification can SIGABRT point_triangulator.
    frames, image_names = filter_frames_registered_in_matches(
        frames, image_names, db, model_dir=sparse_prior
    )

    sparse_out = workspace / "sparse_posed"
    if sparse_out.exists():
        shutil.rmtree(sparse_out)
    sparse_out.mkdir(parents=True, exist_ok=True)

    _run(
        [
            colmap,
            "point_triangulator",
            "--database_path",
            str(db),
            "--image_path",
            str(images),
            "--input_path",
            str(sparse_prior),
            "--output_path",
            str(sparse_out),
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
    )

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
    log.info("COLMAP posed → %s (%s points, %s cross-pano pairs)", out, n, n_pairs)
    # Fail loud AFTER writing PLY so Studio/debug still has the thin cloud.
    if n < min_points:
        raise RuntimeError(
            f"COLMAP point_triangulator too thin ({n} points at {out}; need ≥{min_points}). "
            "SV orbit known poses + cross-pano matches under-constrain geometry. "
            "Next photo-only step: denser drive spacing / more overlap, or OpenCV SIFT/flow stereo."
        )
    return out
