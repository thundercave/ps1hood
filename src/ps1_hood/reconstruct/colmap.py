"""Write a COLMAP-compatible image set + optional COLMAP invocation.

Also documents the likely original 'video → point cloud' backends.
"""

from __future__ import annotations

import logging
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

COLMAP_HINT = """
Reconstruction prefers aligned Street View keyframes (align/cameras.json),
not DIS-interpolated midframes.

Heavier AI / SfM backends:

  COLMAP (classic, still the gold standard for ordered video):
    sudo apt install colmap
    ps1hood reconstruct <run> --backend colmap

  MASt3R / MASt3R-SfM  (Naver, 2024 — 'images in, point cloud out')
    https://github.com/naver/mast3r
    Needs a GPU + torch + downloaded weights (optional extra; not default deps).
    ps1hood reconstruct <run> --backend mast3r
    (exports keyframes to runs/<name>/recon/mast3r_input/images if missing)

  VGGT (Meta, 2025 — feed-forward point maps from a video):
    https://github.com/facebookresearch/vggt

  Luma / Postshot / Polycam:
    drop runs/<name>/interp/drive.mp4 into the app

FILM (Google, large-motion interpolation) is the other missing weight:
    https://github.com/google-research/frame-interpolation
"""

_MIN_POINTS3D_BYTES = 1024


def export_colmap_images(frames: list[dict[str, Any]], dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    images = dest / "images"
    images.mkdir(exist_ok=True)
    used: set[str] = set()
    for i, frame in enumerate(frames):
        src = Path(frame["path"])
        # Keyframes live under cropped/<pano>/hXXX_pY.jpg — bare basenames collide.
        pano = frame.get("pano_id") or src.parent.name
        candidate = f"{pano}_{src.name}"
        if candidate in used:
            candidate = f"{i:05d}_{pano}_{src.name}"
        used.add(candidate)
        target = images / candidate
        if not target.exists():
            shutil.copy2(src, target)
    return dest


def _points3d_count(path: Path) -> int:
    """Number of points in a COLMAP points3D.bin (little-endian uint64 header)."""
    raw = path.read_bytes()
    if len(raw) < 8:
        return 0
    return int(struct.unpack("<Q", raw[:8])[0])


def find_sparse_model(sparse: Path) -> Path | None:
    """Return the first sparse/*/ directory that has a points3D.bin."""
    if not sparse.is_dir():
        return None
    # Prefer numeric model ids (0, 1, …) then any other child.
    children = sorted(
        (p for p in sparse.iterdir() if p.is_dir()),
        key=lambda p: (not p.name.isdigit(), int(p.name) if p.name.isdigit() else p.name),
    )
    for child in children:
        if (child / "points3D.bin").is_file():
            return child
    # Some COLMAP builds write directly into sparse/
    if (sparse / "points3D.bin").is_file():
        return sparse
    return None


def _validate_sparse_model(model: Path) -> int:
    points = model / "points3D.bin"
    if not points.is_file():
        raise RuntimeError(
            f"COLMAP mapper wrote no points3D.bin under {model}. "
            "SV orbits often lack COLMAP init pairs; use flow/mast3r/known poses."
        )
    size = points.stat().st_size
    n = _points3d_count(points)
    if size < _MIN_POINTS3D_BYTES or n == 0:
        raise RuntimeError(
            f"COLMAP sparse model is empty/tiny ({n} points, {size} bytes at {points}). "
            "SV orbits often lack COLMAP init pairs; use flow/mast3r/known poses "
            "instead of classic COLMAP without seeded poses."
        )
    return n


def run_colmap(workspace: Path, *, ply_out: Path | None = None) -> Path:
    """Run COLMAP SfM and convert a valid sparse model to PLY.

    Returns the path to the written PLY (``ply_out`` or
    ``workspace.parent / cloud_colmap.ply``).
    """
    colmap = shutil.which("colmap")
    if not colmap:
        raise RuntimeError("colmap is not on PATH." + COLMAP_HINT)
    db = workspace / "database.db"
    images = workspace / "images"
    sparse = workspace / "sparse"
    sparse.mkdir(exist_ok=True)
    subprocess.run(
        [colmap, "feature_extractor", "--database_path", str(db), "--image_path", str(images)],
        check=True,
    )
    subprocess.run([colmap, "exhaustive_matcher", "--database_path", str(db)], check=True)
    subprocess.run(
        [
            colmap,
            "mapper",
            "--database_path",
            str(db),
            "--image_path",
            str(images),
            "--output_path",
            str(sparse),
        ],
        check=True,
    )

    model = find_sparse_model(sparse)
    if model is None:
        raise RuntimeError(
            f"COLMAP mapper produced no sparse/*/points3D.bin under {sparse}. "
            "SV orbits often lack COLMAP init pairs; use flow/mast3r/known poses."
        )
    n = _validate_sparse_model(model)
    out = ply_out if ply_out is not None else workspace.parent / "cloud_colmap.ply"
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            colmap,
            "model_converter",
            "--input_path",
            str(model),
            "--output_path",
            str(out),
            "--output_type",
            "PLY",
        ],
        check=True,
    )
    if not out.is_file() or out.stat().st_size < 64:
        raise RuntimeError(f"COLMAP model_converter failed to write a usable PLY at {out}")
    log.info("COLMAP sparse model %s → %s (%s points)", model, out, n)
    return out
