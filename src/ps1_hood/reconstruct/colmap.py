"""Write a COLMAP-compatible image set + optional COLMAP invocation.

Also documents the likely original 'video → point cloud' backends.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

COLMAP_HINT = """
If you want the heavier AI / SfM backends the original project used:

  COLMAP (classic, still the gold standard for ordered video):
    sudo apt install colmap
    ps1hood reconstruct <run> --backend colmap

  MASt3R / MASt3R-SfM  (Naver, 2024 — 'images in, point cloud out')
    https://github.com/naver/mast3r
    Point the demo at runs/<name>/interp/frames

  VGGT (Meta, 2025 — feed-forward point maps from a video):
    https://github.com/facebookresearch/vggt

  Luma / Postshot / Polycam:
    drop runs/<name>/interp/drive.mp4 into the app

FILM (Google, large-motion interpolation) is the other missing weight:
    https://github.com/google-research/frame-interpolation
"""


def export_colmap_images(frames: list[dict[str, Any]], dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    images = dest / "images"
    images.mkdir(exist_ok=True)
    for frame in frames:
        src = Path(frame["path"])
        target = images / src.name
        if not target.exists():
            shutil.copy2(src, target)
    return dest


def run_colmap(workspace: Path) -> None:
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
