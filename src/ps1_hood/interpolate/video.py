"""Pack interpolated frames into a drive video for 3D backends that want video."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2


def write_video(frames: list[dict[str, Any]], dest: Path, fps: int = 12) -> Path | None:
    if not frames:
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        pattern = Path(frames[0]["path"]).parent / "%05d.jpg"
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-framerate",
                str(fps),
                "-i",
                str(pattern),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "18",
                str(dest),
            ],
            check=False,
            capture_output=True,
        )
        if dest.exists() and dest.stat().st_size > 0:
            return dest
    first = cv2.imread(frames[0]["path"])
    if first is None:
        return None
    h, w = first.shape[:2]
    avi = dest.with_suffix(".avi")
    writer = cv2.VideoWriter(str(avi), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    if not writer.isOpened():
        return None
    for frame in frames:
        img = cv2.imread(frame["path"])
        if img is None:
            continue
        if img.shape[0] != h or img.shape[1] != w:
            img = cv2.resize(img, (w, h))
        writer.write(img)
    writer.release()
    return avi if avi.exists() else None
