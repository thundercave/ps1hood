"""Optical-flow frame interpolation (always-on backend).

The original project almost certainly used FILM (Google, large-motion)
or RIFE between neighbouring Street Views. Those are optional extras;
OpenCV DIS is the built-in fallback that needs no extra weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ps1_hood.interpolate.sequence import lerp_pose, order_track


def _dis() -> cv2.DISOpticalFlow:
    return cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)


def interpolate_pair(img_a: np.ndarray, img_b: np.ndarray, steps: int) -> list[np.ndarray]:
    if img_a.shape != img_b.shape:
        img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]))
    ga = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
    dis = _dis()
    fwd = dis.calc(ga, gb, None)
    bwd = dis.calc(gb, ga, None)
    h, w = ga.shape
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    frames = [img_a]
    for s in range(1, steps + 1):
        t = s / (steps + 1)
        map_fx = (grid_x + fwd[..., 0] * t).astype(np.float32)
        map_fy = (grid_y + fwd[..., 1] * t).astype(np.float32)
        map_bx = (grid_x + bwd[..., 0] * (1.0 - t)).astype(np.float32)
        map_by = (grid_y + bwd[..., 1] * (1.0 - t)).astype(np.float32)
        wa = cv2.remap(img_a, map_fx, map_fy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        wb = cv2.remap(img_b, map_bx, map_by, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        blend = cv2.addWeighted(wa, 1.0 - t, wb, t, 0)
        frames.append(blend)
    frames.append(img_b)
    return frames


def interpolate_track(
    poses: list[dict[str, Any]],
    dest: Path,
    steps: int,
) -> list[dict[str, Any]]:
    track = order_track(poses)
    dest.mkdir(parents=True, exist_ok=True)
    frames_meta: list[dict[str, Any]] = []
    idx = 0
    prev_img = None
    for k, pose in enumerate(track):
        img = cv2.imread(pose["shot_path"], cv2.IMREAD_COLOR)
        if img is None:
            continue
        if prev_img is None:
            path = dest / f"{idx:05d}.jpg"
            cv2.imwrite(str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            frames_meta.append({"index": idx, "path": str(path), **_pose_fields(pose)})
            idx += 1
            prev_img = img
            prev_pose = pose
            continue
        mid = interpolate_pair(prev_img, img, steps)
        # skip first (already written)
        for s, frame in enumerate(mid[1:], start=1):
            t = s / (len(mid) - 1)
            lp = lerp_pose(prev_pose, pose, t)
            path = dest / f"{idx:05d}.jpg"
            cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            frames_meta.append({"index": idx, "path": str(path), **lp, "interpolated": s != len(mid) - 1})
            idx += 1
        prev_img = img
        prev_pose = pose
        if k == 0:
            pass
    return frames_meta


def _pose_fields(pose: dict[str, Any]) -> dict[str, Any]:
    return {
        "e": pose["e"],
        "n": pose["n"],
        "u": pose["u"],
        "heading": pose["heading"],
        "pitch": pose["pitch"],
        "fov": pose["fov"],
        "width": pose.get("width"),
        "height": pose.get("height"),
        "interpolated": False,
        "pano_id": pose.get("pano_id"),
    }
