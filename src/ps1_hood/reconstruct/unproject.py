"""Dense point cloud from interpolated frames via flow triangulation.

This is the built-in stand-in for the 'AI that makes a point cloud from
video' step (MASt3R / VGGT / COLMAP / Luma / Postshot). Those still win
on quality — this gets a coloured street-scale cloud on the laptop.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ps1_hood.geo import camera_rotation_cv
from ps1_hood.reconstruct.known_pose import select_stereo_pairs, sequential_pairs


def _load_mask(path: str | None, height: int, width: int) -> np.ndarray | None:
    if not path:
        return None
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    if mask.shape[0] != height or mask.shape[1] != width:
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask


def _K(width: int, height: int, fov_deg: float) -> np.ndarray:
    f = 0.5 * width / math.tan(math.radians(fov_deg) / 2.0)
    return np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1.0]], dtype=np.float64)


def _Rt(pose: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    R = np.array(camera_rotation_cv(pose["heading"], pose["pitch"]), dtype=np.float64)
    C = np.array([pose["e"], pose["n"], pose["u"]], dtype=np.float64)
    # world-from-camera R, camera centre C → OpenCV world-to-camera
    Rcw = R.T
    t = -Rcw @ C
    return Rcw, t


def _P(pose: dict[str, Any], width: int, height: int) -> np.ndarray:
    K = _K(width, height, float(pose["fov"]))
    Rcw, t = _Rt(pose)
    return K @ np.hstack([Rcw, t.reshape(3, 1)])


def _cam_center(pose: dict[str, Any]) -> np.ndarray:
    return np.array([pose["e"], pose["n"], pose["u"]], dtype=np.float64)


def _forward(pose: dict[str, Any]) -> np.ndarray:
    return np.array(camera_rotation_cv(pose["heading"], pose["pitch"]), dtype=np.float64)[:, 2]


def _reproj_err(
    P: np.ndarray, pts_w: np.ndarray, pts_img: np.ndarray
) -> np.ndarray:
    """Per-point pixel reprojection error."""
    hom = np.hstack([pts_w, np.ones((len(pts_w), 1), dtype=np.float64)])
    proj = (P @ hom.T).T
    denom = np.clip(proj[:, 2:3], 1e-8, None)
    uv = proj[:, :2] / denom
    return np.linalg.norm(uv - pts_img.T, axis=1)



def _cross_pano_pairs(frames: list[dict[str, Any]], pair_step: int = 2) -> list[tuple[int, int]]:
    """Drive-adjacent pairs only — skip same-center orbit mates (pure rotation)."""
    import math

    def pano_key(f: dict[str, Any]) -> str:
        if f.get("pano_id"):
            return str(f["pano_id"])
        return f"{round(float(f.get('e', 0)), 2)}_{round(float(f.get('n', 0)), 2)}"

    if all("e" in f and "n" in f and "heading" in f for f in frames):
        pairs = select_stereo_pairs(frames)
    else:
        pairs = sequential_pairs(len(frames), pair_step=pair_step)
    out: list[tuple[int, int]] = []
    for i, j in pairs:
        if pano_key(frames[i]) == pano_key(frames[j]):
            continue
        be = abs(float(frames[i]["e"]) - float(frames[j]["e"]))
        bn = abs(float(frames[i]["n"]) - float(frames[j]["n"]))
        if math.hypot(be, bn) < 1.5:
            continue
        out.append((i, j))
    return out


def triangulate_sift_frames(
    frames: list[dict[str, Any]],
    dest_ply: Path,
    *,
    max_points: int = 800_000,
    max_reproj_px: float = 8.0,
) -> dict[str, Any]:
    """OpenCV SIFT stereo on cross-pano known-pose pairs (COLMAP-thin fallback)."""
    if len(frames) < 2:
        raise RuntimeError("need ≥2 frames for SIFT stereo")
    sample = cv2.imread(frames[0]["path"], cv2.IMREAD_COLOR)
    if sample is None:
        raise RuntimeError("could not read first frame")
    h, w = sample.shape[:2]
    pairs = _cross_pano_pairs(frames)
    if not pairs:
        raise RuntimeError("no cross-pano pairs for SIFT stereo")

    sift = cv2.SIFT_create(nfeatures=4000)
    bf = cv2.BFMatcher(cv2.NORM_L2)

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    zs: list[np.ndarray] = []
    cs: list[np.ndarray] = []
    total = 0
    used_pairs = 0
    for i, j in pairs:
        a, b = frames[i], frames[j]
        ia = cv2.imread(a["path"], cv2.IMREAD_COLOR)
        ib = cv2.imread(b["path"], cv2.IMREAD_COLOR)
        if ia is None or ib is None:
            continue
        if ia.shape[:2] != (h, w):
            ia = cv2.resize(ia, (w, h))
        if ib.shape[:2] != (h, w):
            ib = cv2.resize(ib, (w, h))
        ga = cv2.cvtColor(ia, cv2.COLOR_BGR2GRAY)
        gb = cv2.cvtColor(ib, cv2.COLOR_BGR2GRAY)
        kpa, desa = sift.detectAndCompute(ga, None)
        kpb, desb = sift.detectAndCompute(gb, None)
        if desa is None or desb is None or len(kpa) < 30 or len(kpb) < 30:
            continue
        knn = bf.knnMatch(desa, desb, k=2)
        good = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)
        if len(good) < 40:
            continue
        pts1 = np.array([kpa[m.queryIdx].pt for m in good], dtype=np.float64).T
        pts2 = np.array([kpb[m.trainIdx].pt for m in good], dtype=np.float64).T
        ma = _load_mask(a.get("mask"), h, w)
        mb = _load_mask(b.get("mask"), h, w)
        keep = np.ones(pts1.shape[1], dtype=bool)
        if ma is not None:
            xi = np.clip(np.round(pts1[0]).astype(int), 0, w - 1)
            yi = np.clip(np.round(pts1[1]).astype(int), 0, h - 1)
            keep &= ma[yi, xi] == 0
        if mb is not None:
            xi = np.clip(np.round(pts2[0]).astype(int), 0, w - 1)
            yi = np.clip(np.round(pts2[1]).astype(int), 0, h - 1)
            keep &= mb[yi, xi] == 0
        if int(keep.sum()) < 30:
            continue
        pts1 = pts1[:, keep]
        pts2 = pts2[:, keep]
        P1 = _P(a, w, h)
        P2 = _P(b, w, h)
        hom = cv2.triangulatePoints(P1, P2, pts1, pts2)
        pts = (hom[:3] / np.clip(hom[3], 1e-8, None)).T
        cam_a = _cam_center(a)
        cam_b = _cam_center(b)
        fwd_a = _forward(a)
        fwd_b = _forward(b)
        vis_a = (pts - cam_a) @ fwd_a
        vis_b = (pts - cam_b) @ fwd_b
        z = pts[:, 2]
        ok = (vis_a > 0.4) & (vis_b > 0.2) & (z > -1.5) & (z < 40.0)
        dist = np.linalg.norm(pts - cam_a, axis=1)
        ok &= dist < 60.0
        err1 = _reproj_err(P1, pts, pts1)
        err2 = _reproj_err(P2, pts, pts2)
        ok &= (err1 < max_reproj_px) & (err2 < max_reproj_px)
        if int(ok.sum()) == 0:
            continue
        pts_ok = pts[ok]
        xi = np.clip(np.round(pts1[0, ok]).astype(int), 0, w - 1)
        yi = np.clip(np.round(pts1[1, ok]).astype(int), 0, h - 1)
        colors = ia[yi, xi]
        xs.append(pts_ok[:, 0])
        ys.append(pts_ok[:, 1])
        zs.append(pts_ok[:, 2])
        cs.append(colors)
        total += len(pts_ok)
        used_pairs += 1
        if total >= max_points:
            break
    if not xs:
        raise RuntimeError("SIFT triangulation produced no points")
    xyz = np.stack([np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)], axis=1)
    rgb = np.concatenate(cs, axis=0)
    xyz, rgb = _voxel_downsample(xyz, rgb, 0.12)
    dest_ply.parent.mkdir(parents=True, exist_ok=True)
    write_ply(dest_ply, xyz, rgb)
    return {
        "path": str(dest_ply),
        "points": int(len(xyz)),
        "pairs": used_pairs,
        "source": "sift",
    }


def triangulate_frames(
    frames: list[dict[str, Any]],
    dest_ply: Path,
    *,
    stride: int = 3,
    pair_step: int = 2,
    max_points: int = 1_200_000,
    use_pose_pairs: bool = True,
    max_reproj_px: float = 12.0,
) -> dict[str, Any]:
    if len(frames) < 2:
        raise RuntimeError("need at least 2 interpolated frames to triangulate")
    sample = cv2.imread(frames[0]["path"], cv2.IMREAD_COLOR)
    if sample is None:
        raise RuntimeError("could not read first frame")
    h, w = sample.shape[:2]
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)

    if use_pose_pairs and all("e" in f and "n" in f and "heading" in f for f in frames):
        pairs = _cross_pano_pairs(frames, pair_step=pair_step)
        if not pairs:
            # last resort: pose pairs without cross-pano filter
            pairs = select_stereo_pairs(frames)
        if not pairs:
            pairs = sequential_pairs(len(frames), pair_step=pair_step)
    else:
        pairs = sequential_pairs(len(frames), pair_step=pair_step)

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    zs: list[np.ndarray] = []
    cs: list[np.ndarray] = []
    total = 0
    for i, j in pairs:
        a, b = frames[i], frames[j]
        ia = cv2.imread(a["path"], cv2.IMREAD_COLOR)
        ib = cv2.imread(b["path"], cv2.IMREAD_COLOR)
        if ia is None or ib is None:
            continue
        if ia.shape[:2] != (h, w):
            ia = cv2.resize(ia, (w, h))
        if ib.shape[:2] != (h, w):
            ib = cv2.resize(ib, (w, h))
        ga = cv2.cvtColor(ia, cv2.COLOR_BGR2GRAY)
        gb = cv2.cvtColor(ib, cv2.COLOR_BGR2GRAY)
        flow = dis.calc(ga, gb, None)
        uu, vv = np.meshgrid(np.arange(0, w, stride), np.arange(0, h, stride))
        fu = flow[vv, uu, 0]
        fv = flow[vv, uu, 1]
        pts1 = np.vstack([uu.ravel(), vv.ravel()]).astype(np.float64)
        pts2 = np.vstack([uu.ravel() + fu.ravel(), vv.ravel() + fv.ravel()]).astype(np.float64)
        # drop tiny / huge flow (static sky or bad matches)
        mag = np.hypot(fu.ravel(), fv.ravel())
        keep = (mag > 0.8) & (mag < 80.0)
        # never lift leftover Street View chrome into the cloud
        ma = _load_mask(a.get("mask"), h, w)
        mb = _load_mask(b.get("mask"), h, w)
        if ma is not None:
            keep &= ma[vv.ravel(), uu.ravel()] == 0
        if mb is not None:
            bx = np.clip(np.round(pts2[0]), 0, w - 1).astype(int)
            by = np.clip(np.round(pts2[1]), 0, h - 1).astype(int)
            keep &= mb[by, bx] == 0
        # second view must land inside the image
        keep &= (
            (pts2[0] >= 0)
            & (pts2[0] < w)
            & (pts2[1] >= 0)
            & (pts2[1] < h)
        )
        if int(keep.sum()) < 80:
            continue
        pts1 = pts1[:, keep]
        pts2 = pts2[:, keep]
        P1 = _P(a, w, h)
        P2 = _P(b, w, h)
        hom = cv2.triangulatePoints(P1, P2, pts1, pts2)
        pts = (hom[:3] / np.clip(hom[3], 1e-8, None)).T
        # cheirality: in front of both cameras
        cam_a = _cam_center(a)
        cam_b = _cam_center(b)
        fwd_a = _forward(a)
        fwd_b = _forward(b)
        vis_a = (pts - cam_a) @ fwd_a
        vis_b = (pts - cam_b) @ fwd_b
        z = pts[:, 2]
        ok = (vis_a > 0.4) & (vis_b > 0.2) & (z > -1.5) & (z < 40.0)
        dist = np.linalg.norm(pts - cam_a, axis=1)
        ok &= dist < 60.0
        if int(ok.sum()) == 0:
            continue
        # reprojection sanity on surviving points
        err1 = _reproj_err(P1, pts, pts1)
        err2 = _reproj_err(P2, pts, pts2)
        ok &= (err1 < max_reproj_px) & (err2 < max_reproj_px)
        pts = pts[ok]
        if pts.size == 0:
            continue
        colors = ia[vv.ravel()[keep][ok], uu.ravel()[keep][ok]]
        xs.append(pts[:, 0])
        ys.append(pts[:, 1])
        zs.append(pts[:, 2])
        cs.append(colors)
        total += len(pts)
        if total >= max_points:
            break
    if not xs:
        raise RuntimeError("triangulation produced no points — try a denser capture")
    xyz = np.stack([np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)], axis=1)
    rgb = np.concatenate(cs, axis=0)
    xyz, rgb = _voxel_downsample(xyz, rgb, 0.12)
    dest_ply.parent.mkdir(parents=True, exist_ok=True)
    write_ply(dest_ply, xyz, rgb)
    return {"path": str(dest_ply), "points": int(len(xyz)), "pairs": len(pairs)}


def _voxel_downsample(
    xyz: np.ndarray, rgb: np.ndarray, voxel: float
) -> tuple[np.ndarray, np.ndarray]:
    keys = np.floor(xyz / voxel).astype(np.int64)
    # structured array for unique rows
    view = np.ascontiguousarray(keys).view([("", keys.dtype)] * 3).ravel()
    _, idx = np.unique(view, return_index=True)
    return xyz[idx], rgb[idx]


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    n = len(xyz)
    header = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with path.open("w", encoding="ascii") as fh:
        fh.write(header)
        for (x, y, z), (b, g, r) in zip(xyz, rgb, strict=False):
            fh.write(f"{x:.4f} {y:.4f} {z:.4f} {int(r)} {int(g)} {int(b)}\n")
