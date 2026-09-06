"""Pairwise feature matching between neighbouring Street View frames."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _orb() -> cv2.ORB:
    return cv2.ORB_create(nfeatures=3500, scaleFactor=1.2, nlevels=8)


def match_pair(path_a: str | Path, path_b: str | Path) -> dict[str, Any] | None:
    img_a = cv2.imread(str(path_a), cv2.IMREAD_GRAYSCALE)
    img_b = cv2.imread(str(path_b), cv2.IMREAD_GRAYSCALE)
    if img_a is None or img_b is None:
        return None
    orb = _orb()
    kpa, dea = orb.detectAndCompute(img_a, None)
    kpb, deb = orb.detectAndCompute(img_b, None)
    if dea is None or deb is None or len(kpa) < 12 or len(kpb) < 12:
        return None
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = matcher.knnMatch(dea, deb, k=2)
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)
    if len(good) < 16:
        return None
    pts_a = np.float32([kpa[m.queryIdx].pt for m in good])
    pts_b = np.float32([kpb[m.trainIdx].pt for m in good])
    h_a, w_a = img_a.shape[:2]
    f = 0.5 * w_a  # will be overwritten by caller if they know fov
    K = np.array([[f, 0, w_a / 2.0], [0, f, h_a / 2.0], [0, 0, 1.0]], dtype=np.float64)
    E, mask = cv2.findEssentialMat(pts_a, pts_b, K, method=cv2.RANSAC, prob=0.999, threshold=1.2)
    if E is None:
        return None
    inl = int(mask.sum()) if mask is not None else 0
    if inl < 12:
        return None
    _, R, t, mask_pose = cv2.recoverPose(E, pts_a, pts_b, K, mask=mask)
    return {
        "inliers": int(mask_pose.sum()) if mask_pose is not None else inl,
        "matches": len(good),
        "R": R.tolist(),
        "t": t.reshape(-1).tolist(),
        "width": int(w_a),
        "height": int(h_a),
    }


def focal_px(width: int, fov_deg: float) -> float:
    return 0.5 * width / np.tan(np.deg2rad(fov_deg) / 2.0)
