"""Strip Street View chrome so it cannot become 3D points.

Screenshots still pick up copyright, compass, zoom, address chips and
link chevrons. Those are flat, high-contrast, and they reconstruct as
ghost geometry in the cloud — so we crop the bars, mask the widgets,
and inpaint the holes before any interpolation / triangulation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def detect_chrome_margins(image: np.ndarray) -> tuple[int, int, int, int]:
    """Return (top, right, bottom, left) pixels of likely UI chrome."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    top = _edge_bar(gray, axis=0, from_end=False)
    bottom = _edge_bar(gray, axis=0, from_end=True)
    left = _edge_bar(gray, axis=1, from_end=False)
    right = _edge_bar(gray, axis=1, from_end=True)
    # on-canvas © Google / Report-a-problem strip
    bottom = max(bottom, int(round(h * 0.06)))
    top = min(top, h // 8)
    bottom = min(bottom, h // 6)
    left = min(left, w // 10)
    right = min(right, w // 10)
    return top, right, bottom, left


def _edge_bar(gray: np.ndarray, axis: int, from_end: bool) -> int:
    if axis == 0:
        length = gray.shape[0]
        strip = lambda i: gray[i, :]
    else:
        length = gray.shape[1]
        strip = lambda i: gray[:, i]
    limit = max(4, length // 8)
    indices = range(length - 1, length - 1 - limit, -1) if from_end else range(limit)
    run = 0
    for i in indices:
        row = strip(i)
        mean = float(row.mean())
        var = float(row.var())
        dark_bar = mean < 50.0 and var < 1600.0
        light_bar = mean > 220.0 and var < 900.0
        if dark_bar or light_bar:
            run += 1
        else:
            break
    return run


def ui_mask(image: np.ndarray) -> np.ndarray:
    """255 where a pixel looks like Street View chrome, else 0."""
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    top, right, bottom, left = detect_chrome_margins(image)
    if top:
        mask[:top, :] = 255
    if bottom:
        mask[h - bottom :, :] = 255
    if left:
        mask[:, :left] = 255
    if right:
        mask[:, w - right :] = 255

    # Full Google Maps Street View chrome lives in fixed corners.
    # Do NOT Hough-circle the photograph — headlights and cobbles look like buttons.
    mask[: int(h * 0.22), : int(w * 0.48)] = 255  # search + address card
    mask[: int(h * 0.10), int(w * 0.72) :] = 255  # share / close
    mask[int(h * 0.76) :, : int(w * 0.24)] = 255  # minimap
    mask[int(h * 0.90) :, :] = 255  # copyright
    mask[int(h * 0.70) :, int(w * 0.90) :] = 255  # compass + zoom
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _mask_flat_chips(gray, mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def _mask_flat_chips(gray: np.ndarray, mask: np.ndarray) -> None:
    """Pale rounded cards (address, keyboard hint) near the frame edge."""
    h, w = gray.shape
    edge = np.zeros_like(gray)
    m = max(8, h // 12)
    edge[:m, :] = gray[:m, :]
    edge[-m:, :] = gray[-m:, :]
    edge[:, :m] = gray[:, :m]
    edge[:, -m:] = gray[:, -m:]
    pale = ((edge > 210) & (edge > 0)).astype(np.uint8) * 255
    pale = cv2.morphologyEx(pale, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(pale, connectivity=8)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if 80 < area < (w * h) * 0.08 and bw > 20 and bh > 10:
            mask[labels == i] = 255


def crop_image(
    image: np.ndarray,
    *,
    top_frac: float = 0.0,
    bottom_frac: float = 0.07,
    auto: bool = True,
) -> np.ndarray:
    y0, y1, x0, x1 = _crop_box(image, top_frac=top_frac, bottom_frac=bottom_frac, auto=auto)
    return image[y0:y1, x0:x1]


def _crop_box(
    image: np.ndarray,
    *,
    top_frac: float,
    bottom_frac: float,
    auto: bool,
) -> tuple[int, int, int, int]:
    h, w = image.shape[:2]
    top = int(round(h * top_frac))
    bottom = int(round(h * bottom_frac))
    left = right = 0
    if auto:
        at, ar, ab, al = detect_chrome_margins(image)
        top = max(top, at)
        bottom = max(bottom, ab)
        left, right = al, ar
    y0, y1 = top, h - bottom
    x0, x1 = left, w - right
    if y1 - y0 < 32 or x1 - x0 < 32:
        return 0, h, 0, w
    return y0, y1, x0, x1


def clean_frame(
    image: np.ndarray,
    *,
    top_frac: float = 0.0,
    bottom_frac: float = 0.07,
    auto: bool = True,
    inpaint: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (cleaned BGR, mask in cropped space). Mask 255 = was UI."""
    full_mask = ui_mask(image) if auto else np.zeros(image.shape[:2], dtype=np.uint8)
    y0, y1, x0, x1 = _crop_box(image, top_frac=top_frac, bottom_frac=bottom_frac, auto=auto)
    cropped = image[y0:y1, x0:x1].copy()
    mask = full_mask[y0:y1, x0:x1]
    if inpaint and int(mask.sum()) > 0:
        cropped = cv2.inpaint(cropped, mask, 4, cv2.INPAINT_TELEA)
    return cropped, mask


def crop_shots(
    shots: list[dict[str, Any]],
    dest: Path,
    *,
    top_frac: float = 0.0,
    bottom_frac: float = 0.07,
    auto: bool = True,
) -> list[dict[str, Any]]:
    dest.mkdir(parents=True, exist_ok=True)
    out: list[dict[str, Any]] = []
    for shot in shots:
        src = Path(shot["path"])
        image = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if image is None:
            continue
        cleaned, mask = clean_frame(
            image, top_frac=top_frac, bottom_frac=bottom_frac, auto=auto, inpaint=auto
        )
        target = dest / shot["pano_id"] / src.name
        target.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target), cleaned, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        mask_path = target.with_suffix(".mask.png")
        cv2.imwrite(str(mask_path), mask)
        item = dict(shot)
        item["raw_path"] = shot["path"]
        item["path"] = str(target)
        item["mask"] = str(mask_path)
        item["width"] = int(cleaned.shape[1])
        item["height"] = int(cleaned.shape[0])
        out.append(item)
    return out
