"""Download a georeferenced satellite mosaic for the bbox."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ps1_hood.geo import BBox, LocalFrame
from ps1_hood.httputil import get_bytes

# Esri World Imagery — no key. Fine for personal reconstruction / alignment.
ESRI_EXPORT = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/export"
)


def fetch_satellite(bbox: BBox, dest: Path, size: int = 2048) -> dict[str, Any]:
    dest.mkdir(parents=True, exist_ok=True)
    aspect = bbox.width_m() / max(bbox.height_m(), 1e-3)
    if aspect >= 1:
        w, h = size, max(256, int(round(size / aspect)))
    else:
        h, w = size, max(256, int(round(size * aspect)))
    url = (
        f"{ESRI_EXPORT}?bbox={bbox.west},{bbox.south},{bbox.east},{bbox.north}"
        f"&bboxSR=4326&imageSR=4326&size={w},{h}&format=jpg&f=image"
    )
    image_path = dest / "ortho.jpg"
    image_path.write_bytes(get_bytes(url, timeout=60.0))
    meta = {
        "path": str(image_path),
        "width": w,
        "height": h,
        "bbox": bbox.as_dict(),
        "provider": "esri_world_imagery",
        "crs": "EPSG:4326",
    }
    return meta


class Ortho:
    """Pixel <-> ENU lookup for the satellite mosaic."""

    def __init__(self, image: np.ndarray, bbox: BBox, frame: LocalFrame):
        self.image = image
        self.bbox = bbox
        self.frame = frame
        self.h, self.w = image.shape[:2]
        self.sw, self.sh = frame.to_enu(bbox.south, bbox.west)[:2]
        self.ne = frame.to_enu(bbox.north, bbox.east)[:2]
        self.ee, self.nn = self.ne

    @classmethod
    def load(cls, meta: dict[str, Any], frame: LocalFrame) -> "Ortho":
        image = cv2.imread(meta["path"], cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(meta["path"])
        return cls(image, BBox.from_dict(meta["bbox"]), frame)

    def enu_to_px(self, e: float, n: float) -> tuple[float, float]:
        u = (e - self.sw) / max(self.ee - self.sw, 1e-6) * (self.w - 1)
        v = (1.0 - (n - self.sh) / max(self.nn - self.sh, 1e-6)) * (self.h - 1)
        return u, v

    def sample(self, e: float, n: float) -> np.ndarray | None:
        u, v = self.enu_to_px(e, n)
        if u < 0 or v < 0 or u >= self.w - 1 or v >= self.h - 1:
            return None
        return self.image[int(round(v)), int(round(u))]

    def crop_window(self, e: float, n: float, half_m: float) -> tuple[np.ndarray, dict]:
        u0, v0 = self.enu_to_px(e - half_m, n + half_m)
        u1, v1 = self.enu_to_px(e + half_m, n - half_m)
        x0, x1 = int(max(0, min(u0, u1))), int(min(self.w, max(u0, u1)))
        y0, y1 = int(max(0, min(v0, v1))), int(min(self.h, max(v0, v1)))
        crop = self.image[y0:y1, x0:x1]
        return crop, {"x0": x0, "y0": y0, "x1": x1, "y1": y1}
