"""Environment + project settings."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from ps1_hood.geo import BBox

SOURCES = ("google_web", "google_js", "google_static", "mapillary")
INTERP_BACKENDS = ("flow", "rife", "film")
RECON_BACKENDS = ("flow", "colmap", "mast3r", "export")


def load_dotenv(path: Path | None = None) -> None:
    """Tiny .env loader so we do not need python-dotenv."""
    candidates = []
    if path:
        candidates.append(path)
    candidates.append(Path.cwd() / ".env")
    here = Path(__file__).resolve()
    candidates.append(here.parents[2] / ".env")
    for cand in candidates:
        if not cand.is_file():
            continue
        for raw in cand.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            os.environ.setdefault(key, value)
        break


@dataclass
class Settings:
    google_maps_api_key: str = ""
    mapillary_token: str = ""
    source: str = "google_web"
    camera_height_m: float = 2.5
    spacing_m: float = 8.0
    fov_deg: float = 90.0
    image_size: tuple[int, int] = (640, 640)
    js_size: tuple[int, int] = (1920, 1080)
    headings_rel: tuple[int, ...] = (0, 90, 180, 270)
    extra_pitches: tuple[int, ...] = (-30, 0, 18)
    crop_bottom_frac: float = 0.07
    crop_top_frac: float = 0.0
    interp_steps: int = 8
    max_align_shift_m: float = 8.0
    max_align_heading_deg: float = 15.0

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        source = os.environ.get("PS1HOOD_SOURCE", "google_web").strip() or "google_web"
        if source not in SOURCES:
            raise ValueError(f"unknown PS1HOOD_SOURCE {source!r}; expected one of {SOURCES}")
        return cls(
            google_maps_api_key=os.environ.get("GOOGLE_MAPS_API_KEY", "").strip(),
            mapillary_token=os.environ.get("MAPILLARY_TOKEN", "").strip(),
            source=source,
        )


@dataclass
class ProjectSpec:
    name: str
    bbox: BBox
    source: str = "google_web"
    spacing_m: float = 8.0
    camera_height_m: float = 2.5
    fov_deg: float = 90.0
    heading_step: int = 45
    headings_rel: list[int] = field(default_factory=lambda: [0, 90, 180, 270])
    extra_pitches: list[int] = field(default_factory=lambda: [-30, 0, 18])
    interp_steps: int = 8
    interp_backend: str = "flow"
    recon_backend: str = "flow"
    max_panos: int | None = None

    def to_dict(self) -> dict:
        out = {
            "name": self.name,
            "bbox": self.bbox.as_dict(),
            "source": self.source,
            "spacing_m": self.spacing_m,
            "camera_height_m": self.camera_height_m,
            "fov_deg": self.fov_deg,
            "heading_step": self.heading_step,
            "headings_rel": list(self.headings_rel),
            "extra_pitches": list(self.extra_pitches),
            "interp_steps": self.interp_steps,
            "interp_backend": self.interp_backend,
            "recon_backend": self.recon_backend,
        }
        if self.max_panos is not None:
            out["max_panos"] = int(self.max_panos)
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "ProjectSpec":
        pitches_raw = data.get("extra_pitches", data.get("pitches", [-30, 0, 18]))
        max_raw = data.get("max_panos")
        return cls(
            name=str(data["name"]),
            bbox=BBox.from_dict(data["bbox"]),
            source=str(data.get("source", "google_web")),
            spacing_m=float(data.get("spacing_m", 8.0)),
            camera_height_m=float(data.get("camera_height_m", 2.5)),
            fov_deg=float(data.get("fov_deg", 90.0)),
            heading_step=int(data.get("heading_step", 45)),
            headings_rel=[int(x) for x in data.get("headings_rel", [0, 90, 180, 270])],
            extra_pitches=[int(x) for x in pitches_raw],
            interp_steps=int(data.get("interp_steps", 8)),
            interp_backend=str(data.get("interp_backend", "flow")),
            recon_backend=str(data.get("recon_backend", "flow")),
            max_panos=int(max_raw) if max_raw is not None else None,
        )
