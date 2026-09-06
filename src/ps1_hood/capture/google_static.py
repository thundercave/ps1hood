"""Official Street View Static API + metadata (no unofficial tile scrape)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ps1_hood.config import ProjectSpec, Settings
from ps1_hood.geo import wrap_heading
from ps1_hood.httputil import get_bytes, get_json

log = logging.getLogger(__name__)

META_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"
IMAGE_URL = "https://maps.googleapis.com/maps/api/streetview"


def lookup_pano(lat: float, lon: float, api_key: str) -> dict[str, Any] | None:
    data = get_json(
        META_URL,
        params={
            "location": f"{lat:.7f},{lon:.7f}",
            "source": "outdoor",
            "key": api_key,
        },
    )
    if data.get("status") != "OK":
        return None
    loc = data.get("location") or {}
    return {
        "pano_id": data["pano_id"],
        "lat": float(loc["lat"]),
        "lon": float(loc["lng"]),
        "date": data.get("date"),
        "copyright": data.get("copyright"),
        "provider": "google",
    }


def fetch_view(
    *,
    pano_id: str,
    heading: float,
    pitch: float,
    fov: float,
    size: tuple[int, int],
    api_key: str,
) -> bytes:
    w, h = size
    return get_bytes(
        IMAGE_URL,
        params={
            "pano": pano_id,
            "heading": f"{heading:.2f}",
            "pitch": f"{pitch:.2f}",
            "fov": f"{fov:.1f}",
            "size": f"{w}x{h}",
            "source": "outdoor",
            "return_error_code": "true",
            "key": api_key,
        },
        timeout=45.0,
    )


def capture_panos(
    panos: list[dict[str, Any]],
    spec: ProjectSpec,
    settings: Settings,
    dest: Path,
) -> list[dict[str, Any]]:
    if not settings.google_maps_api_key:
        raise RuntimeError("GOOGLE_MAPS_API_KEY is required for google_static capture")
    shots: list[dict[str, Any]] = []
    dest.mkdir(parents=True, exist_ok=True)
    for i, pano in enumerate(panos):
        travel = float(pano.get("travel_heading") or 0.0)
        pdir = dest / pano["pano_id"]
        pdir.mkdir(parents=True, exist_ok=True)
        for rel in spec.headings_rel:
            heading = wrap_heading(travel + rel)
            for pitch in spec.extra_pitches:
                name = f"h{int(round(heading)):03d}_p{int(pitch):+d}.jpg"
                path = pdir / name
                if not path.exists():
                    log.info(
                        "static %s/%s  %s  heading=%.1f pitch=%s",
                        i + 1,
                        len(panos),
                        pano["pano_id"],
                        heading,
                        pitch,
                    )
                    path.write_bytes(
                        fetch_view(
                            pano_id=pano["pano_id"],
                            heading=heading,
                            pitch=float(pitch),
                            fov=spec.fov_deg,
                            size=settings.image_size,
                            api_key=settings.google_maps_api_key,
                        )
                    )
                shots.append(
                    {
                        "pano_id": pano["pano_id"],
                        "lat": pano["lat"],
                        "lon": pano["lon"],
                        "heading": heading,
                        "pitch": float(pitch),
                        "fov": spec.fov_deg,
                        "rel_heading": rel,
                        "path": str(path),
                        "source": "google_static",
                    }
                )
    return shots
