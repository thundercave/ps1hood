"""Open public Street View in Chromium and screengrab the 360 pano.

No API key. Seeds come from OSM; Chromium opens the regular Maps Street
View URL a person would open, waits for the panorama, screenshots the
canvas, then we crop the Maps chrome so it cannot enter the point cloud.

Be polite: pause between panos. Personal reconstruction only — do not
republish the raw Google imagery.
"""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from ps1_hood.capture.google_js import orbit_headings
from ps1_hood.config import ProjectSpec, Settings
from ps1_hood.geo import wrap_heading

# Google snaps the viewpoint to the nearest pano, which can be a block away.
# Keep a little slop for GPS / the far kerb; drop anything farther.
PANO_BBOX_SLOP_M = 25.0

log = logging.getLogger(__name__)

PANO_ID_RE = re.compile(r"!1s([A-Za-z0-9_-]{20,24})")
AT_RE = re.compile(
    r"@(-?\d+\.\d+),(-?\d+\.\d+)(?:,3a)?(?:,(\d+(?:\.\d+)?)y)?"
    r"(?:,(\d+(?:\.\d+)?)h)?(?:,(\d+(?:\.\d+)?)t)?"
)
CONSENT_LABELS = (
    r"Alles afwijzen",
    r"Alle afwijzen",
    r"Reject all",
    r"Reject All",
    r"Afwijzen",
    r"Reject",
    r"Ik ga akkoord",
    r"Accept all",
    r"Accept All",
)


def streetview_url(
    lat: float,
    lon: float,
    heading: float = 0.0,
    pitch: float = 0.0,
    fov: float = 90.0,
    pano_id: str | None = None,
) -> str:
    """Consumer Street View URL (the same one the Maps share button makes)."""
    url = (
        "https://www.google.com/maps/@?api=1&map_action=pano"
        f"&viewpoint={lat:.7f},{lon:.7f}"
        f"&heading={heading:.2f}&pitch={pitch:.2f}&fov={fov:.1f}"
    )
    if pano_id and not pano_id.startswith("seed-"):
        url += f"&pano={pano_id}"
    return url


def parse_maps_url(url: str) -> dict[str, Any]:
    """Pull pano id / snapped lat,lng / heading out of a Maps Street View URL."""
    out: dict[str, Any] = {}
    m = PANO_ID_RE.search(url)
    if m:
        out["pano_id"] = m.group(1)
    at = AT_RE.search(url)
    if at:
        out["lat"] = float(at.group(1))
        out["lon"] = float(at.group(2))
        if at.group(4):
            out["heading"] = float(at.group(4))
        if at.group(5):
            out["pitch"] = float(at.group(5)) - 90.0
    q = parse_qs(urlparse(url).query)
    if "pano" in q and "pano_id" not in out:
        out["pano_id"] = q["pano"][0]
    if "viewpoint" in q and "lat" not in out:
        try:
            lat_s, lon_s = q["viewpoint"][0].split(",")
            out["lat"] = float(lat_s)
            out["lon"] = float(lon_s)
        except ValueError:
            pass
    return out


NO_COVERAGE = (
    "geen street view",
    "no street view",
    "street view imagery not available",
    "street view-afbeeldingen beschikbaar",
)


def looks_like_streetview(url: str) -> bool:
    # The share URL alone is not enough — Maps often stays on the 2D map
    # until the panorama id is written into `data=!3m…!1e1`.
    return "!1e1" in url or bool(PANO_ID_RE.search(url))


def page_says_no_coverage(page: Any) -> bool:
    try:
        text = (page.inner_text("body") or "").lower()
    except Exception:
        return False
    return any(s in text for s in NO_COVERAGE)


def _launch_browser(playwright: Any) -> Any:
    args = [
        "--disable-dev-shm-usage",
        "--no-sandbox",
        "--use-gl=angle",
        "--use-angle=swiftshader",
        "--enable-webgl",
        "--ignore-gpu-blocklist",
        "--headless=new",
    ]
    # Bundled Chromium, always headless unless PS1HOOD_HEADED=1.
    # Do not launch the user's visible Google Chrome.
    headed = os.environ.get("PS1HOOD_HEADED", "").strip().lower() in {"1", "true", "yes"}
    headless = not headed
    if headed:
        args = [a for a in args if not a.startswith("--headless")]
    return playwright.chromium.launch(headless=headless, args=args)


def _dismiss_consent(page: Any) -> None:
    for label in CONSENT_LABELS:
        try:
            btn = page.get_by_role("button", name=re.compile(label, re.I))
            if btn.count() > 0 and btn.first.is_visible():
                btn.first.click(timeout=2500)
                page.wait_for_load_state("domcontentloaded")
                time.sleep(0.6)
                return
        except Exception:
            continue


def _largest_canvas(page: Any) -> Any | None:
    handle = page.evaluate_handle(
        """() => {
          const cs = [...document.querySelectorAll('canvas')];
          if (!cs.length) return null;
          cs.sort((a, b) => (b.width * b.height) - (a.width * a.height));
          return cs[0];
        }"""
    )
    return handle.as_element()


def _wait_for_pano(page: Any, timeout_ms: int = 25000) -> bool:
    deadline = time.time() + timeout_ms / 1000.0
    while time.time() < deadline:
        _dismiss_consent(page)
        if looks_like_streetview(page.url):
            time.sleep(0.8)
            return True
        time.sleep(0.35)
    return looks_like_streetview(page.url)


def _screenshot_canvas(page: Any, path: Path) -> bool:
    el = _largest_canvas(page)
    if el is None:
        return False
    el.screenshot(path=str(path), type="jpeg", quality=90)
    return path.is_file() and path.stat().st_size > 8000


def capture_panos(
    panos: list[dict[str, Any]],
    spec: ProjectSpec,
    settings: Settings,
    dest: Path,
    on_event: Any | None = None,
) -> list[dict[str, Any]]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Chromium screengrab needs Playwright:\n"
            "  uv sync --extra browser\n"
            "  uv run ps1hood setup-browser"
        ) from exc

    dest.mkdir(parents=True, exist_ok=True)
    shots: list[dict[str, Any]] = []
    seen: set[str] = set()
    processed: set[str] = set()
    skipped = 0
    w, h = settings.js_size
    pause = 1.2

    def _tick(current: str, message: str) -> None:
        if on_event:
            on_event(
                {
                    "queued": len(panos),
                    "captured": len(seen),
                    "skipped": skipped,
                    "current": current,
                    "message": message,
                }
            )

    with sync_playwright() as p:
        browser = _launch_browser(p)
        context = browser.new_context(
            viewport={"width": w, "height": h},
            locale="nl-NL",
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
            ),
        )
        context.add_cookies(
            [
                {
                    "name": "CONSENT",
                    "value": "YES+",
                    "domain": domain,
                    "path": "/",
                }
                for domain in (".google.com", ".google.nl", ".google.co.uk")
            ]
        )
        page = context.new_page()
        try:
            for i, pano in enumerate(panos):
                travel = float(pano.get("travel_heading") or 0.0)
                seed_lat = float(pano["lat"])
                seed_lon = float(pano["lon"])
                url = streetview_url(seed_lat, seed_lon, travel, 0.0, spec.fov_deg)
                log.info("open %s/%s  %.6f,%.6f", i + 1, len(panos), seed_lat, seed_lon)
                _tick(f"{seed_lat:.5f},{seed_lon:.5f}", f"opening {i + 1}/{len(panos)}")
                page.goto(url, wait_until="domcontentloaded", timeout=45000)
                _dismiss_consent(page)
                if not _wait_for_pano(page) or page_says_no_coverage(page):
                    log.info("  no street view here")
                    skipped += 1
                    _tick("", f"no coverage {i + 1}/{len(panos)}")
                    time.sleep(pause)
                    continue
                meta = parse_maps_url(page.url)
                pid = meta.get("pano_id") or pano.get("pano_id") or f"web-{seed_lat:.6f}-{seed_lon:.6f}"
                if pid in processed:
                    log.info("  already grabbed %s", pid)
                    skipped += 1
                    _tick(pid, f"duplicate {pid}")
                    time.sleep(0.4)
                    continue
                processed.add(pid)
                lat = float(meta.get("lat") or seed_lat)
                lon = float(meta.get("lon") or seed_lon)
                if not spec.bbox.contains_m(lat, lon, slop_m=PANO_BBOX_SLOP_M):
                    log.info(
                        "  snapped pano %s at %.6f,%.6f is outside the bbox — skip",
                        pid,
                        lat,
                        lon,
                    )
                    skipped += 1
                    _tick(pid, f"outside bbox {pid}")
                    time.sleep(0.4)
                    continue
                seen.add(pid)
                _tick(pid, f"captured {len(seen)} / {len(panos)}")
                pdir = dest / pid
                pdir.mkdir(parents=True, exist_ok=True)
                headings = orbit_headings(spec.heading_step, travel)
                for heading in headings:
                    for pitch in spec.extra_pitches:
                        name = f"h{int(round(heading)):03d}_p{int(pitch):+d}.jpg"
                        path = pdir / name
                        rel = wrap_heading(heading - travel)
                        if not path.exists():
                            hop = streetview_url(lat, lon, heading, float(pitch), spec.fov_deg, pid)
                            page.goto(hop, wait_until="domcontentloaded", timeout=45000)
                            _dismiss_consent(page)
                            _wait_for_pano(page, timeout_ms=8000)
                            if not _screenshot_canvas(page, path):
                                log.warning("  screenshot failed %s", name)
                                continue
                            time.sleep(0.35)
                        shots.append(
                            {
                                "pano_id": pid,
                                "lat": lat,
                                "lon": lon,
                                "heading": float(heading),
                                "pitch": float(pitch),
                                "fov": spec.fov_deg,
                                "rel_heading": rel,
                                "travel_heading": travel,
                                "path": str(path),
                                "source": "google_web",
                            }
                        )
                time.sleep(pause)
        finally:
            context.close()
            browser.close()
    return shots
