"""360° Street View capture by driving Chromium and screengrabbing the pano.

This is the original distribution-friendly loop:

  for each panorama in the bbox
      hide the HTML chrome
      spin heading around 360° (and a couple of pitches)
      screenshot the *canvas* (the photograph, not the widgets)
      later crop/mask anything Google still painted onto the canvas

We point Chromium at *our* official Street View embed (Maps JavaScript API),
not at maps.google.com. A shippable build still needs a Maps JS key — that
is the supported way to render Street View in a third-party app.
"""

from __future__ import annotations

import logging
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from ps1_hood.config import ProjectSpec, Settings
from ps1_hood.geo import heading_diff, wrap_heading

log = logging.getLogger(__name__)

CAPTURE_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <style>
    html, body, #pano { margin: 0; padding: 0; width: 100%; height: 100%; background: #000; overflow: hidden; }
  </style>
</head>
<body>
  <div id="pano"></div>
  <script>
    window.__sv = { ready: false, error: null, current: null };
    function hideOverlays() {
      const root = document.getElementById('pano');
      if (!root) return;
      const keep = new Set();
      for (const c of root.querySelectorAll('canvas')) {
        let n = c;
        while (n && n !== root) { keep.add(n); n = n.parentElement; }
      }
      for (const el of root.querySelectorAll('*')) {
        if (el.tagName === 'CANVAS' || keep.has(el)) continue;
        el.style.visibility = 'hidden';
        el.style.opacity = '0';
        el.style.pointerEvents = 'none';
      }
    }
    function initPano() {
      const el = document.getElementById('pano');
      window.__pano = new google.maps.StreetViewPanorama(el, {
        visible: true,
        disableDefaultUI: true,
        linksControl: false,
        panControl: false,
        zoomControl: false,
        addressControl: false,
        fullscreenControl: false,
        motionTracking: false,
        motionTrackingControl: false,
        showRoadLabels: false,
        clickToGo: false,
        imageDateControl: false,
        enableCloseButton: false
      });
      window.__pano.addListener('tilesloaded', hideOverlays);
      window.__sv.ready = true;
    }
    window.__setView = function(panoId, heading, pitch, fov) {
      return new Promise((resolve, reject) => {
        const pano = window.__pano;
        const zoom = Math.log2(180 / fov);
        let settled = false;
        const finish = (err) => {
          if (settled) return;
          settled = true;
          hideOverlays();
          if (err) { reject(err); return; }
          const loc = pano.getLocation();
          const ll = loc && loc.latLng;
          const info = {
            pano: pano.getPano(),
            heading: pano.getPov().heading,
            pitch: pano.getPov().pitch,
            lat: ll ? ll.lat() : null,
            lng: ll ? ll.lng() : null,
            description: loc ? loc.description : null,
            links: (pano.getLinks() || []).map(l => ({
              pano: l.pano, heading: l.heading, description: l.description
            }))
          };
          window.__sv.current = info;
          resolve(info);
        };
        const same = window.__sv.current && window.__sv.current.pano === panoId;
        const wait = same ? 700 : 2200;
        const t = setTimeout(() => finish(null), wait + 400);
        const onReady = () => {
          hideOverlays();
          setTimeout(() => { clearTimeout(t); finish(null); }, wait);
        };
        if (same) {
          pano.setPov({ heading: heading, pitch: pitch });
          pano.setZoom(zoom);
          google.maps.event.addListenerOnce(pano, 'idle', onReady);
        } else {
          google.maps.event.addListenerOnce(pano, 'status_changed', () => {
            if (pano.getStatus() !== 'OK') {
              clearTimeout(t);
              finish('status ' + pano.getStatus());
            }
          });
          google.maps.event.addListenerOnce(pano, 'pano_changed', () => {
            pano.setPov({ heading: heading, pitch: pitch });
            pano.setZoom(zoom);
          });
          google.maps.event.addListenerOnce(pano, 'tilesloaded', onReady);
          pano.setPano(panoId);
        }
      });
    };
  </script>
  <script async
    src="https://maps.googleapis.com/maps/api/js?key=__API_KEY__&callback=initPano">
  </script>
</body>
</html>
"""


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, fmt: str, *args: object) -> None:
        return


def orbit_headings(step: int, travel_heading: float | None = None) -> list[int]:
    """Absolute compass headings that walk a full 360 around a pano."""
    step = max(15, min(90, int(step)))
    found = list(range(0, 360, step))
    if travel_heading is not None:
        travel = int(round(wrap_heading(travel_heading))) % 360
        if all(abs(heading_diff(travel, h)) > 1.0 for h in found):
            found.append(travel)
    return sorted(set(found))


def _serve(html_path: Path) -> tuple[ThreadingHTTPServer, str]:
    handler = partial(_QuietHandler, directory=str(html_path.parent))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/{html_path.name}"


def _screenshot_canvas(page: Any, path: Path) -> None:
    """Grab the photograph canvas, not the HTML widgets sitting on top."""
    handle = page.evaluate_handle(
        """() => {
          const cs = [...document.querySelectorAll('#pano canvas')];
          if (!cs.length) return document.getElementById('pano');
          cs.sort((a, b) => (b.width * b.height) - (a.width * a.height));
          return cs[0];
        }"""
    )
    element = handle.as_element()
    if element is None:
        page.locator("#pano").screenshot(path=str(path), type="jpeg", quality=92)
        return
    element.screenshot(path=str(path), type="jpeg", quality=92)


def capture_panos(
    panos: list[dict[str, Any]],
    spec: ProjectSpec,
    settings: Settings,
    dest: Path,
    work: Path,
) -> list[dict[str, Any]]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "360 screenshot capture needs Playwright/Chromium. Install with:\n"
            "  uv sync --extra browser\n"
            "  uv run ps1hood setup-browser"
        ) from exc
    if not settings.google_maps_api_key:
        raise RuntimeError(
            "GOOGLE_MAPS_API_KEY is required. The official Street View JS "
            "embed is what Chromium screengrabs — there is no key-free Google path."
        )

    work.mkdir(parents=True, exist_ok=True)
    html_path = work / "sv_capture.html"
    html_path.write_text(
        CAPTURE_HTML.replace("__API_KEY__", settings.google_maps_api_key),
        encoding="utf-8",
    )
    server, url = _serve(html_path)
    dest.mkdir(parents=True, exist_ok=True)
    shots: list[dict[str, Any]] = []
    w, h = settings.js_size
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": w, "height": h})
            page.goto(url, wait_until="domcontentloaded")
            page.wait_for_function("() => window.__sv && window.__sv.ready", timeout=30000)
            for i, pano in enumerate(panos):
                travel = float(pano.get("travel_heading") or 0.0)
                headings = orbit_headings(spec.heading_step, travel)
                pdir = dest / pano["pano_id"]
                pdir.mkdir(parents=True, exist_ok=True)
                for heading in headings:
                    for pitch in spec.extra_pitches:
                        name = f"h{int(round(heading)):03d}_p{int(pitch):+d}.jpg"
                        path = pdir / name
                        rel = wrap_heading(heading - travel)
                        log.info(
                            "grab %s/%s  %s  heading=%s pitch=%s",
                            i + 1,
                            len(panos),
                            pano["pano_id"],
                            heading,
                            pitch,
                        )
                        if not path.exists():
                            info = page.evaluate(
                                """async ({id, heading, pitch, fov}) => {
                                    return await window.__setView(id, heading, pitch, fov);
                                }""",
                                {
                                    "id": pano["pano_id"],
                                    "heading": float(heading),
                                    "pitch": float(pitch),
                                    "fov": spec.fov_deg,
                                },
                            )
                            _screenshot_canvas(page, path)
                        else:
                            info = None
                        shots.append(
                            {
                                "pano_id": pano["pano_id"],
                                "lat": (info or {}).get("lat") or pano["lat"],
                                "lon": (info or {}).get("lng") or pano["lon"],
                                "heading": float(heading),
                                "pitch": float(pitch),
                                "fov": spec.fov_deg,
                                "rel_heading": rel,
                                "travel_heading": travel,
                                "path": str(path),
                                "source": "google_js",
                                "links": (info or {}).get("links") or [],
                            }
                        )
            browser.close()
    finally:
        server.shutdown()
    return shots
