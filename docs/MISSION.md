# Mission

Rebuild a **street-scale 3D neighborhood** that looks like a **PlayStation 1** game world, using **Street View** (or Mapillary) plus **satellite** as the only sensors.

Not a city-scale Google Earth clone. One or two blocks. Walkable. Blocky. Personal.

## What “done” means

1. Draw a small bbox.
2. Pipeline finds road-side panoramas, captures facade-facing views, strips UI chrome, pulls an ortho, seats cameras in a local ENU frame (OSM + optional 3DBAG).
3. Dense-enough geometry becomes a **low-poly, textured hood** — affine UVs, crushed palette, short draw distance — viewable in Studio.
4. You can recognize *your* street without mistaking it for a photogrammetry museum piece.

## Non-goals

- **Not a Street View mirror.** Do not republish, host, or redistribute raw Google / Mapillary captures or near-lossless crops. Personal reconstruction only; respect provider terms.
- **Not a general mapping product.** No global coverage SLA, no multi-user cloud, no “render any address” API.
- **Not film-quality NeRF/splat tourism.** Splats/COLMAP/MASt3R are *tools* on the way to game-like meshes, not the destination.
- **Not a ToS workaround product.** The Chromium screengrab path exists so a hobbyist can rebuild *their* block without pasting an API key on day one — not to industrialize scraping.
- **Not NL-only forever — but BAG is NL-only today.** Best camera seating uses 3DBAG; elsewhere we degrade to OSM + satellite.

## Design principles

- **Finish small blocks.** A completed ugly demo beats a 200-pano queue.
- **Defaults must run offline** after capture (flow interp + flow recon). Heavy AI is optional extras.
- **Geometry first, look second, packaging third** — see `ROADMAP.md`.
- **Keep UI out of the cloud.** Crop/mask Maps chrome before recon.

## One-line pitch

*Your block, PS1-shaped, rebuilt at home — not republished Street View.*
