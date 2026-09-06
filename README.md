# ps1hood

A hobby project to recreate the world in PS1 graphics.

Rebuild a street-scale 3D neighborhood from **Street View** and **satellite** imagery.

The previous install of this project was wiped. This is a reconstruction of that pipeline from memory:

1. Draw a bounding box around a block.
2. Find every Street View (or Mapillary) sample on the roads inside it.
3. Capture those views — including facade-facing headings, not just the drive direction.
4. Crop leftover Street View chrome.
5. Pull a satellite mosaic of the same box.
6. **Re-position the cameras in 3D.** Official metadata is several metres / a few degrees off. Cameras are snapped to OSM roads, matched to their neighbours, and nudged until a synthetic ground-plane render of the satellite agrees with the photo.
7. **Interpolate** between neighbouring views so the capture becomes a dense “drive” video.
8. Build a **coloured point cloud** (and a rough facade blockout) from that video. Optionally hand the same frames to COLMAP, MASt3R, VGGT, Luma, or Postshot — those were the “AI that makes a point cloud from video” family the original build was heading toward.

## Quick start

```bash
cd ~/ps1-hood
cp .env.example .env
# no Google key needed for the default Chrome screengrab path

uv sync --extra dev --extra browser
uv run ps1hood setup-browser
uv run ps1hood studio
```

Studio opens on [http://127.0.0.1:8765](http://127.0.0.1:8765). Draw a **small** rectangle (one or two streets), save, run.

Or from the CLI:

```bash
uv run ps1hood init my-block \
  --south 52.0889 --west 5.1180 --north 52.0904 --east 5.1210 \
  --source google_static --spacing 8 --steps 6

uv run ps1hood run my-block
uv run ps1hood studio
```

Resume a later stage after a failure:

```bash
uv run ps1hood run my-block --from-stage align
```

## Capture backends

The shippable path is **Chromium 360 screengrab**. For every panorama in the bbox the bundled browser hides the HTML chrome, spins heading around 360° (default every 45°) at a few pitches, and screenshots the **canvas** — the photograph, not the widgets. A second pass then crops the copyright bar and masks compass / zoom / address chips, because those reconstruct as ghost geometry in the point cloud.

That is why the original build used Chrome + crop: the UI has to stay out of the cloud. Default path (`google_web`) opens the same public Street View URLs a person would, screengrabs the canvas, then strips chrome. Pause between panos. Personal reconstruction only — don’t republish the raw captures.

| `source` | What it does | Needs |
|---|---|---|
| `google_web` **(default)** | Chromium opens public Street View, 360 orbit, screengrab, crop UI. | `ps1hood setup-browser` |
| `google_js` | Official JS embed (cleaner, billed). | `GOOGLE_MAPS_API_KEY` + browser |
| `google_static` | Official Street View Static API. 640×640, no orbit. | `GOOGLE_MAPS_API_KEY` |
| `mapillary` | Open street-level photos with computed poses. | `MAPILLARY_TOKEN` |

Discovery walks OSM road centerlines (Overpass) and asks the Street View metadata API at regular spacing. Duplicate `pano_id`s are collapsed. Mapillary uses the official bbox image search (must stay under 0.01 deg²).

Imagery is copyrighted. This tool is for personal reconstruction. Respect Google / Mapillary terms and do not republish the raw captures.

## The forgotten AI pieces

The original build interpolated between Street Views, then fed the resulting video to “some AI that makes point clouds.” Two families fit that description and were the obvious 2024–25 choices:

**Interpolation (large baseline, photos not a real video):**

- [FILM](https://github.com/google-research/frame-interpolation) — Google, made for big motion between stills. Most likely what we used or meant to use.
- [RIFE](https://github.com/hzwer/Practical-RIFE) / Practical-RIFE.

This repo ships a no-weight **optical-flow (DIS)** interpolator so the rest of the pipeline runs today. Point `interp_backend` at `film` or `rife` in `project.yaml` once those weights live on the machine; until then it falls back to flow.

**Video → point cloud / splat:**

- [MASt3R](https://github.com/naver/mast3r) / MASt3R-SfM — “drop images, get a point cloud.” Strong match for the missing piece.
- [VGGT](https://github.com/facebookresearch/vggt) — Meta, 2025, feed-forward point maps from a video.
- COLMAP — still the reliable ordered-video baseline.
- Luma / Postshot / Polycam — drag `runs/<name>/interp/drive.mp4` in.

Built-in: flow triangulation from the interpolated drive + a RANSAC vertical-plane facade pass (`recon/cloud.ply`, `recon/facades.obj`). Export the same frames with:

```bash
uv run ps1hood reconstruct my-block --backend export
# or --backend colmap   if colmap is installed
```

## Layout of a run

```
runs/<name>/
  project.yaml
  osm/roads.json
  satellite/ortho.jpg          Esri World Imagery, no key
  discover/panos.json
  raw/<pano>/h090_p+0.jpg
  cropped/...
  align/poses.json             refined ENU metres + heading
  align/overlay.jpg            cameras drawn on the satellite
  interp/frames/00000.jpg
  interp/drive.mp4
  recon/cloud.ply
  recon/facades.obj
  recon/scene.json
```

## Keys

Default `google_web` needs none. Optional backends:

```
GOOGLE_MAPS_API_KEY=...
MAPILLARY_TOKEN=...
PS1HOOD_SOURCE=google_web
```

## Develop

```bash
uv sync --extra dev
uv run pytest
```
