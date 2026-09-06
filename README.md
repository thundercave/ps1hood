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
```

Tiny finishable demo (smoke preset: spacing 25 m, heading_step 90, pitches `[0]`, max 8 panos):

```bash
uv run ps1hood init smoke \
  --south 52.0895 --west 5.1192 --north 52.0900 --east 5.1200 \
  --preset smoke

uv run ps1hood run smoke
uv run ps1hood studio
```

Studio opens on [http://127.0.0.1:8765](http://127.0.0.1:8765). Or draw a **small** rectangle (one or two streets), save, run.

Larger / keyed capture example:

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

Built-in: flow triangulation from **aligned keyframes** (`align/cameras.json`, prefer near-horizon shots) + a RANSAC vertical-plane facade pass. Pairs are chosen by ENU baseline (2–25 m) and overlapping heading (≤60°), not list order alone. DIS midframes are only a fallback if fewer than two keyframes exist. Export / COLMAP / MASt3R use the same keyframes.

**Textured facades:** after plane extraction, each wall picks the most frontal camera, perspective-warps a JPEG into `recon/textures/`, and writes `recon/facades.obj` + `facades.mtl` with UVs. The ground quad can use a satellite ortho crop. Studio’s Three.js viewer loads the OBJ/MTL so walls read without a GPU reconstructor.

COLMAP is hardened for SV orbits: if the sparse model is missing/`points3D.bin` empty or <1 KB, reconstruct errors with a clear "use flow/mast3r/known poses" message; a valid model is converted to `recon/cloud_colmap.ply` (and copied to `cloud.ply`).

```bash
uv run ps1hood reconstruct my-block --backend flow
# or --backend colmap_posed   known ENU poses → point_triangulator (not mapper)
# or --backend colmap         prefers colmap_posed when cameras.json exists
# or --backend sift           OpenCV SIFT stereo on cross-pano pairs
# or --backend export
# or --backend mast3r         optional; needs GPU + naver/mast3r + weights
```

AMD / ROCm (RX 6900 XT): official MASt3R wants **CUDA**. See [`docs/gpu-mast3r-ubuntu.md`](docs/gpu-mast3r-ubuntu.md) for an honest ROCm feasibility note, uv setup on Ubuntu, denser `smoke-dense` capture, COLMAP posed fallbacks, and cloud-CUDA MASt3R commands.

`colmap_posed` seeds `cameras.txt`/`images.txt` from align ENU + FOV PINHOLE,
remaps IMAGE_IDs to the COLMAP database (colmap#497), matches **cross-pano**
pairs only (same-center orbit headings are pure rotation), then runs
`point_triangulator`. On success writes `recon/cloud_photo.ply` as primary.
If too few points, fails loud and falls back to OpenCV SIFT. Geometry stays
photo-derived — OSM/BAG shells are align priors only, not Studio hero mesh.

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
  recon/cloud_colmap.ply        optional COLMAP export
  recon/facades.obj + .mtl
  recon/textures/facade_XX.jpg
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
