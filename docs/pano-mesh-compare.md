# Pano ↔ mesh compare (diagnose-only)

**Goal:** score product façades + roofs against real Street View crops (+ sat XY) so we do not drift from life.

**Sacred:** known ENU poses only · no densify · no free-pose · no product wipe.

## CLI

```bash
uv run ps1hood compare smoke-dense
uv run ps1hood compare smoke-dense --max-cams 40 --patch 64
```

Writes `runs/<name>/recon/compare/`:

| Artifact | Content |
|----------|---------|
| `<pano>_hNNN.jpg` | Side-by-side: **photo \| mesh reproject \| absdiff+edge** (HUD: façade zncc/edge when available, #quads) |
| `summary.json` | Cams ranked worst-first by **façade** finite ZNCC; `worst` top 10; split `global` means + sat footprint Chamfer (m) |

CLI prints the worst 10 by **façade finite ZNCC** (cams that only see untextured
façades or sat roofs/yards score `facade_zncc=nan` and are counted in
`global.n_cams_nan_zncc` / omitted from `worst`). Soft-warn when
**`global.facade_zncc_mean` ≪ façade accept (0.35)** — roofs/yards do **not**
drive the soft-warn.

### Split metrics (hygiene)

Sat roof/yard shells are not photo-consistent by design. Do not read a single
polluted global ZNCC:

| Field | Meaning |
|-------|---------|
| `global.facade_zncc_mean` | Mean of per-cam **façade** finite ZNCC (photo↔mesh tex) |
| `global.roof_zncc_mean` | Mean of per-cam **roof/yard** finite ZNCC (expect low; sat shells) |
| `global.facade_edge_mean` | Mean of per-cam façade edge agreement |
| `global.sat_edge_mean_m` | Ortho footprint Chamfer vs sat Canny (m) — XY check for shells |
| `global.zncc_mean` / `edge_mean` | Mixed (legacy); may include roofs — prefer façade_* |
| `cam.kinds` | Contributing quad kinds (`facade`, `roof`, `yard`, …) |

**Note:** product `mean_zncc` (plane extract, photo↔photo multi-view) is a
different metric from compare façade ZNCC (photo↔mesh-texture). Untextured
façades score `zncc=nan` (no gray-fake). Roofs/yards belong in `roof_zncc_mean`
+ `sat_edge_mean_m`, not in the façade soft-warn.

## What it measures

1. **Per keyframe:** project each façade/roof quad with `P = K[Rcw|t]` (same as photo-planes). If ≥3 corners in front & in-frame → warp photo→ortho patch and mesh tex→same patch → **ZNCC** (skip if untextured) + projected-edge vs photo-Canny Chamfer agreement. Scores are aggregated **per kind**.
2. **Sat XY:** project footprints onto Ortho ENU; edge Chamfer vs sat Canny → `global.sat_edge_mean_m`.

Reuse: `photo_planes.zncc` / `project_points` / `Rt_from_frame`, `sat_edges.ortho_canny`.

## Studio

Optional: after compare,

- `GET /api/runs/<name>/compare/summary.json`
- `GET /api/runs/<name>/compare/<overlay.jpg>`

Or open the JPGs under `recon/compare/` directly. Studio `has_compare` flags runs with a summary. No geometry is modified.

## PC recipe (smoke-dense, after merge)

```bash
cd ~/ps1hood && git pull
uv sync --extra dev
uv run ps1hood compare smoke-dense --max-cams 40
# inspect worst overlays:
ls runs/smoke-dense/recon/compare/*.jpg | head
python -c "import json; s=json.load(open('runs/smoke-dense/recon/compare/summary.json')); print(s['global']); print(s['worst'][:5])"
```

## Non-goals

Fixing geometry in this PR — **diagnose only**. Feed worst **façade** cams into the next gap-fill / seat dig.
