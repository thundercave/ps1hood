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
| `<pano>_hNNN.jpg` | Side-by-side: **photo \| mesh reproject \| absdiff+edge** (HUD: zncc, edge, #quads) |
| `summary.json` | All cams ranked worst-first by mean ZNCC; `worst` top 10; `global` means + sat footprint Chamfer (m) |

CLI prints the worst 10. Soft-warn when global ZNCC ≪ façade accept (0.35).

## What it measures

1. **Per keyframe:** project each façade/roof quad with `P = K[Rcw\|t]` (same as photo-planes). If ≥3 corners in front & in-frame → warp photo→ortho patch and mesh tex/gray→same patch → **ZNCC** + projected-edge vs photo-Canny Chamfer agreement.
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

Fixing geometry in this PR — **diagnose only**. Feed worst cams into the next gap-fill / seat dig.
