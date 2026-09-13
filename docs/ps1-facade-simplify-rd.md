# PS1 façade simplify — PR1 (Manhattan + RGB555)

**Date:** 2026-09-13 · **Branch:** `feat/ps1-facade-simplify`  
**Goal:** kill wavy walls/windows **without** re-recon / more peels / FILM / sat roofs.  
**Sacred:** photo-derived geometry · sat XY lock · no BAG hero · quality-keep not re-run for post-process.

Pack source: `/workspace/ps1-facade-simplify-rd.md` (copied into docs for the PR).

---

## Out of scope (this PR)

Sat roofs · denser panos/FILM · more Path α peels · Instant Meshes · Poisson · MapAnything / peel / align changes

---

## What landed

| File | Change |
|------|--------|
| `src/ps1_hood/reconstruct/ps1_facades.py` | `manhattan_rectify_planes`, `ps1_quantize`, `nearest_resize`, `postprocess_run` |
| `src/ps1_hood/reconstruct/facades.py` | rectify after accept → before warp/write; `--ps1-tex-size` on warp |
| `src/ps1_hood/cli.py` | `facades --ps1-rectify/--no-ps1-rectify` (default on) + `--ps1-tex-size 128`; `ps1hood ps1-facades <run>` |
| `src/ps1_hood/studio/static/viewer.html` | façade maps `NearestFilter` mag/min |
| `tests/test_ps1_facades.py` | opposite edges equal; RGB555 bits |

### Recipe

1. Keep each accepted plane’s **center** + **n** from photo ZNCC.
2. Rebuild quad as ENU rectangle: `right = (−n_y, n_x, 0)`, `up = (0,0,1)`; clamp width 3…20 m, height 2.5…15 m; force equal opposite edges.
3. Warp as today, then **nearest** resize to 128² or 128×256 + **RGB555** (`c >> 3 << 3`), JPEG q≤85.
4. Studio: `magFilter = minFilter = NearestFilter` on façade textures.

### CLI

```bash
# full extract (rectify default on)
ps1hood facades <run> --ps1-rectify --ps1-tex-size 128

# post-process existing product (no re-extract / no quality-keep)
cp runs/<run>/recon/facades.obj runs/<run>/recon/facades.obj.bak   # optional; cmd backups once
ps1hood ps1-facades <run> --ps1-tex-size 128
```

Then Studio hard-reload.

### Acceptance

- Walls look planar rectangles (no bow-tie / wavy sill)
- Windows read as blocky grid
- Same plane count; mean ZNCC gate not re-run on post-process
- No BAG; sat XY unchanged; no densify re-run
