# Roadmap

Concrete plan to make the README mission real: a street-scale neighborhood that *reads* as PS1, rebuilt from Street View + satellite.

Status snapshot (repo as of this write-up):

- Full stage graph is wired: discover → capture → crop → satellite → bag → align → interpolate → reconstruct.
- Unit tests: 38/38. Studio (Leaflet + Three.js) runs locally.
- `runs/demo-utrecht` discovered **199** provisional panos; capture never finished (raw/ empty). `runs/smoke` discovered 10.
- Default recon is OpenCV DIS flow triangulation + RANSAC vertical-plane `facades.obj` (untextured).
- `interp_backend: film|rife` and `recon_backend: mast3r` are config placeholders — they fall back or raise after exporting frames.
- True PS1 look (affine textures, vertex snap, dither/palette, low-poly remesh) is **not implemented**; only a blockout OBJ.

---

## Phase 0 — Working demo

**Goal:** One tiny block you can finish overnight, open in Studio, and show without apology as “pipeline works end-to-end.”

### Work

1. **Capture budget controls** (project.yaml + CLI) — landed: `max_panos`, `--preset smoke`, pitches `[0]`:
   - `max_panos` (hard cap after dedupe; evenly subsampled by index)
   - `heading_step` already exists — document demo defaults (`90`)
   - `extra_pitches: [0]` for smoke; keep multi-pitch as an opt-in quality mode
   - Optional `capture_mode: sparse|orbit` (sparse = travel + ±90° facade only) — still open
2. **Ship a finished smoke artifact** under `runs/smoke` (or a checked-in fixture of *synthetic* frames so CI does not hit Google): crop → align → interp → recon → `scene.json` + `cloud.ply` + `facades.obj`.
3. **Viewer polish for the demo:** load `facades.obj` (not only PLY + BAG meshes); show align overlay + residual stats in HUD.
4. **Preset:** `ps1hood init … --preset smoke` that sets spacing ≥ 20 m, heading_step 90, pitches `[0]`, interp_steps 3–4, max_panos ≤ 12.

### Success criteria

- [ ] From a cold clone: `uv sync --extra browser && ps1hood setup-browser && ps1hood run <smoke>` completes without hanging overnight on a normal block.
- [ ] Studio `/viewer?run=<smoke>` shows satellite ground, cameras, coloured cloud, and facade blockout.
- [ ] Capture stage logs a clear budget: `captured N / max M`, with N ≤ M.
- [ ] README “Quick start” points at the smoke preset, not a 200-pano Utrecht box.

### Exit note

Phase 0 is about *finishability*, not beauty. Ugly cloud + cardboard walls is a pass.

---

## Phase 1 — Better reconstruction (MASt3R)

**Goal:** Replace “flow triangulation as the product” with a real video→cloud path, while keeping flow as the no-GPU fallback.

### Work

1. **Optional `mast3r` extra** (torch + vendored or documented install). Do not put multi-GB weights in the default `uv sync`.
2. Wire `stage_reconstruct` when `recon_backend: mast3r`:
   - consume `interp/frames` (or sparse keyframes from align cameras)
   - write `recon/cloud.ply` (+ optional depth/confidence)
   - fall back to `export` with a clear error if weights missing
3. Prefer **aligned keyframes** over densely interpolated DIS frames as MASt3R input (large-baseline stills are what FILM/MASt3R want; DIS-warped midframes can poison SfM).
4. Keep COLMAP path working; document VGGT / Luma / Postshot as external “drop `drive.mp4` here.”
5. (Stretch) FILM or Practical-RIFE behind `interp_backend` with a weight download script — only after MASt3R path is usable.

### Success criteria

- [ ] On a GPU box with weights present: `ps1hood reconstruct <run> --backend mast3r` produces a denser, more coherent cloud than `--backend flow` on the same smoke run (qualitative side-by-side in Studio; no fake metrics required).
- [ ] Without weights: reconstruct fails with an actionable install message and leaves exported frames on disk.
- [ ] Default remaining `flow` still works offline with zero AI deps.

### Exit note

Phase 1 is geometry quality. Still not “PS1 game art.”

---

## Phase 2 — PS1 look

**Goal:** Someone glancing at the viewer says “that looks like a PS1 neighborhood,” not “that’s a research point cloud.”

The current facade pass is **not enough** (see critique below). Build a dedicated look pipeline on top of better geometry (BAG LoD2.2 and/or MASt3R mesh).

### Work

1. **Low-poly structure**
   - Prefer 3DBAG (NL) solid shells as the authoring mesh; elsewhere: Poisson/ball-pivoting or plane clustering from the cloud, then simplify.
   - Vertex snap / quantize to a coarse grid (e.g. 0.25–0.5 m) so silhouettes jitter like early console geo.
2. **Affine facade textures**
   - Project Street View facade shots onto vertical faces (affine UV, not perspective-correct in the viewer shader).
   - Atlas per block; crush to low resolution (32–128 px strip heights).
3. **Palette + dither**
   - Posterize / optional ordered dither in a post shader; limited fog + short draw distance.
4. **Viewer mode `?look=ps1`**
   - No antialias, nearest-neighbor sampling, optional camera jitter, disable modern PBR lighting.
5. **Ground**
   - Satellite ortho downsampled + palette-matched as a textured ground quad (already partially there).

### Success criteria

- [ ] Textured low-poly mesh exports (`recon/hood.obj` + atlas) viewable without the point cloud.
- [ ] Side-by-side: modern PLY view vs PS1 look mode on the same run.
- [ ] At least one screenshot worthy of the README hero (personal use only — no raw SV republish).

### Exit note

Do not fake “PS1” with untextured RANSAC quads. Texture + quantization + viewer constraints are the look.

---

## Phase 3 — Productization

**Goal:** A hobby tool another engineer can run without reading the source, with clear geographic limits and legal posture.

### Work

1. **Geographic adapters:** 3DBAG is NL-only. Abstract “building prior” (BAG / OSM LoD1 boxes / none) so align degrades gracefully outside the Netherlands.
2. **Job UX:** cancel run, stage resume in Studio UI, disk-usage estimate before capture, polite rate limits documented.
3. **Mapillary-first path** for users who want licensed street-level imagery with poses.
4. **Packaging:** optional Docker with browser deps; pin Playwright Chromium version.
5. **Legal / ethics copy** in-app (not only README): personal reconstruction, no redistribute raw captures, respect ToS.
6. **CI:** keep unit tests; add a synthetic-frame recon smoke that never hits the network.

### Success criteria

- [ ] New contributor finishes smoke demo from README alone.
- [ ] Non-NL bbox completes with satellite/OSM align (no hard fail on missing BAG).
- [ ] Explicit “do not publish raw Street View” notice in Studio create-run flow.

---

## Single best next engineering PR

**PR title (suggested): `capture: budget + smoke preset (max_panos, sparse orbit)`**

Why this one first:

- `demo-utrecht` at ~200 panos × orbit × 3 pitches is the practical blocker; nothing else (MASt3R, PS1 shaders) can be iterated without a finished small run.
- Small, testable change: config fields, discover/capture early-exit, CLI `--preset smoke`, unit tests for the cap.
- Unlocks Phase 0 success criteria without downloading AI weights or rewriting recon.

Out of scope for that PR: FILM weights, MASt3R, PS1 shaders.

---

## Recommended follow-on PRs (priority)

After the capture-budget PR:

1. **Textured facades + load OBJ in viewer** — first “hood” screenshot from BAG planes or RANSAC quads + SV projection.
2. **MASt3R optional backend** — real cloud quality on the smoke run.
3. **PS1 viewer mode** — affine sampling, palette/dither, vertex quantize on export.

---

## Maturity cheat-sheet (for contributors)

| Area | Maturity | Notes |
|------|----------|-------|
| Discover / OSM | Good | Overpass + spacing; google_web seeds are provisional until capture |
| Capture google_web | Good but slow | Playwright canvas grab; `max_panos` + smoke preset for budget |
| Crop | Good | Chrome strip + auto mask |
| Satellite | Good | Esri ortho, no key |
| 3DBAG + edge snap | Strong (NL) | Best align signal in-repo |
| Pose graph / seat | Good | Collapse fixes, footprint push |
| Interpolate | Baseline only | DIS flow; FILM/RIFE not wired |
| Recon flow | Baseline | Coloured PLY; noisy at SV baselines |
| Facades OBJ | Prototype | Untextured vertical planes ≠ PS1 |
| Studio | Usable | No PS1 look mode; OBJ not loaded |
