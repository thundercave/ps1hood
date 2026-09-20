# Re-texture bare façades

**Date:** 2026-09-20 · Product walls can pass ZNCC yet still lack `map_Kd`  
**Cause:** post-rectify full-quad warp hard-fails (corners >40 px out of frame) after ZNCC scored a smaller patch  
**Sacred:** no free-pose · quality-keep · don’t clobber good textured walls · **no** gap-fill / planarize / densify

Pack: `/workspace/retexture-bare-facades-rd.md`.

---

## What it does

1. Load `recon/planes.json` + `facades.mtl`.
2. Identify **bare** = missing `textures/facade_XX.jpg` **or** MTL without `map_Kd`.
3. For each bare index: try top-K cameras ranked by frontal×coverage (prefer `view_indices` when present).
4. Softened `_warp_facade_texture`: retry margin 40→120, then clip UV to frame.
5. Stage as `facades.candidate.*` / `textures/candidate/` **or** `--in-place` with `*.pre_retex` backup.
6. Quality-keep: if textured rises and plane count unchanged → promote (clause 1).

Other textured JPGs are copied / left untouched (mtime/hash preserved on the keep set).

---

## Usage

```bash
# diagnose
uv run ps1hood facades-retexture smoke-dense --dry-run

# bake bare only → candidate, then quality-keep promote
uv run ps1hood facades-retexture smoke-dense --bare-only --candidate

# or in-place with backup
uv run ps1hood facades-retexture smoke-dense --bare-only --in-place
```

---

## Acceptance

- Bare indices gain `map_Kd` + `textures/facade_XX.jpg`
- Already-good textures unchanged (hash)
- Textured count rises; plane count stays put
- No pose / cloud change

## Non-goals

Overlay roof-over-street · free-pose · full extract / gap-fill rerun
