# Studio ENU sculpt MVP

**Date:** 2026-09-20 · Product façades (e.g. 18/18) · complete-scene rank 3  
**Sacred:** sat absolute XY · **no free-pose** · **don’t wipe product without bak** · cloud-first hero (façades overlay; sculpt auto-enables)  
**Non-goal:** Mapillary garage (`roof_099_e0`) — separate

## What shipped

Studio: pick one façade → **TransformControls** in local façade frame:

| Control | Behaviour |
|---|---|
| Translate | along plane **n** only (depth nudge), clamp ±3 m; optional tangent slide ±2 m |
| Scale | non-uniform **w×h** about center (vertical rect) |
| Rotate | **off** in v1 |

On **save+bake**:

1. Snapshot `planes.json` / `facades.obj`+`.mtl` / `textures/facade_*.jpg` → `*.bak_sculpt_<ts>` (+ `textures/bak_sculpt_<ts>/`)
2. Write edited plane (`n`, `d`, `center`, `corners`/`quad`, `width_m`, `height_m`, `sculpt_edit=true`)
3. Rewrite that façade’s OBJ verts (others untouched)
4. Re-bake texture via existing `_warp_facade_texture` from chosen / best frontal **known** pano (soft-fail keeps old JPG)

**Undo:** Studio button or `ps1hood sculpt-undo <run>` restores latest bak.

## Wire

- Viewer `viewer.html`: sculpt toggle · raycast pick · TransformControls · bake-cam dropdown · save/undo
- Entering sculpt **auto-enables** `#togFacades` / `facadeRoot` if currently off (cloud left as-is)
- API: `GET …/planes.json` · `GET …/sculpt/cameras` · `POST …/sculpt` · `POST …/sculpt/undo`
- Python: `reconstruct/sculpt.py`
- CLI: `ps1hood sculpt-apply <run> --plane facade_03 --delta-d 0.4` · `sculpt-undo`

## Acceptance

- Nudge one wall ~1 m along n; sat roofs / cams unchanged
- Resize w×h; texture re-bakes from a real pano when possible
- Reload Studio → edit persists; Undo restores full plane count
- No free-pose; no silent wipe of other planes

## PC how-to

```bash
cd ~/ps1hood   # or your clone
git pull
# hard-reload Studio (cache-bust the module viewer)
uv run ps1hood studio
# open http://127.0.0.1:8765/viewer?run=smoke-dense
# Chrome/Firefox: Ctrl+Shift+R (hard reload) so viewer.html + TransformControls refresh
```

Enable **sculpt** → click a façade → **nudge n** / **resize w×h** → **save+bake** → reload to confirm → **undo** if needed.

## One-liner

*Studio: pick façade → TransformControls along n + resize w×h in ENU → re-bake from known pano → bak + write planes/textures — metre fixes without free-pose or wiping product.*
