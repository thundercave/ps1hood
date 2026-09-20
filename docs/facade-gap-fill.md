# PR-C — Façade gap fill (ZNCC-gated)

**Date:** 2026-09-20 · After PR-A roofs + PR-B densify (~976k)  
**Goal:** close side walls / corners / untextured gaps without peel spam  
**Product bar:** hybrid **10/8/0.418** (quality-keep) · sat XY · no BAG  
**Ship from:** Path α dual-source (`--a-source flow` + MA peels) + existing ZNCC / `a_priority`

Pack: `/workspace/pr-c-facade-gap-fill-recipe.md` · parent [`complete-scene-no-holes-rd.md`](complete-scene-no-holes-rd.md).

---

## What it does

1. **Keep A core** — if `planes.json` exists, lock it as the A family (product 10/8).
2. **Re-texture first** — bake retries untextured product planes (do not drop them).
3. **Gap seeds (hypotheses only):**
   - Manhattan ±90° from street/travel heading @ 6–18 m (side / return walls)
   - Sat roof AABB corner wraps (two headings 90° apart, inset) — *not* BAG
4. **MA peels** on denser cloud → road-center reject (cam corridor ±3 m / sat street mask) → prefer normals ⟂ travel → peel cap ≤32
5. Score seeds + peels with existing Path α ZNCC (`zncc≥0.35`; far-side `min_frontal=0.20`)
6. **`a_priority` union** — keep all A accepts first; add non-dup MA (NMS XY 6 m / split 4 m); `max_keep` 16–18
7. **Quality-keep** (PR #24 clauses) vs live product → promote or `*.candidate`

Sacred: sat = absolute XY · photo multi-view = relative 3D · **no BAG hero** · don’t destroy roofs.

---

## Usage

```bash
# After densify (~976k) + existing hybrid façades:
uv run ps1hood facades smoke-dense \
  --source mapanything --a-source flow \
  --gap-fill --max-planes 16 --zncc-accept 0.35

# Optional PS1 post only after gaps look closed:
uv run ps1hood ps1-facades smoke-dense
```

`--gap-fill` forces dual-arm + `a_priority`. Does **not** wipe product when the union is weaker — writes `facades.candidate.*` / `planes.candidate.json`.

---

## PC after merge (smoke-dense)

```bash
cd ~/ps1hood && git pull
# Expect denser MA cloud from PR-B (~976k) already in recon/
uv run ps1hood facades smoke-dense \
  --source mapanything --a-source flow \
  --gap-fill --max-planes 16 --zncc-accept 0.35
# Optional if promote landed clean side walls:
uv run ps1hood ps1-facades smoke-dense
```

Studio hard-reload; BAG off; look for side/return walls at corners; product must stay ≥10/8 unless quality-keep says promote.

Log greps:

```bash
rg -n "gap_fill|candidate NOT promoted|clause|union_kept|road-center" \
  runs/smoke-dense/logs/*.log 2>/dev/null || true
```

---

See also [`worst-cam-gap-fill.md`](worst-cam-gap-fill.md) for targeted seeds toward compare worst cams (`--worst-cams` / `--worst-from-compare`).

---

## Non-goals

BAG/OSM shells · free-pose · peel-as-hero (peel_max >32) · PS1 polish before gaps closed · hungry cloud clip

---

## One-liner

*Keep 10/8 A core; add ZNCC-gated side/corner seeds from manhattan + denser MA peels; quality-keep promote.*
