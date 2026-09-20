# PR-C — Façade gap fill (tight)

**Date:** 2026-09-20 · After PR #34 densify (~976k)  
**Goal:** close side walls / corners / untextured gaps without peel spam  
**Product bar:** hybrid **10/8/0.418** (quality-keep) · sat XY · no BAG  
**Ship from:** Path α dual-source (`--a-source flow` + MA peels) + existing ZNCC / `a_priority`

---

## Focus (not “more peels everywhere”)

| Gap type | Seed |
|----------|------|
| Side / return walls | Manhattan ±90° from street heading @ 6–18 m (existing A manhattan seeds) |
| Corner wraps | Two headings 90° apart sharing center near building corner (from sat roof AABB corners ± inset) |
| Untextured of the 10 | Re-warp only; don’t drop plane |
| Far-side façades | Prefer MA peels facing away from drive; lower min_frontal slightly (0.20) for those only |

---

## Pipeline

```text
existing product planes.json (10)  ──keep──┐
                                           ▼
A arm: search_photo_consistent_planes(flow xyz)     # full A, not gated planarize scorer
  + extra manhattan / corner seeds from sat roof footprints (hypotheses only)
MA arm: segment_vertical_planes on denser ~976k PLY  # peel for NEW side walls only
        → score_planar_hyps (±n refine, zncc≥0.35)
        │
        ▼
a_priority union: keep all A accepts first
  add MA only if non-dup (NMS XY 6 m; no sibling 4 m vs A)
  max_keep 16–20
        │
        ▼
texture + optional --ps1-rectify
quality-keep vs 10/8/0.418 (PR #24 clauses) → promote or *.candidate
```

---

## Knobs

| Knob | Value |
|------|--------|
| zncc_accept | 0.35 |
| Prefer new planes with `source=manhattan` / `ma_segment` whose normal ⟂ street travel |
| Reject peels with center on road (sat road mask / cam corridor ±3 m) |
| Don’t raise peel_max past ~32 — gap focus via seeds, not volume |
| Re-texture existing 2 untextured before inventing new planes |

---

## CLI sketch
```bash
ps1hood facades <run> --source mapanything --a-source flow \
  --gap-fill --max-planes 16 --zncc-accept 0.35
# gap-fill = manhattan+corner seeds from sat roofs + MA peels; a_priority; quality-keep
```

---

## Acceptance
- Side/corner walls appear where Studio showed gaps  
- Promote only if textured≥8 and planes≥10 under PR #24 rules (or beat mean)  
- No street-slab façades; sat XY unchanged  

## Non-goals
BAG shells · free-pose · peel-as-hero · PS1 until gaps closed (ps1-rectify OK if already default)

---

## One-liner
*Keep 10/8 A core; add ZNCC-gated side/corner seeds from manhattan + denser MA peels; quality-keep promote.*
