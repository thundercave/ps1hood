# sat_edge re-anchor + seed-edge AABB (PR follow-up to #42)

**Date:** 2026-09-20 · Product **18/18** lock · PC: promote NO · 1 ZNCC accept then `sat_aabb reject dist=4.96`  
**Sacred:** lock 18 · no free-pose · no densify · don’t promote clutter · don’t widen AABB for MA peels

Pack: `/workspace/sat-edge-aabb-reject-rd.md`.

---

## Problem

ZNCC ±n refine walks a `sat_edge` seed off its roof AABB segment (~5 m). The AABB gate then measured **global nearest** roof edge — often a *different* building — and rejected with a mystery `dist≈4.96`. Separately, uncovered edges sliced `[:8]` with **no return-wall priority**, so garage returns never entered the seed list (`needs=0` but garage empty). Cover used center≤**4 m**, so a front wall could “cover” a return.

---

## Fixes shipped

1. **Re-anchor after score (`--sat-edge-reanchor`, default on)** — for `source` containing `sat_edge` only, after ZNCC accept / before AABB: keep refined `n`, set `center_xy` to closest point on seed edge segment, `d ← -n·center`. Reject `refine_drift` if `|d_refined − d_anchor| > --sat-edge-max-drift` (default **3.0 m**).
2. **AABB to seed `edge_id`** — `sat_aabb_edge_ok(..., edge_id=…)` measures distance to **that** segment, not global nearest. Reject/ok why includes `dist_seed_edge` + `dist_nearest_any`.
3. **Uncovered prefer returns** — sort by `|heading−travel|≈90°` before slice; `max_edges` **8→16**; cover center **4→2 m**; log covered-by + uncovered edge_ids.
4. **Soft AABB 2→5 for sat_edge only (last resort)** — `--sat-aabb-gate-sat-edge 5.0`. Does **not** widen for `ma_segment` / manhattan / worst_cam.

---

## CLI (PC retry — same sat-edge gap-fill as #42)

```bash
uv run ps1hood facades smoke-dense --gap-fill --a-source product --gap-seeds sat-edge \
  --sat-aabb-gate 2.0 --gap-min-views 2 --max-gap-adds 3 \
  --sat-edge-reanchor --sat-edge-max-drift 3.0
# optional escape only if A+B still starve:
#   --sat-aabb-gate-sat-edge 5.0
```

**Acceptance:** ZNCC-accept sat_edge either re-anchors onto its edge and adds, or rejects `refine_drift` — never a `dist≈5` mystery. Garage return appears in uncovered (or explicit cover log). Product stays 18 unless a same-building add promotes cleanly.

**Follow-up:** [`sat-edge-union-dup.md`](sat-edge-union-dup.md) — edge-aware a_priority dup (no opposite-d / tighter XY vs product_lock).

**Non-goals:** widen gate for MA peels · free-pose · wipe 18 · invent peels when needs>0 · densify / clutter promote

---

## One-liner

*ZNCC walked sat_edge ~5 m off the seed; re-anchor to the segment (cap drift), gate against that edge_id, prefer return edges in the uncovered list — don’t just bump 2→5 for everyone.*
