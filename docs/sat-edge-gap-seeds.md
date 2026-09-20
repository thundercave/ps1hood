# sat_edge gap seeds

**Date:** 2026-09-20 · Product **18/18/0.440** (lock) · depends on PR **#41**  
**Sacred:** lock 18 · sat XY · no free-pose · no wipe · no densify

**Follow-up:** [`sat-edge-reanchor.md`](sat-edge-reanchor.md) — re-anchor after ZNCC, seed-edge AABB, prefer returns, optional sat_edge-only soft gate.

Pack: `/workspace/sat-edge-gap-seeds-rd.md`.

---

## What it does

1. **Uncovered roof AABB edges** — per sat roof/building footprint, four AABB sides. An edge is *covered* if some product plane has `|n · n_out| ≥ 0.7`, `|d + n·M| ≤ 2 m`, and center within 2 m of the segment. Uncovered edges become candidates.
2. **Same-side facing cams → `sat_edge` hyps** — cam dist 6–35 m, looking at midpoint (`(v/‖v‖)·fwd ≥ 0.5`), not grazing (`|n_out·fwd| ≥ 0.35`), outside the building (`(C − roof_c)·n_out ≥ 0`). Prefer return walls (`|heading − travel| ≈ 90°`).
3. **`gap_needs.json`** — if an edge has **zero** facing cams, append `recon/gap_needs.json` (do **not** invent MA peels). Hint: fetch far-side SV panos at fixed poses.
4. **Score under #41 gates** — road-reject → sat AABB → `--gap-min-views 2` / grazing / median → `a_priority` add ≤ `--max-gap-adds 3` → quality-keep vs product 18. Ranking: `sat_edge` > manhattan/worst > `ma_segment`.

---

## CLI

```bash
uv run ps1hood facades smoke-dense --gap-fill --a-source product --source mapanything \
  --gap-seeds sat-edge \
  --sat-aabb-gate 2.0 --gap-min-views 2 --ma-peel-cap 3 --max-gap-adds 3
```

| `--gap-seeds` | Behaviour |
|---|---|
| `sat-edge` (**default**) | Uncovered AABB edges + facing cams only |
| `legacy` | manhattan + corner_sat (+ worst-cam) |
| `both` | sat_edge first, then AABB-gated legacy |

---

## Non-goals

Densify · free-pose · auto-wipe of product 18 · inventing MA peels for `gap_needs`

---

## One-liner

*Uncovered sat roof edges → `sat_edge` hyps from same-side facing cams; else `gap_needs.json`; score under PR #41 gates; wire `--gap-seeds sat-edge`.*
