# Gap-fill stricter multi-view (gap adds only)

**Date:** 2026-09-20 · After PR #40 (18→23 promote; compare façade mean ↓)  
**Goal:** stop clause-2 from promoting volume that hurts global compare  
**Sacred:** sat XY · no free-pose · don’t auto-revert live 23 (PC triage of +5 is separate)  
**Out:** densify · free-pose · compare-aware keep (pack §3 C — next)
**Follow-up shipped:** [`sat-edge-gap-seeds.md`](sat-edge-gap-seeds.md) (`--gap-seeds sat-edge`); [`sat-edge-reanchor.md`](sat-edge-reanchor.md) (re-anchor + seed-edge AABB)

Packs: `/workspace/gap-fill-compare-regress-rd.md` §3 B · `/workspace/gap-fill-seed-rethink-rd.md` §2 A.

---

## What it does

For **gap adds only** (not `product_lock`):

1. **Multi-view** — need ZNCC ≥ 0.35 on ≥ `--gap-min-views` cams (ref + source pair scores); reject if median of finite pair scores < 0.10; reject grazing if max `|n · cam_fwd|` over scoring cams < 0.4.
2. **Sat AABB hard gate** — center within `--sat-aabb-gate` m (default 2.0) of a sat roof footprint *boundary* edge, with plane normal aligned to that edge’s outward (`|n · n_edge| ≥ 0.7`). Soft-pass when no roofs on disk.
3. **Cap MA peels** — `--ma-peel-cap` default **3** (was 32).
4. **Cap gap adds** — `--max-gap-adds` default **3**; prefer non-`ma_segment` when ranking.

Quality-keep / product lock unchanged. `--gap-seeds sat-edge` (default) invents uncovered roof-edge hyps; zero facing cams → `recon/gap_needs.json` — see [`sat-edge-gap-seeds.md`](sat-edge-gap-seeds.md).

---

## Usage

```bash
uv run ps1hood facades smoke-dense --gap-fill --a-source product \
  --gap-seeds sat-edge \
  --sat-aabb-gate 2.0 --ma-peel-cap 3 --gap-min-views 2 --max-gap-adds 3 \
  --zncc-accept 0.35 --max-planes 24
```

`--sat-aabb-gate 0` disables the footprint gate (multi-view + caps still apply).

---

## Non-goals

Densify · free-pose · wipe / auto-revert live product · compare-aware promote (§3 C)

---

## One-liner

*Gap adds must clear multi-view ZNCC + sat roof-edge AABB and stay ≤3 — product_lock untouched.*
