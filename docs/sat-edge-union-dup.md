# sat_edge edge-aware union dup (false-dup vs product_lock)

**Date:** 2026-09-20 · Product **18/18** lock · PC PR **#43** follow-up (`ma_added=0`)  
**Sacred:** lock 18 · no clutter flood · sat XY · no free-pose · no densify

Pack: `/workspace/sat-edge-union-drop-rd.md`.

---

## Problem

Filter survivor (`ma_gap_sat_edge`) dies in `a_priority` NMS as a **false dup** of `product_lock`:

```text
dup if |n·n_k| ≥ 0.85
 AND (|d − d_k| < 2.5  OR  |d + d_k| < 2.5)   # opposite-normal
 AND ||c_xy − c_k|| < 6 m
```

On small NL footprints (garage / return): opposite wall or adjacent return often has `n ≈ −n_k`, small `|d+d_k|`, and centers **<6 m** after re-anchor → `ma_added=0` with keep_cap still open.

---

## Fix (tight — no global NMS loosen)

For **`sat_edge` / `ma_gap_sat_edge` vs product_lock / A only**:

1. **Cover** — if `product_plane_covers_edge(k, seed_edge)` → real same-wall dup (suppress).
2. **Else geometric** — `|n·n| ≥ 0.95` AND `|d − d_k| < 1.0` (**drop** `|d + d_k|`) AND `dxy < 3.0`.

`manhattan` / `ma_segment` / other MA keep global **6 m / 2.5 / opposite-d**. Do not raise `max_keep` or weaken all NMS.

### Telemetry

```text
union reject sat_edge edge_id=… vs product_lock[i]
  n·n=… Δd=… d+d=… dxy=… cover=yes|no
ma_pre_nms / ma_added / n_dup_vs_a / remaining
```

### CLI

```bash
--gap-nms-xy 3.0 --gap-nms-d-tol 1.0 --gap-nms-no-opposite
# default on for ma_gap_sat_edge only
```

---

## PC retry (PR #43 flags + sat-edge)

```bash
uv run ps1hood facades smoke-dense --gap-fill --a-source product --gap-seeds sat-edge \
  --sat-aabb-gate 2.0 --gap-min-views 2 --max-gap-adds 3 \
  --sat-edge-reanchor --sat-edge-max-drift 3.0 \
  --sat-aabb-gate-sat-edge 5.0 \
  --gap-nms-xy 3.0 --gap-nms-d-tol 1.0 --gap-nms-no-opposite
```

**Expect:** filter survivor with `cover=no` vs all product planes enters candidate (`ma_added≥1`) under lock-18; same-wall true dups still suppressed; log shows reject vs product plane id when dup.

**Non-goals:** loosen global NMS · force-insert without dup check · densify / free-pose / clutter flood · invent peels for `roof_099_e0` `no_facing_cam`

---

## One-liner

*Survivor died in a_priority NMS as a false dup of product_lock (6 m XY + opposite-d on small footprints); edge-aware dup (cover check, no `d+d_k`, tighter XY) lets sat_edge enter candidate without flooding clutter.*
