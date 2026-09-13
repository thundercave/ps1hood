# Path α — Reclaim A-family planes (a_kept≈4 → historic ~7)

**Audience:** Chief / PC implementers (MessageSubagent → PR)  
**Date:** 2026-09-13 (Europe/Amsterdam)  
**Trigger:** PR #23 `a_priority` hybrid: **a_kept=4, ma_added=2, union 6, mean 0.377** — still short of historic Milestone A product **~7 planes / 5 textured / mean ZNCC ~0.42**. Earlier hybrid `nms`: a_kept=2–3.  
**Sacred:** photo-derived geometry; no OSM/BAG invent. Do **not** dig peels / Open3D / α2 patches (parked). Seed/scoring under hybrid only. Quality-keep (post-#23 tighten) stays.

**Code reviewed (`gh` /tmp — no clone):**
- `/tmp/facades23.py` — hybrid injects `hypothesize_vertical_planes` → `score_planar_hyps` (full `search_photo_consistent_planes` **only** on 0-accept fallback)
- `/tmp/planarize23.py` — `score_planar_hyps`, `union_a_priority` / `union_keep_planes`, NMS, extra gates
- `/tmp/pp23.py` (= `photo_planes.py`) — `hypothesize_vertical_planes`, `search_photo_consistent_planes` full loop

**Prior packs:** `path-alpha-more-accepts-rd.md`, `path-alpha-zncc-fail-rd.md`, `path-alpha-zncc-neg1-and-quality-keep.md`  
**Compare §:** append §17 (this reclaim).

---

## 0) Executive call (for Chief)

| Question | Answer |
|----------|--------|
| Why a_kept≈4 ≪ historic ~7? | Hybrid **never runs A's full search loop** — only seeds → `score_planar_hyps` (extra gates). Historic 7 came from `search_photo_consistent_planes`. |
| Is NMS / max_keep / zncc the cutter? | **No.** Same `zncc_accept=0.35`, same NMS radii (6 m / 0.85 n / 2.5 d). Hybrid `max_keep=16` is **looser** than A's default 12. |
| Top fix? | **A arm = full `search_photo_consistent_planes`**, then add non-dup MA via existing `a_priority` union. |
| Acceptance | Smoke: **a_kept ≥ 6–7** (or match `--no-planarize` control on same frames); union may still add MA; quality-keep must not demote 7/5/0.42 on mean regress. |

**One-liner:** *Stop scoring A through Path α's gated hybrid scorer; run Milestone A's full search as the A arm, then union non-dup MA.*

---

## 1) Stippy digest (confirmed deltas)

| Item | Historic Milestone A | Hybrid Path α (PR #21–23) |
|------|----------------------|---------------------------|
| Entry | `search_photo_consistent_planes` | `hypothesize_vertical_planes` → inject → `score_planar_hyps` |
| `zncc_accept` | **0.35** | **0.35** (same) |
| NMS radii | n·n≥0.85, Δd&lt;2.5, XY&lt;**6 m** | **Same** (split siblings 4 m — A seeds width=8 **not** split) |
| `max_keep` | default **12** | hybrid **16** (looser — cannot explain 7→4) |
| Refine deltas | (−2,−1,+1,+2) only for `heading_distance`/`manhattan` | (−2,−1,−0.5,+0.5,+1,+2) **+ opposite-n** for all sources |
| Extra gates | **none** (view picker dist 2–35 only) | **tilt** \|n·UP\|&gt;0.20; **xy40**; hard cam [1,50]; soft cam [2,35] per refine cand |
| Normal handling | picker orients locally | `_flip_n_toward_cams` pre-pick + flip to ref cam + opposite-n candidates |
| When full A runs | always (non-planarize) / fallback if α keeps 0 | **skipped** whenever hybrid keeps ≥1 |

**Stippy bullets (fold-in):**
1. Same zncc 0.35 & NMS radii → count gap is **not** threshold/NMS tightening.
2. Hybrid max_keep=16 is looser than A’s 12 → **not** max_keep cutting 7→4.
3. Historic 7 = full `search_photo_consistent_planes`; hybrid = hypothesize→`score_planar_hyps` with **extra tilt/cam/xy40 gates + opposite-n**.
4. Prefer fix: **A arm = full `search_photo_consistent_planes`, then add non-dup MA.**

---

## 2) PC telemetry (facts)

| Run | Signal | a_kept | Notes |
|-----|--------|--------|-------|
| Historic product | flow-era / photo_consistency | **~7** | 5 textured, mean ZNCC ~0.42 |
| PR #16 fallback | full `search_photo_consistent_planes` after α=0 | **2** | mean 0.377 — **same-run full A ≠ 7** (see hyp 6) |
| PR #21 hybrid `nms` | `ma=24 a=165 → windows=344; pre_nms=10 → kept 5` | **3** | ma_kept=2; mean 0.399 |
| PR #22 A2 `nms` | `pre_nms=15 → kept 5` | **2** | ma_kept=3; NMS starves A |
| PR #23 `a_priority` | union 6 / 6 textured / mean **0.377** | **4** | ma_added=2; promote hole (fixed quality-keep) |

**Read:** `a_priority` recovered A from 2–3 → 4 by protecting A from MA eviction. Gap to 7 is **upstream of union** (A ZNCC accepts under hybrid scorer / seed path), not intra-A NMS with max_keep=16.

---

## 3) Ranked causes

| Rank | Cause | Likelihood | Evidence | Why a_kept≈4 |
|------|--------|------------|----------|--------------|
| **1** | **Hybrid A arm ≠ full Milestone A search** | **Very high** | `facades23.py` ~715–748: `hypothesize` → `score_planar_hyps(seed_hyps=…)`. Full `search_photo_consistent_planes` only at fallback when `not accepted` (~784–795). Historic product used the full loop. | A seeds scored under α scorer; missing dedicated A search packaging even though refine is *richer*. |
| **2** | **Extra tilt / cam-depth / xy40 gates in `score_planar_hyps`** | **High** | `planarize23.py` ~907–918, 948–951: tilt, xy40, hard [1,50], soft [2,35] per candidate. `search_photo_consistent_planes` has **no** equivalents — only `_pick_scoring_views` dist∈[2,35]. | Borderline A heading/manhattan/sparse hyps skipped before ZNCC; lowers a_pre_nms. |
| **3** | **Historic 7 not apples-to-apples with current MA hybrid run** | **High (confirm)** | Product = flow-era / `--no-planarize` / flow PLY seeds. PR #16 **full** A fallback on MA-context run kept only **2**. PC #21: 165 A hyps → ~3–4 A accepts. | Expectation “hybrid A should equal 7” overstates; control = `--no-planarize` on **same frames**. Still reclaim toward that control + historic code path. |
| **4** | **Opposite-n + pre-flip change view sets vs A loop** | **Medium** | Hybrid: flip toward cams → refine incl. −n → re-pick views per cand. A search: pick views once on seed n, refine ±n only (no opposite). | Can change which patches win; may merge/alter accepts vs pure A (usually helps MA; uncertain for A parity). |
| **5** | **Sparse xyz seed source differs (MA dense @0.08 vs flow @0.20)** | **Medium** | Hybrid passes planarize `xyz` voxel_m (0.08) into `hypothesize` (`facades23.py` ~649, 718). Non-planarize A uses 0.20 on flow/recon PLY. Sparse PCA seeds ≤16. | Different sparse hyps; heading×distance grid identical — sparse is the only seed inventory delta. |
| **6** | **`max_heading_seeds=24` / distance grid too thin** | **Low** | Same defaults as `hypothesize` in A (`distances_m=(6,10,14,18,22)`). PC: **a=165** seeds — inventory ample. | Not the bottleneck; raising seeds alone won’t close 4→7 if gates kill accepts. |
| **7** | **`max_keep` / NMS on A family cuts 7→4** | **Ruled out** | Same NMS radii; hybrid max_keep **16 ≥** A’s 12. `a_priority` keeps A first. a_kept=4 ⇒ a_pre_nms ~4–6, not 7+ cut down. | Union is not the cutter post-#23. |
| **8** | **zncc_accept / patch / view picker differ** | **Ruled out (zncc/patch); low (picker)** | zncc 0.35 same; patch 64 same (clamp only if w/h&gt;12 — A width=8). Picker is shared `_pick_scoring_views`; hybrid re-picks after flip (see #4). | Threshold not the gap. |

---

## 4) Code delta: full A search vs hybrid score path

### `search_photo_consistent_planes` (historic A)

```
hyps = hypothesize_vertical_planes(frames, xyz, …)  # heading + manhattan + sparse
for hyp in hyps:
    idxs = _pick_scoring_views(n, center)  # once
    candidates = [(n,d,c)] + ±{1,2}m along n   # only heading_distance|manhattan
    score_vertical_plane(…)
NMS (0.85 / 2.5 / 6.0), max_keep (default 12)
# NO tilt / xy40 / hard|soft cam gates / opposite-n
```

### Hybrid today (`extract_facades` Path α)

```
ma_hyps = planes_from_mapanything_ply(…)          # peels — out of scope for this pack
seed_hyps = hypothesize_vertical_planes(…)        # same seeder
accepted = score_planar_hyps(ma_hyps, frames,
             seed_hyps=seed_hyps, …)               # ALL through α scorer
  → expand splits (A width=8 → no split)
  → gates: tilt, xy40, hard/soft cam
  → flip n; refine ±{0.5,1,2} + opposite-n
  → re-pick views per candidate
  → union_keep_planes(a_priority)
# full search_photo_consistent_planes ONLY if accepted==[]
```

**Conclusion:** Hybrid is not “missing refine” (it has **more**). It is **missing the ungated A search arm** and subject to α-only reject gates.

---

## 5) Concrete fix (recommended PR)

### ★ Fix A — Dual arm (prefer)

In `extract_facades` when `use_planarize and hybrid_heading`:

```python
# A arm — byte-parity with historic Milestone A
accepted_a = search_photo_consistent_planes(
    frames,
    xyz if len(xyz) >= 30 else None,  # consider flow/recon PLY for sparse if available
    zncc_accept=min(float(zncc_accept), 0.35),
    ground_z=ground_z,
    max_keep=n_planes,  # 16 ok; A internal default was 12
)

# MA arm — peels only (no A seed inject)
accepted_ma = score_planar_hyps(
    hyps,  # MA peels
    frames,
    zncc_accept=float(zncc_accept),
    max_keep=n_planes,
    seed_hyps=None,  # ← stop double-scoring A through gated path
    split_trigger_width_m=split_trigger,
    split_window_m=split_window,
    split_overlap_m=split_overlap,
    nms_xy_m=nms_xy,
    nms_xy_split_m=nms_xy_split,
    union_strategy="nms",  # MA-only NMS; final union below
)

# Union — reuse a_priority semantics
accepted = union_keep_planes(
    list(accepted_a) + list(accepted_ma),
    strategy="a_priority",
    max_keep=n_planes,
    nms_xy_m=nms_xy,
    nms_xy_split_m=nms_xy_split,
)
```

**Telemetry:** log `a_arm=search_photo_consistent_planes kept=…; ma_arm=…; a_kept=… ma_added=…`.

**Fallback:** if both arms empty, keep today’s fail-loud / keep-previous path (no need for nested full-A retry).

### Fix B — Bypass α gates for A sources (smaller, weaker)

Inside `score_planar_hyps`, if `hyp["source"] in {heading_distance, manhattan, sparse}`: skip tilt/xy40/soft-depth (or only apply hard extremes); optionally disable opposite-n for A to match A loop. Still not true parity with view-once + refine-only-heading.

### Fix C — Control smoke (measurement, not product)

```bash
ps1-hood facades <smoke> --no-planarize --zncc-accept 0.35
# Expect a_kept / planes ≈ historic control on THESE frames (may be <7 if not flow-era).

ps1-hood facades <smoke> --planarize --source mapanything --zncc-accept 0.35 \
  --union-strategy a_priority --max-planes 16
# After Fix A: a_kept ≥ max(6, control_planes) ideally; ma_added ≥ 0 non-dup.
```

### Out of scope (parked)
- Peel / Open3D / `detect_planar_patches` / α2
- Lowering `zncc_accept`
- Weakening quality-keep (post-#23 rule stays)

---

## 6) Acceptance

- [ ] Hybrid Path α smoke: **a_kept ≥ 6–7** *or* a_kept ≥ `--no-planarize` control on same frames (whichever is the honest ceiling).
- [ ] Log shows A arm used **`search_photo_consistent_planes`**, not only seed inject.
- [ ] `ma_added` only non-dup vs kept A (`a_priority`).
- [ ] Mean ZNCC of union does not demote product under tightened `_is_strictly_better` (planes↓ or mean−eps blocked).
- [ ] Unit: `test_hybrid_a_arm_calls_full_search` (spy); `test_union_a_priority_after_dual_arm`.

---

## 7) PR sketch

| Change | File |
|--------|------|
| Dual-arm hybrid: full A search + MA-only `score_planar_hyps` + `union_keep_planes(a_priority)` | `facades.py` |
| Optional: expose `--hybrid-a-arm {full_search,seeds}` default `full_search` | `cli.py` |
| Telemetry `a_arm=… a_kept=…` | `facades.py` / `planarize.py` |
| Tests as above | `tests/…` |

**Do not** re-inject A seeds into `score_planar_hyps` when A arm is `full_search` (double work + gated rejects).

---

## 8) Hypotheses checklist (user list → verdict)

| # | Hypothesis | Verdict |
|---|------------|---------|
| 1 | Hybrid only runs seeds through `score_planar_hyps`, missing full search | **#1 cause — confirm** |
| 2 | `max_heading_seeds=24` / distance grid | **Low** (a=165 already) |
| 3 | `max_keep` / NMS cuts 7→4 | **Ruled out** (same radii; max_keep looser) |
| 4 | zncc / patch / view picker differ | **zncc/patch same; picker re-pick = medium via opposite-n** |
| 5 | xyz sparse seed missing / different | **Medium** (MA @0.08 vs flow @0.20) |
| 6 | Historic 7 different CLI (`--no-planarize` / flow PLY) | **High confirm** — run control; PR #16 full A got 2 |

---

*End. Top fix → parent: A arm = `search_photo_consistent_planes`, then non-dup MA via `a_priority`.*
