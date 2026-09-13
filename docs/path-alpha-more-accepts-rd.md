# Path α — More ZNCC accepts after PR #20 (R&D → PR)

**Audience:** Chief / PC implementers (MessageSubagent → PR)  
**Date:** 2026-09-13 (Europe/Amsterdam)  
**Trigger:** Path α after **PR #20** (`zncc_accept` MA default **0.35**): kept **1 / 24** MA hyps at **ZNCC=0.439** (peel+threshold **WORKS**). Quality-keep correctly refused promote (1 textured ≪ product **7/5/0.42**). Goal: raise **number** of MA peels that pass ZNCC≥0.35 so Path α (or hybrid A+MA) can beat product.

**Sacred:** photo-derived geometry; no OSM/BAG invent. Never demote better product. Quality-keep stays.

**Code reviewed (main @ PR #20 merge `188120b`, via `gh api` — no clone):**
- `src/ps1_hood/reconstruct/planarize.py` — peel, `split_long_hyp`, `score_planar_hyps`
- `src/ps1_hood/reconstruct/facades.py` — `extract_facades`, `_is_strictly_better`
- `src/ps1_hood/reconstruct/photo_planes.py` — `hypothesize_vertical_planes`, `search_photo_consistent_planes` NMS

**Prior packs:** `path-alpha-planarize-recipe.md`, `path-alpha-zncc-fail-rd.md`, `path-alpha-zncc-neg1-and-quality-keep.md`

---

## 0) Executive call (for Chief)

| Question | Answer |
|----------|--------|
| Is Path α “broken”? | **No** — 1 accept @ 0.439 proves ZNCC path + 0.35 threshold work. |
| Why not promote? | Quality-keep: **1 textured < 5** on product 7/5/0.42 — correct. |
| Why only 1/24? | Mostly **peel inventory** (few façade-sized, frontal-centered hyps) + **A seeds never mixed** when MA keeps ≥1; split defaults still coarse; peel budget spends on large slabs. |
| First PR? | **Splits + hybrid A seeds into same scorer + union(A,MA) promote** — param sweep second. |
| Success bar | Candidate textured ≥5 **or** mean_zncc beats 0.42 at same/higher count → promote over product. |

**One-liner:** *Stop treating MA peels and A heading seeds as mutually exclusive; split slabs denser; promote the NMS union if it beats product.*

---

## 1) Why only 1/24 passed (ranked hypotheses)

PC signal: **24 MA peels → 1 keep @ ZNCC 0.439**; product still **7 planes / 5 textured / mean 0.42**.

| Rank | Cause | Likelihood | Evidence (code + PC) | Why it yields ~1 accept |
|------|--------|------------|----------------------|-------------------------|
| **1** | **Hybrid A seeds never run when MA keeps ≥1** | **Very high** | `extract_facades`: `if not accepted and fallback_heading:` only (`facades.py` ~636). With 1 MA accept, Milestone A `hypothesize_vertical_planes` / `search_photo_consistent_planes` **skipped**. Product’s 7 walls came from A-style heading×distance historically. | MA alone must discover every façade; A’s proven seeds sit idle → count stuck at 1. |
| **2** | **Peel budget = largest verticals first** | **High** | `segment_plane` / numpy RANSAC peels **biggest** inlier sets first; `max_planes=24` / `min_inliers=400` (`planarize.py` defaults). Long street-parallel slabs consume slots before smaller side façades. | Many of 24 hyps share near-identical `n` along the drive; only one window aligns with SV → 1 ZNCC pass. |
| **3** | **Split exists but coarse / late** | **High** | PR #19: `DEFAULT_SPLIT_TRIGGER_WIDTH_M=12`, `WINDOW=10`, `OVERLAP=2` (`planarize.py` 46–48). `inliers_to_quad` already clamps **max_width=25** about AABB center — then split only if width > 12. Peels ≤12 m **never** split; 12–25 m → ~2–3 windows sharing same `(n,d)`. | Too few façade-local centers along a block; mid-slab windows still weak ZNCC. |
| **4** | **NMS collapses overlapping split siblings** | **Medium–high** | `score_planar_hyps` NMS: `|n·n_k|≥0.85` and `|d−d_k|<2.5` and XY center `<6 m` (`planarize.py` 791–803). Overlapping 10 m windows on same wall often `<6 m` apart → **one survivor** even if several scored ≥0.35. | Inflates “1/24 kept” relative to raw accepts before NMS. |
| **5** | **Score clamp 12 m ignores split footprint** | **Medium** | Scoring uses `w = min(width_m, 12)` but kept plane stores **full** `hyp["width_m"]` (`planarize.py` 675–765). Split pieces are ~8–10 m — OK — but unsplit ≤12 peels score a crop of a possibly wrong center. | Weak ZNCC on unsplit peels; one lucky center passes. |
| **6** | **Peel params favor thick street sheets** | **Medium** | `voxel=0.08`, `distance_threshold=0.08`, `min_inliers=400`, `residual_stop=1500`, `vertical_dot=0.15`. Loose distance + high min_inliers → fewer, fatter planes. CLI exposes only `--voxel` / `--plane-dist` / `--max-planes` — **not** min_inliers / residual / vertical_dot / split knobs. | Hard to get more façade-like peels without a smoke grid. |
| **7** | **Quality-keep bar is correct but high** | **Confirmed (product)** | `_is_strictly_better`: need **more textured** than 5, or same textured + more planes / +0.02 mean_zncc (`facades.py` 403–459). 1 @ 0.439 cannot promote — by design. | Not a ZNCC bug; blocks weak Path α from replacing product. |
| **8** | **View picker / grazing on street-normal peels** | **Low–medium** | `_pick_scoring_views(..., min_frontal=0.25)`. Street-parallel peels can still score (finite 0.439 on one) but most stay below 0.35. | Explains many near-misses, not zero. |
| **9** | **Threshold 0.40 was the prior wall** | **Resolved by #20** | PR #20 set `DEFAULT_ZNCC_ACCEPT_MA=0.35`. PC: 0.439 ≥ 0.35 → 1 keep. | Unlocks accepts; does not multiply them. |

**Bottom line:** Path α scoring works. The bottleneck is **hypothesis diversity + coverage** (peel+split inventory) and **missing hybrid with A**, not the ZNCC kernel. Quality-keep will promote once textured_count (or mean) beats 7/5/0.42.

### PC log greps (confirm before coding)

```bash
rg -n "planarize: kept|expanded|FAIL-LOUD|Path α:|candidate NOT promoted|photo_consistency" \
  runs/<smoke>/logs/*.log

# Expect something like:
# planarize: expanded N peels → M score windows
# planarize: kept 1 / 24 MA hyps (ZNCC≥0.35, mean=0.439, ...)
# facades: candidate NOT promoted — fewer textured (1<5) ...
```

Also log (add if missing): pre-NMS accept count, split window count, per-hyp best ZNCC histogram (bins −0.2…0.6).

---

## 2) Concrete peel param knobs + recommended smoke grid

### Current defaults (post–#20)

| Knob | Default | Where | CLI today |
|------|---------|-------|-----------|
| `voxel` | 0.08 m | `planes_from_mapanything_ply` | `--voxel` |
| `distance_threshold` | 0.08 m | peel | `--plane-dist` |
| `min_inliers` | 400 | peel | `--min-inliers` (PR-2) |
| `residual_stop` | 1500 | peel | `--residual-stop` (PR-2) |
| `vertical_dot_max` | 0.15 | peel | `--vertical-dot` (PR-2) |
| `max_planes` | 24 (peel) / 12 keep | peel / score | `--peel-max-planes` / `--max-planes` (PR-2) |
| NMS XY | 6 m / **4 m** split siblings | score | `--nms-xy` / `--nms-xy-split` (PR-2) |
| `aabb_percentile` | 5–95 | `inliers_to_quad` | no |
| `max_width/height` | 25 / 15 | quad | no |
| `min_height` | 2.5 | quad | no |
| `split_trigger / window / overlap` | 8 / 8 / 2 | `split_long_hyp` | `--split-*` (PR-1) |
| `zncc_accept` | **0.35** | score | `--zncc-accept` |
| `refine_deltas` | ±0.5…2 m | score | no |

### Recommended smoke grid (param sweep = **second** PR)

Run on smoke block with product bak restored; `--keep-previous-on-fail` on; log kept / pre-NMS / mean_zncc / textured after bake.

**Grid A — peel aggressiveness (8–12 runs):**

| # | voxel | plane_dist | min_inliers | residual_stop | vertical_dot | max_planes (peel) |
|---|-------|------------|-------------|---------------|--------------|-------------------|
| A0 | 0.08 | 0.08 | 400 | 1500 | 0.15 | 24 | ← baseline (PC 1/24) |
| A1 | 0.06 | 0.06 | 300 | 1000 | 0.15 | 32 |
| A2 | 0.05 | 0.05 | 250 | 800 | 0.18 | 40 |
| A3 | 0.08 | 0.05 | 250 | 1000 | 0.15 | 32 |
| A4 | 0.10 | 0.10 | 500 | 2000 | 0.12 | 24 | ← coarser control |
| A5 | 0.06 | 0.08 | 200 | 800 | 0.20 | 40 |

**Hypothesis:** A1/A2/A3 → more, smaller verticals → more ZNCC passes; A4 → fewer/fatter (worse).

**Grid B — split only (hold peel at A0 or best A):**

| # | trigger | window | overlap | notes |
|---|---------|--------|---------|-------|
| B0 | 12 | 10 | 2 | current |
| B1 | **8** | **8** | **2** | denser façade windows ★ |
| B2 | **8** | **10** | **3** | wider + more overlap |
| B3 | 6 | 8 | 2 | aggressive |
| B4 | 12 | 8 | 2 | split earlier length, smaller crops |

**Wire CLI (minimal for smoke):**  
`--min-inliers`, `--residual-stop`, `--vertical-dot`, `--split-trigger`, `--split-window`, `--split-overlap`, `--peel-max-planes` (distinct from keep `max-planes`), `--nms-xy` / `--nms-xy-split` (PR-2).

**Do not** lower `zncc_accept` below 0.35 for product runs (smoke may try 0.30 for telemetry only).

---

## 3) Split recipe (overlapping 8–12 m windows along `right`)

### Intent

Street-slab peels have correct-ish `(n,d)` but **wrong center** for SV. Slide façade-sized windows along the in-plane `right` axis so centers sit on real house fronts.

### Recipe (replace / tighten defaults)

```text
trigger_width_m  = 8.0     # was 12 — also split medium peels
window_m         = 8.0…10  # façade-sized (NL row houses ~6–10 m)
overlap_m        = 2.0…3.0
min_piece_m      = 1.5     # already in split_long_hyp
max_width clamp  = 25      # keep inliers_to_quad; split AFTER clamp
```

### Algorithm (already in `split_long_hyp` — change defaults + call sites)

```text
for each vertical peel hyp:
  if width_m <= trigger: emit hyp as-is
  else:
    origin = corners mean; right = normalize(BR−BL); up = normalize(TL−BL)
    s from s_min → s_max along right
    step = window − overlap
    for s = s_min; s < s_max; s += step:
      e = min(s+window, s_max)
      if e−s < min_piece: break
      center_i = project_to_plane(origin + mid_s*right + mid_t*up)
      emit hyp copy: same n,d; new center/corners/width; source=ma_segment; split_parent=True
```

### Extra (first PR, small)

1. **Always expand** peels with `width_m > trigger` (already).  
2. Optionally **force one window per ~8 m of inlier span before AABB clamp** if raw percentile width > 25 (today clamp hides true length — consider splitting on **pre-clamp** width stored as `raw_width_m`).  
3. Cap expanded windows (e.g. `max_score_windows=64`) to bound ZNCC cost.  
4. After split, **relax NMS XY** for `split_parent` siblings: e.g. XY dup radius **3.5–4 m** (was 6) so adjacent façades on same plane can both keep — **or** NMS on scored center after clamp-w only.

### Tests

- Synthetic 30 m × 8 m peel → ≥3 windows with centers ~8 m apart, same `n,d`.  
- Peel width 7 m → **no** split when trigger=8.  
- Overlap: consecutive windows share ≥2 m along right.

---

## 4) Hybrid seed: inject A heading×distance into `score_planar_hyps`

### Gap

Today: MA peels **or** (if 0 accepts) full Milestone A search. Never **both** in one ZNCC pass.

### Design

```text
hyps_ma = planes_from_mapanything_ply(...)           # existing peel
hyps_a  = hypothesize_vertical_planes(frames, xyz,   # photo_planes
              distances_m=(6,10,14,18,22),
              width_m=8, height_m=9, ground_z=...)
# tag sources: ma_segment vs heading_distance / manhattan / sparse

hyps = hyps_ma + hyps_a   # or score_planar_hyps(ma, extra_hyps=a)

accepted = score_planar_hyps(
    hyps,
    frames,
    zncc_accept=0.35,
    split_long=True,   # only splits wide MA peels; A already ~8×9
    refine=True,       # already refines all sources in score_planar_hyps
)
# existing NMS inside score_planar_hyps
```

### Wiring options (prefer B)

| Opt | Where | Pros |
|-----|-------|------|
| **A** | `extract_facades`: concat lists then one `score_planar_hyps` | Minimal; one NMS |
| **B** | `score_planar_hyps(..., seed_hyps=None)` appends A seeds | Keeps α scoring + refine/split in one place ★ |
| **C** | Always run A search + MA score then union | Two refine loops; more CPU; clearer provenance |

**Recommend B:** inside `score_planar_hyps` or a thin `score_planar_hyps_hybrid` called from `extract_facades` when `hybrid_heading=True` (default **on** for Path α).

### Notes

- A hyps already have `width_m=8`, `height_m=9` — `split_long` no-ops.  
- `score_planar_hyps` already refines **all** sources (unlike A’s `search_photo_consistent_planes`, which only refines `heading_distance`/`manhattan`). Good — keep.  
- Cap A seeds (`max_heading_seeds=24`) so MA windows dominate compute.  
- Log: `hybrid: ma=N a=M → windows=W → kept=K (ma_kept=… a_kept=…)`.

### Fallback behavior

- `fallback_heading`: only if hybrid still 0 accepts (legacy).  
- Or deprecate fallback when hybrid default-on (same coverage, one path).

---

## 5) Promote rule: union(A, MA) with NMS

### Goal

Promote when **combined** set beats product — even if MA-alone is 1 plane.

### Algorithm

```text
accepted_ma = score(...) on MA (± split)     # or single hybrid pass → tagged sources
accepted_a  = ...                            # if not already in hybrid pass
union = NMS( sort_by_zncc( accepted_ma ∪ accepted_a ) )
  # reuse score_planar_hyps NMS (optionally XY=4 m for split siblings)

new_q = {
  textured: count after bake (or len(union) pre-bake optimistic),
  plane_count: len(union),
  mean_zncc: mean(zncc of union),
  source: "mapanything_hybrid" if any ma and any a else ...
}

if _is_strictly_better(new_q, prev_q):  # existing
  promote
else:
  write facades.candidate.* / planes.candidate.json; keep product
```

### Why this beats “MA only then fallback”

| Mode | Smoke expectation |
|------|-------------------|
| MA only | 1 @ 0.439 → no promote |
| A fallback only (α=0) | historically 2–7; may still lose to 7/5/0.42 if weak |
| **union hybrid** | MA’s strong wall(s) + A’s multi-heading coverage → textured ≥5..7 possible |

### NMS parameters (suggested)

| Param | Current | Hybrid tweak |
|-------|---------|--------------|
| \|n·n\| | ≥0.85 | keep |
| \|Δd\| | <2.5 m | keep |
| XY center | <6.0 m | **4.0 m** when either hyp has `split_parent` or sources differ and ZNCC gap >0.05 (keep higher-ZNCC) |

Do **not** change `_is_strictly_better` floors (min textured 3 / mean 0.35) — product safety stays.

### Source tag

`mapanything_hybrid` when both families present in kept set; else `mapanything_planarize` / `photo_consistency`.

---

## 6) First PR scope (smallest) vs second

### PR-1 ★ — splits + hybrid seeds + union promote (ship first)

**In**
1. Tighten split defaults: trigger **8**, window **8–10**, overlap **2–3**; optional `raw_width_m` pre-clamp split.  
2. `hybrid_heading=True` (default): inject `hypothesize_vertical_planes` into Path α scoring (same `score_planar_hyps` + NMS).  
3. Promote on **union** result via existing `_is_strictly_better` (source `mapanything_hybrid`).  
4. Telemetry: pre-NMS accepts, ma_kept/a_kept, windows=, candidate reason.  
5. CLI: `--hybrid-heading/--no-hybrid-heading`, `--split-trigger`, `--split-window`, `--split-overlap`.  
6. Tests (§7).

**Out**
- Full peel param sweep / exposing all peel knobs (PR-2).  
- Changing zncc_accept, quality-keep floors, OSM/BAG, Poisson.

### PR-2 — peel param sweep ★ (ship after #21)

**PC #21:** hybrid kept **5** @ mean 0.399; promote NO vs product **7/5/0.42**. Quality-keep correct — grow count via peel grid, do not weaken promote.

**In**
- CLI: `--min-inliers`, `--residual-stop`, `--vertical-dot`, `--peel-max-planes` (defaults **unchanged**; knobs for PC grid).  
- NMS XY **4 m** for `split_parent` siblings (`--nms-xy` 6 / `--nms-xy-split` 4); log `pre_nms→kept`.  
- Smoke grid A (§2) first cell **A1**, then **B1**; pick new defaults only if PC shows ≥+2 textured without slab spam.  
- Docs: compare §14 + this note. `detect_planar_patches` / DBSCAN remain parked.

**First PC recipe (A1):**
```
ps1-hood facades smoke-dense --planarize --source mapanything --zncc-accept 0.35 \
  --voxel 0.06 --plane-dist 0.06 --min-inliers 300 --residual-stop 1000 --peel-max-planes 32
```

---

## 7) Acceptance tests

### Unit / synthetic

1. **`test_split_long_hyp_8m_windows`** — 30 m peel, trigger=8, window=8, overlap=2 → ≥3 pieces; same `n,d`; centers spaced ~6–8 m.  
2. **`test_split_no_op_below_trigger`** — width 7 m → single hyp.  
3. **`test_hybrid_seeds_scored`** — mock frames + one MA hyp + A heading hyps → `score_planar_hyps` sees both sources; NMS can keep one of each if centers far.  
4. **`test_union_promote_beats_product`** — prev product textured=5 mean=0.42; candidate union textured=6 → `_is_strictly_better` True.  
5. **`test_union_no_promote_weaker`** — candidate textured=1 @ 0.44 → False; mirrors PC post-#20.  
6. **`test_extract_facades_hybrid_default`** — Path α branch with `hybrid_heading=True` calls hypothesize (mock) even when MA would keep ≥1 (spy/mock).

### PC smoke (after PR-1)

```bash
# Restore 7/5/0.42 bak first
ps1-hood facades <smoke> --planarize --zncc-accept 0.35 --source mapanything
# Expect log: hybrid ma=… a=… kept≥… ; if textured≥5 or better mean → promote
# Else candidate.* + product unchanged
```

**Pass criteria**
- [ ] ≥1 MA accept still possible (no regression of PR #20).  
- [ ] Hybrid run scores A seeds in same pass (log `a_kept` or source mix).  
- [ ] If union textured >5 or (textured≥5 and mean_zncc>0.44): product promoted.  
- [ ] If still weaker: product **unchanged**; `planes.candidate.json` written.  
- [ ] No OSM/BAG; quality-keep floors unchanged.

---

## 8) File / symbol cheatsheet

| Symbol | File | Change in PR-1 |
|--------|------|----------------|
| `DEFAULT_SPLIT_*` | `planarize.py` | trigger 8; window 8–10; overlap 2–3 |
| `split_long_hyp` / `expand_hyps_for_scoring` | `planarize.py` | defaults + optional raw width |
| `score_planar_hyps` | `planarize.py` | accept/merge A seed hyps; log ma/a |
| `hypothesize_vertical_planes` | `photo_planes.py` | call from α path (no change required) |
| `extract_facades` | `facades.py` | `hybrid_heading=True`; stop relying on 0-accept-only fallback |
| `_is_strictly_better` | `facades.py` | **unchanged** — feed it union metrics |
| `facades_cmd` | `cli.py` | hybrid + split CLI flags |

---

## 9) Stippy / Open3D notes

Open3D scrape notes (DBSCAN on inliers, `detect_planar_patches` min_plane_edge 8–12 m, MVS dist 0.03–0.10) are **parked** — see Addendum. Do **not** implement in PR-1. Iterative `segment_plane` + denser split + hybrid seeds is the PR-1 scope. Peel param knobs → PR-2 grid.

---

## 10) Pointer

Compare-and-pathforward **§12** (workspace): Path α more accepts / hybrid after PR #20.

*End. Chief: MessageSubagent into PR-1 (splits + hybrid + union promote).*

---

## Addendum — Stippy Open3D scrape (2026-09-13)

**Parked — do NOT implement in PR-1.** Optional for PR-2 / α2 only; does not change PR-1 top-3 (denser splits + hybrid A seeds + union promote).

### Peel / giant-wall hygiene
- After each `segment_plane`, optional **DBSCAN on inliers** to drop disconnected coplanar scraps before quad/split.
- MVS distance_threshold band **~0.03–0.10 m** (TLS lit often 0.02–0.05); tune to residual. Stop peel when |inliers| < ~0.1–1% of cloud or floor ~1k–5k @ 500k.
- Crop AABB/street-block before peel when block is huge.

### Splits
- Tangent `t = n × up`; project `s = (p−p0)·t`; bin 8–12 m; soft overlap **0.5–1 m** also OK (pack uses 2–3 m — either fine). Cut at density gaps / DBSCAN on `s` as alt to fixed windows.

### `detect_planar_patches` (α2 candidate)
- Better one-pass multi-façade; needs normals.
- Set **`min_plane_edge_length` ~8–12** (façade-scale; default auto is tiny).
- Tighten: ↓ `normal_variance_threshold_deg`, ↑ `coplanarity_deg`; raise `min_num_points` for dense MVS.
- Keep iterative `segment_plane` for α1 (hard Manhattan/extent control).

### Links worth citing
- Open3D PC tutorial / PointCloud API
- Boulaassal 2007 sequential RANSAC TLS façades (20–40 mm)
- Araújo & Oliveira 2020 → O3D `detect_planar_patches`
- PolyFit / City3D / CGAL Shape Detection (cluster_epsilon analog)
- Multiple_Planes_Detection (yuecideng)

