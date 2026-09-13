# Path α — Dual-source A(flow) + MA(peels) union (post–PR #25)

**Audience:** Chief / PC implementers (MessageSubagent → PR)  
**Date:** 2026-09-13 (Europe/Amsterdam)  
**Trigger:** PR #25 dual-arm PC: **A_arm kept=1 (MA ply)**; MA_arm 9→3; **a_kept=1 ma_added=3 union 4/4/0.454**; promote NO (4 textured < product 5). A-only `--no-planarize` on **MA ply**: **2/2/0.374**. Product held **7/5/0.42**.  
**Sacred:** photo-derived geometry; no OSM/BAG invent. Quality-keep (PR #24 clauses) stays. Do **not** dig peels / Open3D / α2 (parked).

**Code reviewed (`gh api` main @ d6dad33 — no clone dig):**
- `src/ps1_hood/reconstruct/facades.py` — dual-arm + single `ply_path` → both arms
- `src/ps1_hood/cli.py` — `facades --source mapanything|recon|auto`
- `src/ps1_hood/pipeline.py` — `_facade_pass` prefers MA PLY over flow `cloud.ply`
- `src/ps1_hood/reconstruct/planarize.py` — `resolve_dense_ply`
- `src/ps1_hood/reconstruct/photo_planes.py` — `search_photo_consistent_planes` / hypothesize
- `src/ps1_hood/reconstruct/unproject.py` — `triangulate_frames` → `recon/cloud.ply` (source=`flow`)

**Prior packs:** `path-alpha-a-reclaim-rd.md` (PR #25 shipped Fix A dual-arm; **wrong xyz** left), `path-alpha-more-accepts-rd.md`, `ps1hood-compare-and-pathforward.md` §17.  
**Compare §:** append §18 (this pack).

**Stippy PLY map (fold-in):**
| Arm | PLY | How |
|-----|-----|-----|
| **A** | `recon/cloud.ply` | flow triangulate (`--source recon` / pipeline default writer) |
| **MA** | `mapanything/cloud.ply` → `recon/cloud_mapanything.ply` | densify product |
| Dual CLI | **must not** force both arms onto MA ply | separate `--a-source` / `--ma-source` |

---

## 0) Executive call (for Chief)

| Question | Answer |
|----------|--------|
| Did PR #25 dual-arm work? | **Yes** as code: A=`search_photo_consistent_planes`, MA peels only, `a_priority` union. Telemetry matches design. |
| Why still a_kept≈1 ≪ historic ~7? | Both arms fed **MA dense PLY**. A on MA ply ceiling is **~1–2** (PC control). Historic **7/5/0.42** is flow-era A on `recon/cloud.ply`, not MA-ply A. |
| Top fix? | **Split PLY sources:** A xyz = flow/`recon/cloud.ply` (or locked product `planes.json` seeds); MA peels = MapAnything PLY; union `a_priority`. |
| Side bug? | A-only control overwrote `facades.candidate.*` after dual-arm — write **`facades.control.*`** for controls; never clobber hybrid candidate. |
| Acceptance | Smoke: **a_kept≈7** from flow/product; **ma_added≥0** non-dup; promote **only** if beats **7/5/0.42** under PR #24 clauses. |

**One-liner:** *Dual-arm is correct; dual-**source** is missing — stop forcing A onto MA ply; A=`recon/cloud.ply` (flow), MA=`mapanything/cloud.ply`, then `a_priority`.*

---

## 1) PC facts (post–#25)

| Run | Signal | Result | Notes |
|-----|--------|--------|-------|
| Dual-arm hybrid | A on **MA ply** + MA peels | a_kept=**1**, ma_added=**3**, union **4/4/0.454** | promote NO (textured 4 < product 5) |
| A-only `--no-planarize` | same **MA ply** | **2/2/0.374** | promote NO — MA-ply A ceiling confirmed |
| Live product | flow-era Milestone A | **7/5/0.42** | held by quality-keep |
| Side | A-only control after dual-arm | overwrote `facades.candidate.*` | hygiene bug |

**Confirms reclaim pack hyp 3/6:** historic product **≠** MA-ply A ceiling. PR #25 closed the *search-path* gap; the *support-cloud* gap remains.

---

## 2) Why A on MA ply collapses

### What A actually uses from `xyz`

`search_photo_consistent_planes` → `hypothesize_vertical_planes(frames, xyz)`:

| Seed family | Depends on xyz? | Role |
|-------------|-----------------|------|
| `heading_distance` / `manhattan` | **Indirect** via `ground_z` → wall center `u` | Primary inventory (historically ~150+ hyps) |
| `sparse` | **Direct** — local XY PCA on points above ground | ≤16 seeds; neighborhood structure matters |

In `extract_facades` today (PR #25 dual-arm):

```text
ply_path = resolve_dense_ply(..., source="mapanything")  # ONE PLY
xyz_raw  = read(ply_path)                                 # MA ~580k
xyz      = voxel_downsample(xyz_raw, voxel_m=0.08)        # dense MA support
ground_z = _fit_ground_z(xyz)
accepted_a = search_photo_consistent_planes(frames, xyz)  # ← A sees MA, not flow
accepted_ma = score_planar_hyps(MA peels from same ply)
```

### Ranked collapse causes

| Rank | Cause | Likelihood | Evidence |
|------|--------|------------|----------|
| **1** | **Wrong support cloud for A** — MA dense @0.08 vs flow @0.20 street-biased | **Confirmed (PC)** | A-only on MA ply → 2; dual A_arm=1; product 7 from flow-era |
| **2** | **`ground_z` from MA dense** shifts wall `u` → warp bands miss façades | **High** | Heading seeds share centers' Z from `_fit_ground_z(MA)`; flow ground band differs |
| **3** | **Sparse PCA on MA** yields different/noisy tangents vs flow | **Medium** | Sparse ≤16; not whole gap but pollutes hyp set / NMS neighbors |
| **4** | **Pipeline `_facade_pass` always swaps to MA** when present | **Confirmed (code)** | Ignores triangulated `cloud_ply` once `mapanything/cloud.ply` exists |
| **5** | Code path still gated hybrid scorer | **Ruled out** | #25 dual-arm uses full `search_photo_consistent_planes` |

**Density note:** More points ≠ better A seeds. Flow ~99k (product PC) / local smoke `recon/cloud.ply` (source=`flow`) is the cloud A was tuned against; MA ENU is the right cloud for **peels**, wrong as A's sole sparse/ground support.

**Not inventing flow-RANSAC walls:** A still scores heading×distance with ZNCC under known poses — flow PLY is **seed support / ground only**, same as historic Milestone A. Product geometry remains photo-consistent accepts.

---

## 3) Concrete dual-source recipe

### ★ Fix — Split xyz / peel PLY (prefer)

```python
# Resolve independently (Stippy map)
ply_a  = resolve_a_ply(root, a_source)   # flow|product|mapanything
ply_ma = resolve_ma_ply(root, ma_source) # mapanything (default)

xyz_a_raw = _read_ply_xyz(ply_a) if ply_a else empty
# A voxel = historic 0.20 (not MA 0.08)
xyz_a = _voxel_downsample_xyz(xyz_a_raw, 0.20) if len(xyz_a_raw) >= 20 else xyz_a_raw
ground_z_a = _fit_ground_z(xyz_a) if len(xyz_a) >= 20 else ground_from_cams(frames)

# --- A arm ---
if a_source == "product":
    accepted_a = load_locked_product_planes(root / "recon" / "planes.json")
    # treat as already-accepted A family (source tag photo_consistency / heading_*)
    # optional: re-score ZNCC once for fresh metrics; do NOT drop below product set
else:
    accepted_a = search_photo_consistent_planes(
        frames,
        xyz_a if len(xyz_a) >= 30 else None,
        zncc_accept=min(float(zncc_accept), 0.35),
        ground_z=ground_z_a,
        max_keep=n_planes,
    )

# --- MA arm ---
hyps, ground, residual_pts = planes_from_mapanything_ply(ply_ma, cam_c, ...)
accepted_ma = score_planar_hyps(
    hyps, frames, seed_hyps=None, union_strategy="nms", ...
)

# --- Union ---
accepted = union_keep_planes(
    list(accepted_a) + list(accepted_ma),
    strategy="a_priority",
    max_keep=n_planes,
    ...
)
# telemetry: a_ply=... ma_ply=... a_kept=... ma_added=...
```

### Alt — Locked product seeds (when flow PLY missing / thin)

If `recon/cloud.ply` absent or ≪30 pts above ground:

1. Read live product `recon/planes.json` (the held **7/5/0.42**).
2. Convert entries → accepted-plane dicts (`n`, `d`, `center`/`quad`, `width_m`, `height_m`, `zncc`, `source=product_lock`).
3. Use as `accepted_a` (locked); still run MA peels + `a_priority` add non-dup.
4. Promote still gated by PR #24 — locking seeds does **not** auto-promote weaker unions.

### Path resolution (Stippy)

```text
resolve_a_ply(a_source):
  flow | recon → recon/cloud.ply
                 (optional alias: recon/cloud_flow.ply if present & cloud.ply missing)
  product      → no PLY; planes.json seeds
  mapanything  → mapanything/cloud.ply | recon/cloud_mapanything.ply  (escape hatch only)

resolve_ma_ply(ma_source):
  mapanything  → mapanything/cloud.ply → recon/cloud_mapanything.ply
  (required for Path α peels)
```

**Do not** let `--source mapanything` alone set A's xyz (today's bug). Keep `--source` as deprecated alias for `--ma-source` during transition, or map old `--source` → ma only and default `--a-source flow`.

---

## 4) CLI / wiring sketch

### New flags (`cli.py` `facades`)

```bash
ps1-hood facades <run> \
  --a-source flow|product|mapanything \  # default: flow
  --ma-source mapanything \              # default: mapanything
  --planarize \                          # MA peels on
  --union-strategy a_priority \
  --zncc-accept 0.35 \
  --max-planes 16

# Control (measurement only — must not clobber candidate):
ps1-hood facades <run> --no-planarize --a-source flow --control-out
# or auto: when --no-planarize / hybrid off → write facades.control.*
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--a-source` | `flow` | A xyz / seeds: `recon/cloud.ply` \| product `planes.json` \| MA (debug) |
| `--ma-source` | `mapanything` | Peel PLY only |
| `--source` | *(compat)* | Prefer deprecate → sets `--ma-source`; warn if used alone |
| `--hybrid-a-full-search` | on | keep #25 behavior |
| `--control-out` / auto | | stage to `*.control` not `*.candidate` |

### `extract_facades` signature delta

```python
def extract_facades(
    ply_path: Path | None,          # MA peel PLY (or None)
    dest_obj: Path,
    *,
    a_ply_path: Path | None = None, # NEW — flow/recon for A xyz
    a_source: str = "flow",         # flow|product|mapanything
    output_kind: str = "auto",      # auto|product|candidate|control
    ...
)
```

When `a_ply_path` is None and `a_source=="flow"`: resolve `dest_obj.parent / "cloud.ply"` (`recon/cloud.ply`).

### `pipeline.py` `_facade_pass`

Today: swaps seed to MA whenever MA exists → **both arms on MA**.

```python
def _facade_pass(cloud_ply: Path, meta: dict) -> None:
    ma_ply = first_existing(mapanything/cloud.ply, recon/cloud_mapanything.ply)
    extract_facades(
        ply_path=ma_ply or cloud_ply,   # MA peels
        a_ply_path=cloud_ply,           # flow triangulate recon/cloud.ply
        a_source="flow",
        ...
    )
```

---

## 5) Candidate snapshot hygiene

### Bug

Dual-arm (weaker than product) correctly wrote `facades.candidate.*` / `planes.candidate.json`.  
Subsequent A-only `--no-planarize` control **reused the same candidate paths** → wiped the hybrid candidate artifact Chief needs for Studio compare.

### Fix

| Run kind | Dest when not promoting | Rule |
|----------|-------------------------|------|
| Hybrid / Path α product attempt | `facades.candidate.*`, `planes.candidate.json`, `textures/candidate/` | unchanged |
| Control (`--no-planarize`, A-only, ablations) | **`facades.control.*`**, `planes.control.json`, `textures/control/` | **never** touch `*.candidate` or live product |
| Optional hybrid debug dump | `planes.hybrid.json` (telemetry-only) | write alongside; **do not** let control overwrite |

```python
kind = output_kind
if kind == "auto":
    if not use_planarize or not hybrid_heading:
        kind = "control"
    elif prev_product_nonempty:
        kind = "candidate"
    else:
        kind = "product"

suffix = {"control": ".control", "candidate": ".candidate", "product": ""}[kind]
# out_obj = facades{suffix}.obj, planes{suffix}.json, textures/{control|candidate}/
```

Live product (`facades.obj` / `planes.json` / `textures/facade_*.jpg`) only changes on PR #24 promote success.

**Tests:** `test_control_run_does_not_clobber_candidate`; `test_candidate_survives_no_planarize_rerun`.

---

## 6) First PR scope + acceptance

### Scope (smallest ship)

1. **Dual-source PLY:** `a_ply_path` / `--a-source flow` default; MA peels stay on `--ma-source mapanything`.
2. Wire `pipeline._facade_pass` + `cli.facades` as above.
3. **Control output paths** (`*.control`) so controls never overwrite `*.candidate`.
4. Telemetry: `a_ply=… ma_ply=… a_arm_kept=… ma_added=… union=…`.
5. Optional follow-up in same PR if cheap: `--a-source product` locked reinject.
6. **Out of scope:** peel param grid, zncc_accept drop, quality-keep weakening, OSM/BAG.

### Acceptance (PC smoke)

- [ ] Dual-source hybrid log: `a_ply=…/recon/cloud.ply` (or product lock), `ma_ply=…/mapanything/cloud.ply`.
- [ ] **a_kept ≈ 7** (or ≥ honest `--no-planarize --a-source flow` control on same frames).
- [ ] **ma_added ≥ 0** non-dup via `a_priority` (MA may add 0–N).
- [ ] Promote **only** if union beats product **7/5/0.42** under PR #24:
  - clause1: textured↑ ∧ planes≥ ∧ mean≥ prior−0.02
  - clause2: textured= ∧ planes↑ ∧ mean≥ prior−0.02
  - clause3: textured= ∧ planes= ∧ mean≥ prior+0.02
- [ ] If weaker: product **unchanged**; hybrid → `*.candidate`; control → `*.control` (candidate intact).
- [ ] Unit: `test_dual_source_a_uses_recon_cloud_not_ma`; `test_control_does_not_overwrite_candidate`; existing dual-arm + a_priority tests still green.

### Smoke commands

```bash
# Product attempt (dual-source)
ps1-hood facades smoke --a-source flow --ma-source mapanything \
  --planarize --union-strategy a_priority --zncc-accept 0.35

# A ceiling control (must write *.control, leave candidate)
ps1-hood facades smoke --no-planarize --a-source flow --control-out

# Debug: prove MA-ply A still collapses (optional)
ps1-hood facades smoke --no-planarize --a-source mapanything --control-out
```

---

## 7) Code map (where to touch)

| Change | File |
|--------|------|
| Dual-source resolve + A xyz vs MA peel split; `output_kind` staging | `reconstruct/facades.py` |
| `--a-source` / `--ma-source` / `--control-out`; deprecate solo `--source` | `cli.py` |
| `_facade_pass`: pass flow `cloud_ply` as `a_ply_path`, MA as peel | `pipeline.py` |
| `resolve_a_ply` / split `resolve_dense_ply` docs | `reconstruct/planarize.py` |
| Optional `planes_from_product_json` | `photo_planes.py` or `facades.py` |
| Tests | `tests/test_planarize.py`, `tests/test_photo_planes.py` |

### Current bug locus (for reviewers)

```text
facades.py dual_arm:
  accepted_a = search_photo_consistent_planes(frames, xyz)  # xyz from MA ply_path
pipeline._facade_pass:
  seed = MA if exists else cloud_ply   # single seed → both arms
cli --source mapanything:
  resolve_dense_ply → one PLY for extract_facades
```

---

## 8) Hypotheses checklist

| # | Hypothesis | Verdict |
|---|------------|---------|
| 1 | Dual-arm code path wrong (still gated scorer) | **Ruled out** (#25) |
| 2 | A ceiling on MA ply is ~1–2 | **Confirmed PC** |
| 3 | Historic 7 needs flow/`recon/cloud.ply` support | **Confirmed / Stippy** |
| 4 | NMS / zncc / max_keep cut 7→1 | **Ruled out** (same as reclaim) |
| 5 | Control overwrite of candidate is a real hygiene bug | **Confirmed PC** |
| 6 | Product lock seeds are valid fallback when flow thin | **Recommended alt** |

---

*End. Top fix → Chief: ship dual-source PR — A=`recon/cloud.ply` (flow), MA=mapanything peels, `a_priority`; control → `*.control`.*
