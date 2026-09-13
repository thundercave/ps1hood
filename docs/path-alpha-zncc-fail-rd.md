# Path α ZNCC FAIL — diagnosis + fix recipe (PR #15)

**Audience:** Chief / PC implementers (MessageSubagent → fix PR, no more research)  
**Date:** 2026-09-13 (Europe/Amsterdam)  
**Symptom:** Path α PR #15 merged → planarize+ZNCC on MA ENU → **23 vertical hyps, 0 passed ZNCC≥0.40**; best rejected ZNCC ≈ **−0.077** (uncorrelated, not anti-correlated); empty `planes.json`; `facades.obj` wiped to ~253 B (ground-only) — **lost prior 7 flow ZNCC~0.42 planes**.

**Sacred:** photo-derived geometry; no OSM/BAG / flow-RANSAC invention to fill gates. Preserve product on fail.

**Sources reviewed (main @ `4546c03` / PR #15):**
- `src/ps1_hood/reconstruct/planarize.py`
- `src/ps1_hood/reconstruct/facades.py` (`extract_facades`)
- `src/ps1_hood/reconstruct/photo_planes.py` (Milestone A — worked)
- `cli.py` `facades` cmd, `pipeline._facade_pass`

---

## 1) Ranked root causes (code-backed)

| Rank | Cause | Likelihood | Evidence | Why ZNCC≈−0.077 fits |
|------|--------|------------|----------|----------------------|
| **1** | **No ±n depth refine on MA hyps** | **Very high** | A: `search_photo_consistent_planes` tries `delta ∈ {−2,−1,+1,+2}` m along `n` before accept (`photo_planes.py` ~520–557). Path α: `score_planar_hyps` scores each hyp **once as-is** (`planarize.py` ~467–505). Recipe §3 said “score exactly like search_photo_consistent_planes loop” — **refine never wired**. Refine gated to `heading_distance`/`manhattan` only — even if shared, `ma_segment` would skip. | Homography with plane depth off by ~0.5–2 m warps to wrong façade / street strip → patches uncorrelated → ZNCC~0. A survived because heading seeds + 1D search found good `d`. |
| **2** | **AABB / street-slab peel** | **High** | `inliers_to_quad` uses full inlier **min/max** (`planarize.py` 126–127). RANSAC peels **largest** vertical first (`segment_plane` / numpy). On MA street clouds the biggest vertical is often long street-parallel façade stretch, curb, parked-car wall, or multi-building slab — not a single photo façade. Scoring clamps `w,h ≤ 12` (`planarize.py` 494–495) but still uses that slab’s `n,d,center` — wrong plane eq → wrong H. No max-extent reject; no percentile AABB. | Sampling a 12 m crop of a wrong-depth / curb plane → noise vs noise → ZNCC≈0. |
| **3** | **Overwrite wipe on 0 accepts** | **Confirmed bug (product loss)** | `extract_facades` **always** deletes `textures/*.jpg`, writes OBJ/MTL/planes.json even when `planes==[]` (`facades.py` 437–441, 500–531). Test `test_extract_facades_planarize_branch_writes_planes_json` encodes this. 253 B OBJ = ground quad only. | Did not cause ZNCC fail; **destroyed** prior good Milestone A product. |
| **4** | **No Milestone A fallback when planarize accepts=0** | **High (product)** | `extract_facades` Path α branch sets `accepted = score_planar_hyps(...)` and **never** calls `search_photo_consistent_planes` on empty keep (`facades.py` 385–417). Fallback only if PLY too thin / planarize off. | Same run that wiped product could have recovered ~7 heading×distance planes. |
| **5** | **View picker / grazing on bad normals** | **Medium** | `_pick_scoring_views(..., min_frontal=0.25)` (`photo_planes.py` 407–454). If hyp `n` is street-slab / sideways, “frontal” cams may still be grazing enough for garbage warps. Scoring **did** run (finite best≈−0.077 ⇒ ≥1 valid source warp), so picker not empty — but quality poor for wrong planes. | Grazing warp → scrambled patches → ZNCC~0. |
| **6** | **Quad rebuilt from center+w+h** | **Low–medium** | `score_vertical_plane` always `sample_plane_quad_world(n,d,center,w,h)` (`photo_planes.py` 207) — ignores hyp `corners`. For vertical AABB this matches unclamped corners; **clamp to 12 m** recenters a crop on slab AABB center (often mid-block / wrong façade). | Contributes when width≫12; not sole cause. |
| **7** | **Normal / cam facing** | **Low** | `_flip_n_toward` ≡ flip so signed distance of `cam_centroid` > 0 (`planarize.py` 89–95). Picker locally flips `n_xy` toward each cam for frontal test; scoring uses hyp `n`. Convention consistent with A. No evidence of systematic n flip bug. | Would tend toward anti-correlation (~−1) if images inverted, not −0.077. |
| **8** | **Homography / warp** | **Low** | Same `plane_homography` + ZNCC path that passed Milestone A. Breaks only if `n/d` wrong (causes 1–2). | — |
| **9** | **ENU up** | **Very low** | `UP=(0,0,1)`; MA ENU gate passed (ΔC 7.78 m, z p50 2.71 vs cam u 2.98). Vertical filter produced 23 hyps. | — |
| **10** | **Open3D missing → numpy RANSAC** | **Unknown / secondary** | Both backends share `inliers_to_quad` + same scoring gap. Numpy is weaker peel but 23 hyps means *something* vertical was found. Log `planarize[open3d|numpy]:` on PC to confirm. | Worse peels amplify #2; do not explain missing refine (#1). |

**Bottom line:** ZNCC≈−0.077 = **uncorrelated patches** → plane depth/identity wrong for SV, not a threshold-only fail. Primary code gap vs working A: **no depth refine on MA seeds** + **largest-plane AABB peel**. Product loss: **unconditional overwrite**.

---

## 2) Immediate PC debug checklist

Run on the smoke project that failed (do **not** overwrite product until overwrite-guard lands — copy recon aside first).

```bash
# 0) Backup current (already wiped) + restore prior good if available
cp -a recon/facades.obj recon/facades.obj.bak_empty || true
# restore from git/backup if you have the 7-plane OBJ

# 1) Confirm backend + hyp dump
ps1-hood facades <name> --planarize --zncc-accept 0.25 --source mapanything
# watch log: planarize[open3d|numpy]: N vertical hyps ...
```

### 2.1 Instrument `score_planar_hyps` (temporary)

For **every** hyp (or top-10 by inliers), log before score:

```text
hyp_i  inliers=…  n=(nx,ny,nz)  d=…  center=(e,n,u)
       w=… h=…  depth_to_nearest_cam=…  view_idxs=[…]
       frontal_scores=[…]  zncc=…  reason=…
```

Also log after optional refine deltas (once added): best delta and zncc per delta.

### 2.2 Save patches for worst / best hyp

In `score_vertical_plane` (debug flag), write:

```text
debug/zncc/hyp{XX}_ref.png
debug/zncc/hyp{XX}_src{j}_warped.png
debug/zncc/hyp{XX}_meta.json   # n,d,center,w,h,views,zncc,scores
```

**Read:** if ref looks like windows but warped is asphalt/sky → depth/`d` wrong (#1). If both look like curb/ground strip → slab peel (#2). If both façades but misaligned horizontally → `d` off by ~1–2 m.

### 2.3 Plane identity sanity

```python
# For each hyp: distance of cam_centroid to plane; XY extent
signed = n @ cam_c + d          # expect >0 after flip, typically 4–25 m
width_m, height_m               # flag if width_m > 25 or height_m > 15
# Inlier Z histogram: façade should span ~ground+1 … ground+8, not a thin band at curb height
```

### 2.4 A/B same run

```bash
ps1-hood facades <name> --no-planarize --zncc-accept 0.35   # expect ~flow-era planes back
ps1-hood facades <name> --planarize --zncc-accept 0.25      # smoke: any MA accepts?
```

### 2.5 Quick numeric gates to print

| Check | Healthy | Sick (this FAIL) |
|-------|---------|------------------|
| `#vertical hyps` | 8–24 | 23 (OK count, bad quality) |
| `best zncc` | ≥0.35 | ≈−0.077 |
| `median |width_m|` | 5–15 | often ≫20 if slabs |
| `median cam–plane depth` | 6–18 m | <3 or >30 or ~0 |
| `backend` | open3d preferred | log it |

---

## 3) Concrete code fixes (diff-level guidance)

### Fix A — Port depth refine into `score_planar_hyps` (do first)

**File:** `planarize.py` `score_planar_hyps`

Replace single `score_vertical_plane(...)` call with A-style candidate loop; **enable for `ma_segment`**:

```python
# inside for hyp in hyps: after views loaded
candidates = [(n, d, center)]
# Path α: always refine MA seeds (and any source)
for delta in (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0):
    c2 = center + n * delta
    d2 = float(-n @ c2)
    candidates.append((n, d2, c2))
# optional: also try -n with re-flipped d if best views prefer opposite
# for n2, d2 in ((-n, -d),):
#     candidates.append((n2, d2, center))

best_local = None
for n_c, d_c, c_c in candidates:
    result = score_vertical_plane(views, n_c, d_c, c_c, width_m=w, height_m=h, ...)
    ...
    if result.get("ok") and (best_local is None or result["zncc"] > best_local["zncc"]):
        best_local = {**result, "n": n_c, "d": d_c, "center": c_c, ...}
if best_local:
    accepted.append(best_local)
```

Also track `best_reject` across all deltas (today only as-is).

**Alt (cleaner):** factor shared `score_hyp_with_refine(hyp, views, *, deltas=..., zncc_accept=...)` used by both `search_photo_consistent_planes` and `score_planar_hyps`; set `refine=True` for `ma_segment`.

### Fix B — Percentile AABB + max extent + depth gate

**File:** `planarize.py` `inliers_to_quad`

```python
# was: s0, s1 = float(s.min()), float(s.max())
s0, s1 = np.percentile(s, [5, 95])
t0, t1 = np.percentile(t, [5, 95])
...
MAX_W, MAX_H = 25.0, 15.0
if width_m > MAX_W or height_m > MAX_H:
    # shrink to percentiles already; if still huge, clamp extent about center
    width_m = min(width_m, MAX_W)
    height_m = min(height_m, MAX_H)
    # rebuild corners from clamped w/h about AABB center (not full slab)
```

**File:** `score_planar_hyps` pre-score reject:

```python
cams = ...
depths = np.abs(cams @ n + d)          # |signed dist|
if float(depths.min()) < 4.0 or float(depths.min()) > 25.0:
    continue
# or: depth of center to nearest cam in 4–25 m
```

### Fix C — Prefer façade-like peels (optional α1.1)

Before accepting a vertical peel as hyp:

- Require inlier height span `t1-t0 ≥ 2.5 m` (already min_height 1.5 — raise to 2.5).
- Prefer planes with inlier density in upper band (reject curb-only: median z close to ground_z).
- Cap peel width: if `s1-s0 > 20`, **split** along `right` into overlapping 8–12 m windows (multiple hyps, same `n,d`) — gives ZNCC a chance on real façades.

### Fix D — Use hyp corners when present (minor)

**File:** `photo_planes.score_vertical_plane` — if caller passes `corners=` and w/h not clamped, use them; else `sample_plane_quad_world`. Or pass clamped center/w/h consistently and store scored corners back onto hyp (already returns `corners` in result).

### Fix E — CLI knobs

**File:** `cli.py` `facades_cmd` — add:

```text
--keep-previous-on-fail / --no-keep-previous-on-fail   (default: keep)
--fallback-heading / --no-fallback-heading             (default: fallback on)
--refine-deltas  (string "−2,-1,-0.5,0.5,1,2" or flag)
--zncc-accept already exists — try 0.25 smoke
--max-plane-extent-m 25
--aabb-percentile 5,95
```

Wire through `extract_facades(...)`.

---

## 4) Overwrite-guard patch sketch

**One-liner:** *If `accepted==0` and `--keep-previous-on-fail` (default true): do not unlink textures, do not overwrite `facades.obj` / `.mtl` / `planes.json`; write diagnostics to `facades.failed.*` instead.*

**File:** `facades.py` `extract_facades` — restructure write phase:

```python
def extract_facades(..., keep_previous_on_fail: bool = True, fallback_heading: bool = True):
    ...
    # --- after scoring ---
    if use_planarize and not accepted and fallback_heading:
        log.warning("Path α: 0 ZNCC accepts — falling back to Milestone A heading×distance")
        accepted = search_photo_consistent_planes(
            frames, xyz if len(xyz) >= 30 else None,
            zncc_accept=min(float(zncc_accept), 0.35),
            ground_z=ground_z, max_keep=n_planes,
        )
        source_tag = "photo_consistency_fallback"
        # if still empty, keep_previous applies below

    planes = [plane_dict_for_obj(pl, ground_z) for pl in accepted]

    dest_obj.parent.mkdir(parents=True, exist_ok=True)
    tex_dir = dest_obj.parent / "textures"
    planes_json = dest_obj.parent / "planes.json"
    mtl_path = dest_obj.with_suffix(".mtl")

    if not planes and keep_previous_on_fail:
        # FAIL-LOUD but preserve product
        failed_obj = dest_obj.with_suffix(".failed.obj")
        failed_json = dest_obj.parent / "planes.failed.json"
        write_planes_json(failed_json, planes=[], ground_z=ground_z,
                          residual_points=residual_pts, source=source_tag + "_failed")
        # optional: write ground-only to failed_obj for debug — NOT to dest_obj
        _write_obj(failed_obj, [], extent_xyz, [materials_ground_only], ...)
        log.error(
            "facades: 0 accepts (source=%s) — kept previous %s / textures / planes.json; "
            "diagnostics → %s",
            source_tag, dest_obj.name, failed_json.name,
        )
        return {
            "path": str(dest_obj),  # unchanged on disk
            "planes": 0,
            "preserved_previous": True,
            "source": source_tag,
            ...
        }

    # success path only:
    if tex_dir.is_dir():
        for old_tex in tex_dir.glob("facade_*.jpg"):  # do NOT delete ground.jpg early
            old_tex.unlink(missing_ok=True)
    ...
    # bake textures, write obj/mtl/planes.json as today
```

**Tests to update/add:**

1. `test_extract_facades_planarize_branch_writes_planes_json` — on 0 accepts with keep=True, **product files unchanged** if pre-seeded; `planes.failed.json` exists.  
2. New: pre-write a fake `facades.obj` + texture → run planarize fail → assert bytes of obj/texture unchanged.  
3. New: planarize 0 accepts + fallback finds A planes → `source` contains `fallback`, planes≥1.  
4. `keep_previous_on_fail=False` → current wipe behavior (debug only).

**CLI:** `--keep-previous-on-fail` default **True**; pipeline `_facade_pass` passes default.

---

## 5) Fallback: planarize 0 → Milestone A on same run

Already sketched in §4. Contract:

```text
Path α peel → score_planar_hyps
    │
    ├─ accepts ≥ 1 → write product (source=mapanything_planarize)
    │
    └─ accepts == 0
           → search_photo_consistent_planes (zncc_accept≤0.35, refine=True)
                 ├─ accepts ≥ 1 → write product (source=photo_consistency_fallback)
                 └─ accepts == 0 → keep-previous guard (§4); fail-loud log
```

**Do not** invent RANSAC/OSM walls. Heading×distance is the known-good photo path.

Pipeline should log which branch won so Studio/compare stays honest.

---

## 6) Retune knobs (try on PC in order)

| Knob | Current | Try | Why |
|------|---------|-----|-----|
| `zncc_accept` | 0.40 (MA) | **0.25 smoke**, then 0.35 | Isolate “zero because threshold” vs “uncorrelated”; expect 0.25 still near-zero if #1/#2 dominate |
| Depth refine deltas | *none* | **±0.5,±1,±2 m** | Primary fix |
| `distance_threshold` | 0.08 | 0.05 / 0.10 / 0.12 | Tighter = less thick slabs; looser = more inliers |
| `voxel` | 0.08 | 0.05 / 0.10 | Affects RANSAC scale |
| `min_inliers` | 400 | 250 / 600 | Lower → more small façades; higher → only big slabs |
| AABB | min/max | **percentile 5–95%** | Kill outlier inliers |
| Max plane extent | clamp score 12 m only | **reject/split if w>25 or h>15** | Stop street-slab hyps |
| Plane center depth | soft 40 m XY | **require 4–25 m** to nearest cam | Drop floaters / far ghosts |
| `min_height` | 1.5 | **2.5–3.0** | Reject curb ribbons |
| `min_frontal` (score) | 0.25 | 0.35 | Less grazing |
| `max_planes` peel | 24 | keep; split long walls instead | — |

**Expected after Fix A+B+fallback:** either MA accepts with ZNCC≳0.35, or same-run A restores ~heading planes; **never** empty wipe of prior product.

---

## 7) Suggested PR sequence (implementer)

1. **PR-fix-guard:** overwrite-guard + tests (unblocks safe iteration).  
2. **PR-fix-score:** depth refine for `ma_segment` + shared helper with A; zncc smoke.  
3. **PR-fix-peel:** percentile AABB, max extent / split, depth gate, min_height.  
4. **PR-fix-fallback:** auto Milestone A when planarize accepts=0; CLI flags.  

Can squash 2–4 if small; **guard must land first** before more PC planarize runs.

---

## 8) Acceptance criteria for “fixed”

- [ ] PC re-run Path α: `accepted ≥ 1` **or** fallback recovers heading planes; mean ZNCC ≥ 0.35 (soft warn <0.42).  
- [ ] Forced 0-accept path: prior `facades.obj` + `textures/facade_*.jpg` + `planes.json` **unchanged**; `planes.failed.json` written.  
- [ ] Debug dump optional behind flag; default logs hyp summary (n/d/w/h/inliers/best zncc/delta).  
- [ ] No OSM/BAG/flow-RANSAC walls as product.  
- [ ] Tests: guard + refine-improves-synthetic (optional) + fallback branch.

---

## 9) File / symbol cheatsheet

| Symbol | File | Role |
|--------|------|------|
| `inliers_to_quad` | `planarize.py:98` | min/max AABB → hyp |
| `score_planar_hyps` | `planarize.py:430` | MA score **without** refine |
| `search_photo_consistent_planes` | `photo_planes.py:457` | A path **with** refine |
| `score_vertical_plane` | `photo_planes.py:187` | ZNCC; rebuilds quad |
| `extract_facades` | `facades.py:318` | wipe+write always |
| `facades_cmd` | `cli.py:452` | `--zncc-accept`, no keep/fallback yet |

---

*End of pack. Pointer from path-forward §10.*

---

## Post–PR #16 addendum — sentinel −1 + quality-aware promote

See `/workspace/path-alpha-zncc-neg1-and-quality-keep.md` and **compare-and-pathforward §11**.

**Product rule:** promote new façades **only if strictly better** than existing
(textured count → plane count → mean_zncc + 0.02). Else write
`facades.candidate.*` / `planes.candidate.json` and keep live product.
Fallback with 2 planes @ 0.377 must **not** overwrite 7-plane / ZNCC 0.42.

**Score telemetry:** `best_reject = -1.0` is a **SENTINEL** until a finite ZNCC
is recorded. FAIL-LOUD must include `scored=` / `finite=` / skip histogram and
label `SENTINEL` when no finite score. Cam-depth gate uses **center→nearest-cam
Euclidean** (not min `|n·C+d|` alone), defaults 2–35 m.

