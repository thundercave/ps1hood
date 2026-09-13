# Path α post–PR #16 — ZNCC≈−1 sentinel vs real + quality-aware keep

**Audience:** Chief / PC implementers (MessageSubagent → fix PR)  
**Date:** 2026-09-13 (Europe/Amsterdam)  
**Trigger:** Path α PR #16 retry FAIL — still **0 ZNCC accepts**; reported best≈**−1**; Milestone A fallback wrote **2** planes ZNCC **0.377** and **clobbered** better **7**-plane product (only in `.bak`).

**Sacred:** photo-derived geometry; no OSM/BAG invent. **Never demote a better product.**

**Code reviewed (main = PR #16 merge, via `gh api` / `/tmp/planarize16.py` / `/tmp/facades16.py`):**
- `src/ps1_hood/reconstruct/planarize.py` → `score_planar_hyps`
- `src/ps1_hood/reconstruct/facades.py` → `extract_facades` keep/fallback
- `src/ps1_hood/reconstruct/photo_planes.py` → `score_vertical_plane`, `_pick_scoring_views`

---

## 0) Executive call (for Chief)

| Question | Answer |
|----------|--------|
| Is best≈−1 proof of anti-correlation? | **No — usually the sentinel.** |
| Likely this run? | **Sentinel** (`best_reject` never updated) — **confirm with `scored=N` in FAIL-LOUD line.** |
| Why vs PR #15 (−0.077)? | PR #15 scored ≥1 finite ZNCC; PR #16 added **depth gate 4–25 m** + peel clamps → many/all hyps can skip **before** any finite score. |
| Product bug? | **Confirmed:** `keep_previous_on_fail` only when `not planes`. Fallback with 2 planes **bypasses** guard → clobbers 7-plane product. |
| First PR? | **Quality-aware promote / keep-previous** (product safety). Then α score using `scored=` branch. |

**One-liner quality-keep:** *Promote new façades only if strictly better than existing product (more textured walls, or same count with higher mean ZNCC above mins); else write `facades.candidate.*` and keep product.*

---

## 1) Sentinel vs real −1 — decision tree

### Code fact (verified)

```python
# planarize.score_planar_hyps (post-PR16)
best_reject = -1.0   # SENTINEL — not a measured ZNCC
n_scored = 0
...
z = result.get("zncc")
if isinstance(z, float) and not math.isnan(z):
    best_reject = max(best_reject, z)
...
# FAIL-LOUD:
# "… (best rejected≈%.3f; scored=%s). …" % (best_reject, n_scored)
```

`score_vertical_plane` returns `zncc: float("nan")` when it never gets a valid pairwise score (need≥2 views, quad behind cam, too few source warps, etc.). Nan **does not** update `best_reject`.

Homography is invariant under `(n,d)→(-n,-d)` for the **same** views; opposite-n candidates alone do **not** flip ZNCC. (PR #16 still tries −n because view picker ran on +n; for same views opposite-n is redundant compute.)

### Decision tree

```text
PC log: planarize: FAIL-LOUD — 0 / H MA hyps passed ZNCC≥T
        (best rejected≈B; scored=S)
                    │
                    ▼
            ┌───────┴────────┐
            │  parse B and S │
            └───────┬────────┘
                    │
     ┌──────────────┼──────────────────────────────┐
     ▼              ▼                              ▼
 S == 0         S > 0 and B ≈ -1.000           S > 0 and B finite,
 (B stays -1)   (all scored returns nan)       B ≳ -0.2 … +0.4
     │              │                              │
     ▼              ▼                              ▼
 SENTINEL         SENTINEL-ish                  REAL scores
 all hyps         score_vertical_plane          (PR#15 style)
 skipped BEFORE   called but never got          If B ≈ -0.9…-1.0
 any score call   finite pairwise ZNCC          → real anti-corr
 → §3             → §3 / view+warp              → §2 warp/frame
```

| Branch | Meaning | Next |
|--------|---------|------|
| **`scored=0`** | Every hyp hit `continue` before `score_vertical_plane` (tilt / XY40 / **depth 4–25** / view picker / load_view) | §3 — gate telemetry |
| **`scored>0`, best≈−1.000** | Calls happened; every `zncc` was nan (no valid source warp / behind cam / empty scores) | Instrument reasons; still **not** anti-corr |
| **`scored>0`, best≈−0.07…+0.3** | Real uncorrelated / weak scores (PR #15) | Peel + refine (already in #16); dump patches |
| **`scored>0`, best≈−0.9…−1.0** | Real anti-correlation | §2 warp/frame |

**Likelihood this FAIL:** **sentinel (`scored=0` or all-nan)** — Chief said best≈**−1** (sentinel print), not −0.077. Fallback still found 2 A-planes @ 0.377 ⇒ camera/frames/ZNCC path itself is alive.

### PC log greps (run first — do not re-clobber product)

```bash
# From the failed facades / pipeline log (adjust path)
rg -n "FAIL-LOUD|scored=|best rejected|Path α|fallback|preserved_previous|ZNCC" \
  runs/<smoke>/logs/*.log  2>/dev/null \
  || journalctl --user -u …   # whatever PC uses

# Canonical FAIL-LOUD line (planarize.py):
# planarize: FAIL-LOUD — 0 / 23 MA hyps passed ZNCC≥0.40 (best rejected≈-1.000; scored=0)

# Also useful:
rg -n "planarize\[|vertical hyps|depth|Path α: .* segmented|photo_consistency_fallback|kept previous" \
  runs/<smoke>/logs/*.log
```

**Decision greps (copy):**

```bash
# Extract scored + best in one shot
rg -o "best rejected≈[-0-9.]+; scored=[0-9]+" runs/<smoke>/logs/*.log

# If scored=0 → sentinel; do NOT chase warp sign first
# If scored>0 and best≈-1.000 → all-nan path
# If scored>0 and -1 < best < 0.35 → real weak scores
```

**Restore product before more runs:**

```bash
# If .bak holds the 7-plane product:
cp -a recon/facades.obj.bak recon/facades.obj   # names as on PC
cp -a recon/planes.json.bak recon/planes.json 2>/dev/null || true
# restore textures from bak if needed
# Do NOT re-run facades until quality-aware promote lands (or --no-fallback-heading + keep)
```

---

## 2) If real −1 (scored>0 and finite z≈−1): warp/frame causes

Ranked only **after** log proves finite anti-corr (rare vs sentinel).

| Rank | Cause | Why ZNCC→−1 | Debug |
|------|-------|-------------|-------|
| **1** | **Wrong plane identity** (slab/curb/sky strip) with enough warp coverage | Ref patch vs source patch are complementary street content (e.g. bright window vs dark asphalt) → strong negative | Dump patches (§2.1) |
| **2** | **Plane through / past cameras** (`c` in H near 0 or sign flip zone) | Homography ill-conditioned / folds → scrambled inverted look | Log `c = n·(Rᵀt)−d` per view; reject `|c|<0.5` |
| **3** | **Ref vs src view mismatch** (picker frontal ok but different façades) | Same hyp center, two panos see different walls | Log view idxs + frontal + pano_id |
| **4** | **ENU / Rt convention bug** on one path only | Would also break Milestone A — A still scores 0.377 ⇒ **unlikely** as global | A/B same frames |
| **5** | **Opposite-n with different views** | Algebraically H invariant for same views; only matters if picker re-run | Do not blame −n alone |
| **6** | Image channel / gray invert | Would be systematic on A too | Unlikely |

### 2.1 Debug patch dumps (pasteable)

Temporary flag e.g. `PS1_ZNCC_DEBUG=1` or `debug_zncc_dir=…` inside `score_vertical_plane` / `score_planar_hyps`:

```python
# After computing tmpl / warped / s for each hyp candidate:
import json, cv2
from pathlib import Path
dbg = Path("recon/debug/zncc")  # or /tmp
dbg.mkdir(parents=True, exist_ok=True)
tag = f"hyp{hyp_i:02d}_d{delta:+.1f}"
cv2.imwrite(str(dbg / f"{tag}_ref.png"), tmpl)
cv2.imwrite(str(dbg / f"{tag}_src{j}_warped.png"), warped)
(dbg / f"{tag}_meta.json").write_text(json.dumps({
    "n": n_c.tolist(), "d": float(d_c), "center": c_c.tolist(),
    "w": w, "h": h, "zncc": s, "view_idxs": idxs,
    "nearest_depth": nearest_depth, "reason": result.get("reason"),
}, indent=2))
```

**Read:** windows vs asphalt → depth/identity; both façades misaligned → refine; photographic negative / fold → H/`c`.

---

## 3) If sentinel: why all hyps skipped after PR #16

### Skip chain in `score_planar_hyps` (order)

1. `|n·up| > 0.20` → continue  
2. XY dist to nearest cam `> 40 m` → continue  
3. **NEW PR #16:** `nearest_depth = min |cam·n + d|` ∉ **[4, 25] m** → continue  
4. `_pick_scoring_views(..., min_frontal=0.25)` `< 2` → continue  
5. `load_view` fails → `< 2` views → continue  
6. Else score (+ refine ±deltas + opposite-n); nan zncc does not bump `best_reject`

### Why scored→0 is plausible on MA street peels

| Gate | Default | Failure mode on MA peels |
|------|---------|--------------------------|
| **Cam depth 4–25 m** | `DEFAULT_MIN/MAX_CAM_DEPTH_M` | Street-parallel / curb / multi-building slabs often have **cam almost on plane** (`|n·C+d| < 4`) or far floaters `> 25`. Largest RANSAC verticals are often *along* the drive path → nearest cam depth tiny → **all skipped**. |
| Peel clamps | AABB 5–95, max w/h 25/15, min_h **2.5** | Fewer but “tighter” hyps; survivors may still fail depth. |
| View picker | frontal≥0.25, dist 2–35, cross-pano | After depth gate, remaining hyps may be grazing-only. |
| Refine | ±0.5..2 m along n | **Never runs** if hyp skipped pre-score — cannot rescue depth-gated hyps. |

PR #15 could still score wrong slabs → best≈**−0.077**. PR #16 depth gate **silences** those scores → best stays **−1.000**, `scored=0`. Same root peel quality; different log symptom.

### Fixes (α score path — after product safety)

1. **Log skip reasons** (must-have telemetry):

```python
skip = {"tilt": 0, "xy40": 0, "depth": 0, "views": 0, "load": 0}
...
skip["depth"] += 1
# end:
log.error("… scored=%s skips=%s …", n_scored, skip)
# per-hyp one-liner: depth=… frontal_n=… reason=depth_gate|…
```

2. **Depth gate soften / redefine:**
   - Soft: warn `<4` or `>25` but still score (or only reject `<1.5` / `>40`).
   - Better: depth of **plane center → nearest cam** (Euclidean) in 4–25, not `|n·C+d|` alone — cams beside a long façade can have small signed plane distance while still viewing the wall.
   - Or: require depth in band for **≥1 frontal cam**, not `min` over all cams (a cam *on* the slab line zeros the min).

3. **Split long walls** before gate (prior pack Fix C): overlapping 8–12 m windows so center depth is façade-like.

4. **Do not depth-gate before refine:** allow candidates with depth in 2–30 after ±n deltas; gate on **best** candidate.

5. Keep opposite-n optional; do not treat it as ZNCC polarity fix.

### Suggested one-line log upgrade

```text
planarize: FAIL-LOUD — 0/23 ZNCC≥0.40 (best≈-1.000 SENTINEL; scored=0;
  skips tilt=0 xy40=0 depth=21 views=2 load=0). …
```

Print `SENTINEL` when `n_scored==0` or no finite z — stops Chief/PC misreading −1 as anti-corr.

---

## 4) Quality-aware promote — algorithm + sketch

### Confirmed gap

```python
# facades.extract_facades (PR #16)
if not accepted and fallback_heading:
    accepted = search_photo_consistent_planes(...)  # may return 2 planes @ 0.377
...
if not planes and keep_previous_on_fail and _facade_product_looks_nonempty(...):
    # preserve — ONLY empty accepts
    ...
# else: DELETE facade_*.jpg, overwrite obj/mtl/planes.json  ← 2-plane fallback lands here
```

So: Path α fail → A fallback with **any** planes ≥1 → **always overwrites**, even if worse than existing 7-plane / higher-ZNCC product.

### Promote rule (strictly better)

Define product score:

```text
existing = read_product_metrics(dest_obj.parent)
  # prefer planes.json: n_planes, mean_zncc, textured count
  # else: count textures/facade_*.jpg; n_planes from OBJ usemtl facade_*; mean_zncc=None

candidate = { n_planes, mean_zncc, textured, source }

BETTER iff:
  (textured > existing.textured)
  OR (textured == existing.textured AND n_planes > existing.n_planes)
  OR (textured == existing.textured AND n_planes == existing.n_planes
      AND mean_zncc is not None AND existing.mean_zncc is not None
      AND mean_zncc > existing.mean_zncc + 0.02)   # epsilon

HARD FLOORS (optional): refuse promote if textured < 3 or mean_zncc < 0.35
  when existing already meets floors — never demote a good block.

Else: write facades.candidate.obj / .mtl / planes.candidate.json / textures_candidate/
      keep product; log reason.
```

When Path α fails and fallback runs: **same** promote check vs existing (and vs bak if `planes.json` already wiped — prefer bak metrics).

### Pasteable sketch for `extract_facades`

```python
def _read_product_metrics(recon_dir: Path) -> dict:
    import json
    recon_dir = Path(recon_dir)
    metrics = {"n_planes": 0, "mean_zncc": None, "textured": 0, "source": None}
    pj = recon_dir / "planes.json"
    if pj.is_file():
        try:
            payload = json.loads(pj.read_text(encoding="utf-8"))
            planes = payload.get("planes") or []
            metrics["n_planes"] = len(planes)
            metrics["source"] = payload.get("source")
            zs = [float(p["zncc"]) for p in planes if p.get("zncc") is not None]
            if zs:
                metrics["mean_zncc"] = sum(zs) / len(zs)
            gates = payload.get("gates") or {}
            if gates.get("mean_zncc") is not None:
                metrics["mean_zncc"] = float(gates["mean_zncc"])
        except Exception:
            pass
    tex = recon_dir / "textures"
    if tex.is_dir():
        metrics["textured"] = len(list(tex.glob("facade_*.jpg")))
    if metrics["n_planes"] == 0 and metrics["textured"]:
        metrics["n_planes"] = metrics["textured"]  # texture-only fallback
    return metrics


def _is_strictly_better(new: dict, old: dict, *, min_textured: int = 3,
                        min_mean_zncc: float = 0.35) -> tuple[bool, str]:
    if old["n_planes"] <= 0 and old["textured"] <= 0:
        return True, "no existing product"
    # refuse demotion below floors when old is already good
    old_good = old["textured"] >= min_textured or (
        old["mean_zncc"] is not None and old["mean_zncc"] >= min_mean_zncc
    )
    if old_good:
        if new["textured"] < old["textured"]:
            return False, f"fewer textured ({new['textured']}<{old['textured']})"
        if new["textured"] == old["textured"] and new["n_planes"] < old["n_planes"]:
            return False, f"fewer planes ({new['n_planes']}<{old['n_planes']})"
        if (
            new["textured"] == old["textured"]
            and new["n_planes"] == old["n_planes"]
            and old["mean_zncc"] is not None
            and (new["mean_zncc"] is None or new["mean_zncc"] <= old["mean_zncc"] + 0.02)
        ):
            return False, (
                f"not higher mean_zncc ({new['mean_zncc']} vs {old['mean_zncc']})"
            )
    if new["textured"] > old["textured"]:
        return True, "more textured walls"
    if new["n_planes"] > old["n_planes"]:
        return True, "more planes"
    if (
        new["mean_zncc"] is not None
        and old["mean_zncc"] is not None
        and new["mean_zncc"] > old["mean_zncc"] + 0.02
    ):
        return True, "higher mean_zncc"
    if not old_good and (new["n_planes"] > 0 or new["textured"] > 0):
        return True, "replace empty/weak product"
    return False, "candidate not strictly better"


# --- inside extract_facades, AFTER planes list built, BEFORE unlink/write: ---

old_m = _read_product_metrics(dest_obj.parent)
new_m = {
    "n_planes": len(planes),
    "mean_zncc": (
        float(np.mean([p.get("zncc") for p in accepted if p.get("zncc") is not None]))
        if any(p.get("zncc") is not None for p in accepted) else None
    ),
    "textured": None,  # filled after bake if needed; for pre-check use n_planes / zncc
    "source": source_tag,
}
# Pre-bake promote gate using plane count + zncc; textured≈len(planes) expected
new_m["textured"] = new_m["n_planes"]  # optimistic; re-check after bake optional

if keep_previous_on_fail and _facade_product_looks_nonempty(dest_obj):
    ok_promote, why = _is_strictly_better(new_m, old_m)
    if not ok_promote:
        # write candidate sidecars only
        cand_obj = dest_obj.with_name(dest_obj.stem + ".candidate.obj")
        cand_json = dest_obj.parent / "planes.candidate.json"
        # ... bake into textures_candidate/ OR skip bake and only write planes.candidate.json
        write_planes_json(cand_json, planes=planes_payload, ground_z=ground_z,
                          residual_points=residual_pts, source=source_tag + "_candidate",
                          textured_maps=[])
        log.error(
            "facades: candidate (%s planes, mean_zncc=%s, source=%s) NOT promoted — %s; "
            "kept product; wrote %s",
            new_m["n_planes"], new_m["mean_zncc"], source_tag, why, cand_json.name,
        )
        return {
            "path": str(dest_obj),
            "planes": old_m["n_planes"],
            "preserved_previous": True,
            "candidate_planes": new_m["n_planes"],
            "candidate_reason": why,
            "ok": False,
            "source": source_tag,
            ...
        }

# else: existing success write path
```

**Also:** when Path α fails, prefer **skip fallback overwrite** unless fallback beats product:

```python
if not accepted and fallback_heading:
    fb = search_photo_consistent_planes(...)
    if fb:
        # stash as candidate; only assign accepted=fb if _is_strictly_better(...)
        ...
```

**CLI:** keep `--keep-previous-on-fail` (default True); add `--promote-mode {always,strict,off}` default **strict**.

### Tests

1. Seed 7-plane product (planes.json + 7 jpgs) → run path that yields 2 planes @ 0.377 → product unchanged; `planes.candidate.json` exists.  
2. Seed empty → 2 planes → promote.  
3. Seed 2 @ 0.35 → candidate 2 @ 0.45 → promote.  
4. `keep_previous_on_fail=False` → old wipe behavior.

---

## 5) Recommended first PR scope

### PR-A — Product safety (ship first) ★

**In**
- Quality-aware promote (§4) in `extract_facades`
- Candidate sidecars: `facades.candidate.*` / `planes.candidate.json`
- When Path α 0 accepts: fallback only promotes if better; else candidate + keep
- FAIL-LOUD log distinguishes `preserved_previous` + reason
- Tests above
- Optional: auto-restore from `*.bak` if present and product empty but bak good (PC already restoring manually)

**Out**
- Depth-gate retune, peel splits, debug dumps (PR-B)

### PR-B — α score (after greps)

**In**
- Print `scored=` + **skip histogram** + `SENTINEL` label when no finite z
- Redefine / soften cam-depth gate (§3.2); per-hyp depth log
- Optional ZNCC debug patch dumps behind flag
- Only if greps show `scored>0` and real ≈−1: warp/`c` asserts

**PC order**
1. Restore 7-plane bak  
2. Land PR-A  
3. Grep `scored=` on next α run (read-only / `--no-fallback-heading` if needed)  
4. Branch §2 vs §3 → PR-B

---

## 6) Acceptance criteria

- [ ] PC: fallback 2 @ 0.377 **cannot** overwrite 7-plane product; candidate files written; log states reason.  
- [ ] FAIL-LOUD line always includes `scored=N`; if N=0 or no finite z, log says `SENTINEL` not “anti-correlated”.  
- [ ] After PR-B: either MA accepts ≥1 with mean ZNCC≥0.35, or transparent keep + candidate; never silent demotion.  
- [ ] No OSM/BAG / invent walls.

---

## 7) File / symbol cheatsheet

| Symbol | File | Note |
|--------|------|------|
| `best_reject = -1.0` | `planarize.py` `score_planar_hyps` | **Sentinel** |
| `n_scored` | same | Gate for §1 tree |
| `DEFAULT_MIN/MAX_CAM_DEPTH_M` | `planarize.py` 4 / 25 | New skip source in #16 |
| `keep_previous_on_fail` | `facades.py` | Only when `not planes` — gap |
| `fallback_heading` | `facades.py` | 2-plane clobber vector |
| `_facade_product_looks_nonempty` | `facades.py` | Existence only, not quality |
| `score_vertical_plane` | `photo_planes.py` | nan zncc on fail paths |

**Prior packs:** `/workspace/path-alpha-zncc-fail-rd.md`, `/workspace/path-alpha-planarize-recipe.md`

---

*End. Pointer: path-forward §11.*
