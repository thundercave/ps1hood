# Worst-cam targeted gap-fill

**Date:** 2026-09-20 · Product **18/18** textured · `facade_zncc_mean≈0.138`  
**Goal:** seed walls in front of the worst finite compare cams — not global peel spam  
**Sacred:** sat XY · no free-pose · quality-keep · don’t wipe 18/18  
**Out:** densify · free-pose · roof floaters

Pack: `/workspace/worst-cam-gap-fill-rd.md` · base [`facade-gap-fill.md`](facade-gap-fill.md).

---

## What it does

1. Parse cam id `pano_hNNN` → pano + heading (`_h000` → 0°, `_h120` → 120°).
2. For each worst cam C: distance rings **8 / 12 / 16 / 20 m** × yaw **0 / ±45 / ±90** → vertical plane facing `heading+yaw`; tag `source=worst_cam`.
3. Existing manhattan + sat-corner seeds are **filtered** to hyps with frontal ≥ 0.25 in those cams.
4. Road-center reject; cap new hyps ~40; score with Path α ZNCC (`≥0.35`); prefer worst-cam frames in the view picker.
5. **`a_priority`:** lock product planes first; NMS-add new only (XY 6 m, n·n≥0.85, |Δd|<2.5); `max_keep` **24**.
6. Bake textures for **new indices only** (copy locked product JPGs into candidate).
7. Stage `facades.candidate.*`; quality-keep vs 18/18 (planes↑ or textured≥18 & mean not ↓>0.02). On fail, **never** delete `textures/facade_00..17.jpg`.

---

## Usage

```bash
# explicit worst cams (from compare / pack)
uv run ps1hood facades smoke-dense --gap-fill \
  --worst-cams 1-hH0xS8V_656nPgqQnl6A_h000,Vd7vlCY1OAUkLlszrnLbDg_h120,1g2FRL3bCGcwgPO48E70-Q_h180 \
  --a-source product --source mapanything --max-planes 24

# or auto-read recon/compare/summary.json top-N
uv run ps1hood facades smoke-dense --gap-fill --worst-from-compare 3 \
  --a-source product --source mapanything --max-planes 24
```

---

## PC after merge (smoke-dense)

```bash
cd ~/ps1hood && git pull
uv run ps1hood facades smoke-dense --gap-fill \
  --worst-cams 1-hH0xS8V_656nPgqQnl6A_h000,Vd7vlCY1OAUkLlszrnLbDg_h120,1g2FRL3bCGcwgPO48E70-Q_h180 \
  --a-source product --source mapanything --max-planes 24
```

Studio hard-reload; check overlays for those 3 cams (ZNCC finite ↑). Product stays ≥18 textured unless candidate is strictly better.

Log greps:

```bash
rg -n "worst_cam|gap_fill|candidate NOT promoted|prefer_frames|union_kept" \
  runs/smoke-dense/logs/*.log 2>/dev/null || true
```

---


See also [`gap-fill-stricter-multiview.md`](gap-fill-stricter-multiview.md) (gap-add multi-view + sat AABB + peel/add caps).

## Non-goals

Densify · free-pose · roof Z / floater fix · global `peel_max` inflate

---

## One-liner

*Seed planes in front of worst cams (heading±yaw × distance); ZNCC-gate; a_priority add to locked 18; candidate + quality-keep.*
