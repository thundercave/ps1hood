# Far-side panos for `gap_needs`

**Date:** 2026-09-20 · Product **18/18** locked · sat-edge peels **STOPPED**  
**Need:** `recon/gap_needs.json` → e.g. **`roof_099_e0`** `{e, n, heading, reason: no_facing_cam}`  
**Sacred:** sat absolute XY · **no free-pose** · lock 18 · photo search (not sat_edge)

Pack: `/workspace/far-side-gap-needs-rd.md`.

---

## Prefer existing facing (then fetch)

Before probing, **prefer existing nearly-facing frames** already in `align/cameras.json`
(same look-at / 6–35 m / same-side filters; **no graze gate** so near-misses like
`Ot6jYiTfjNNn-G8YqwyfCQ` heading 180 @ ~22 m for `roof_099_e0` count).

`fetch` still probes for **better** panos (more frontal / ideal 8–20 m rings). Existing
facing are never discarded.

---

## CLI

```bash
# List needs + ENU→ll + probes + existing facing
uv run ps1hood gap-needs show smoke-dense
uv run ps1hood gap-needs show smoke-dense --need-id roof_099_e0

# Probe outside along outward normal → attach SV → crop new only → align sat
uv run ps1hood gap-needs fetch smoke-dense --need-id roof_099_e0 \
  --radii 8,12,16,20 --max-panos 6
# aliases: --id  ·  --no-capture  ·  --no-align  ·  --no-film
```

Fail-loud if `recon/gap_needs.json` is missing.

### Internals

1. `LocalFrame` from scene/live origin or project bbox; `enu_to_ll(need.e, need.n)`.
2. Probes at **outward** `look_heading` × `{8,12,16,20}` m (outside the wall).
3. Lookup / provisional seeds; skip panos already in `discover/panos.json` or `raw/`.
4. Facing filter: `(need−cam)/‖·‖ · cam_fwd ≥ 0.5`, dist ∈ [6, 35] m, same-side.
5. Tag `source=gap_needs`, `need_id=…`; capture/crop **new** panos only.
6. `ps1hood align … --align-prior sat` — seat into same ENU; no free-pose / no BAG snap.
7. If still **&lt;2** facing panos → ENU `lerp_pose` FILM spur (PR-B); skip if ≥2.

---

## Then façades — photo search, not sat_edge

```bash
uv run ps1hood facades smoke-dense \
  --a-source product \
  --no-gap-fill \
  --no-planarize
# Milestone A / search_photo_consistent_planes sees new frames
# product_lock 18; quality-keep; do NOT --gap-fill --gap-seeds sat-edge
```

---

## Acceptance

| Result | Action |
|---|---|
| New facing panos + textured garage plane, promote OK | Done for this need |
| Panos fetched, A still 0 for that wall | Leave need open; Studio sculpt later — **no peel** |
| Zero coverage at probes | Widen radii / manual pano pick; still no sat_edge |

**Non-goals:** sat_edge gate PRs · MA peels · wipe 18 · free-pose · hungry cloud clip

---

## One-liner

*Prefer existing facing → probe outside `roof_*_e*` → fetch/attach SV → sat-align → FILM only if sparse → façades with product_lock + photo search (no sat_edge).*
