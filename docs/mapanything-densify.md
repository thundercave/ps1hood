# MapAnything densify (pose-locked ENU)

**Goal:** feed-forward dense cloud from Street View / FILM frames **without inventing extrinsics**.

**Hard rule:** fixed ENU cam2world. Never pass `--ignore_pose_inputs` /
`ignore_pose_inputs=True`. Discard any predicted poses from the API as
authority (log median ΔC only).

**Hardware:** needs a GPU torch stack (**CUDA** preferred). This box / AMD 6900 XT
have no NVIDIA CUDA — export the bundle here, run infer on cloud CUDA, or try
best-effort ROCm on gfx1030 ([mapanything-rocm-gfx1030.md](mapanything-rocm-gfx1030.md)).
Prefer Apache weights.

Upstream: [facebookresearch/map-anything](https://github.com/facebookresearch/map-anything)  
Recipe: `/workspace/mapanything-enu-recipe.md` (Scrapy).

---

## Install (cloud CUDA)

```bash
git clone https://github.com/facebookresearch/map-anything.git
cd map-anything
conda create -n mapanything python=3.12 -y && conda activate mapanything
# install torch for your CUDA first, then:
pip install -e ".[colmap]"
```

Weights (Hugging Face):

| Id | License | Use |
|----|---------|-----|
| `facebook/map-anything-apache` | Apache-2.0 | **Preferred** (`--apache`) |
| `facebook/map-anything` | CC-BY-NC 4.0 | Research only |

MapAnything is **not** vendored in this MIT tree (same pattern as OpenMVS / MASt3R).

---

## Path A — posed COLMAP demo (fastest) ★

After `ps1hood reconstruct <run> --backend colmap_posed`:

```bash
# On a CUDA host with map-anything cloned (MAPANYTHING_ROOT or on PATH):
python scripts/demo_inference_on_colmap_outputs.py \
  --colmap_path runs/<run>/recon/colmap/sparse_posed \
  --stride 2 \
  --apache \
  --save_glb \
  --save_colmap \
  --output_directory runs/<run>/mapanything/colmap_out

# DO NOT pass --ignore_pose_inputs
```

`ps1hood densify <run> --backend mapanything` prefers this path when the demo
script is found and posed sparse exists.

---

## Path B — align / interp bundle + Python API

### 1. Export (CPU OK)

```bash
uv run ps1hood export mapanything-bundle <run> --stride 2 --max-views 48
# → runs/<run>/mapanything/bundle/{images/,manifest.json,poses_cam2world.npz}

# or densify with export-only:
uv run ps1hood densify <run> --backend mapanything --export-only --stride 2
```

Manifest views match upstream multi-modal input:

- `image` — relative path under the bundle
- `intrinsics` — 3×3 K from crop FoV at image WxH (`K_from_fov`)
- `camera_poses` — 4×4 OpenCV **cam2world**  
  `R` from `camera_rotation_cv(heading, pitch)`, `C = (e, n, u)` ENU metres
- `is_metric_scale: true`
- `enu` — locked metadata (asserted equal to `camera_poses` on export / load)

### 2. Infer (CUDA)

```bash
# helper in this repo:
python scripts/run_mapanything_bundle.py \
  runs/<run>/mapanything/bundle --apache \
  --import-recon runs/<run>

# or: uv run ps1hood densify <run> --backend mapanything --apache --stride 2
```

Infer flags (also stored under `manifest.infer_flags`):

| Flag | Value |
|------|--------|
| `ignore_pose_inputs` | **False** |
| `ignore_calibration_inputs` | **False** |
| `ignore_pose_scale_inputs` | False |
| `ignore_depth_inputs` | True (we do not feed depth) |
| `is_metric_scale` | True on every view |
| `minibatch_size` | 1 (VRAM-safe) |
| `amp_dtype` | **`bf16` on CUDA**; **`fp16` when `torch.version.hip` is set** (ROCm / RDNA2 — see [mapanything-rocm-gfx1030.md](mapanything-rocm-gfx1030.md)) |
| `--ignore_pose_inputs` (CLI) | **never** |

### Product PLY frame (hard rule)

Studio (smoke-dense) once showed MapAnything ~591k pts in the **wrong frame**:
centroid ~(20.5, -0.5, 50.3)m, z p50~60.6m, ΔC vs cameras ~(+27,-3,+47)m,
while flow stayed on-street ~(−20,5,4). Predicted-pose |ΔC|~53–60m was
**discarded as authority** (correct), but fused geometry was still wrong because
raw model-world `pts3d` was written to PLY.

**Fix:** export camera-frame geometry with **our** locked cam2world only:

```text
WORLD_ENU = R_c2w @ X_cam + C_enu
```

- `R_c2w`, `C_enu=(e,n,u)` from `camera_rotation_cv` / align — never MA predicted pose
- Prefer `pts3d_cam` (else `depth_z` + rays/K); **refuse** dumping raw `pts3d` as ENU
- After fuse, **fail-loud** if cloud centroid is ≫20–25 m from camera centroid or
  cloud z p50 diverges from camera u p50 (the ~+50 m float symptom)

KEEP FLOW as product until these gates pass on a live re-run (PC).

Fuse → `mapanything/cloud.ply`, copy to `recon/cloud_mapanything.ply` and
`recon/cloud.ply` (Studio load path) only when the ENU check passes.

---

## Studio

Studio serves `recon/cloud.ply`. Successful densify copies the MapAnything PLY
there. Reload Studio / re-open the run. Scene cameras remain align ENU
(`recon/scene.json`).

---

## Failures

| Situation | Behavior |
|-----------|----------|
| No CUDA | Exit 1 + install hint; `--export-only` still works |
| MapAnything not installed | Exit 1 + clone/`pip install -e` hint |
| <2 posed frames | Fail loud — run align (and optionally interpolate) |
| Bundle pose drift vs `enu` | Fail loud on export / load assert |
| `--ignore_pose_inputs` | Refused in our argv builder |
| Raw `pts3d` without `pts3d_cam`/`depth_z` | Fail loud — refuse model-world as ENU |
| Cloud centroid / z p50 far from cameras | Fail loud ENU frame check (Studio +50 m bug) |

---

## Tests

```bash
uv run pytest -q tests/test_mapanything.py
```

Mocks cover bundle shape, cam2world vs w2c convention, `pts3d_cam`→ENU fuse
(locked poses), refuse-raw-`pts3d`, ENU centroid/z gates, COLMAP demo argv
(no ignore flag), CUDA fail-loud, and HIP → `fp16` AMP default — no GPU required.

---

## Links

- https://github.com/facebookresearch/map-anything  
- `scripts/demo_inference_on_colmap_outputs.py`  
- https://huggingface.co/facebook/map-anything-apache  
- Related: [rd-interp-and-v2pc.md](rd-interp-and-v2pc.md), [openmvs-densify.md](openmvs-densify.md), [mapanything-rocm-gfx1030.md](mapanything-rocm-gfx1030.md) (RX 6900 XT / ROCm)
