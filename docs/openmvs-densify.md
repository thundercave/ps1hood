# Optional OpenMVS densify (Milestone B)

**Goal:** denser photo-derived cloud from **known poses + posed sparse seed**, CPU-friendly.

**Not:** OSM/BAG hero meshes. **Not:** densifying an unfiltered flow street cloud.

## License

OpenMVS is **AGPL-3.0**. This repo stays **MIT** and does **not** vendor or redistribute OpenMVS binaries. Densify is an optional external tool invoked by CLI when present on `PATH`.

If you ship a SaaS or closed binary that *links* OpenMVS, you owe AGPL obligations (source offer). Shelling out to a user-installed binary for personal hobby use is the intended path here.

## Install

Put these on `PATH` (build from [cdcseacave/openMVS](https://github.com/cdcseacave/openMVS) or your distro package):

- `InterfaceCOLMAP`
- `DensifyPointCloud`

Also need `colmap` (for `image_undistorter`).

```bash
# sanity
which InterfaceCOLMAP DensifyPointCloud colmap
```

## Pipeline (what `ps1hood densify` runs)

Prerequisite: posed sparse from known-pose triangulator:

```bash
uv run ps1hood reconstruct <run> --backend colmap_posed
# → recon/colmap/images + recon/colmap/sparse_posed + recon/cloud_photo.ply
```

Then:

```bash
uv run ps1hood densify <run> --backend openmvs
# optional: --resolution-level 2 --number-views 4
```

Internally:

1. `colmap image_undistorter`  
   `recon/colmap/images` + `recon/colmap/sparse_posed` → `openmvs/dense/` (COLMAP)
2. `InterfaceCOLMAP -i openmvs/dense -o openmvs/scene.mvs --image-folder openmvs/dense/images`
3. `DensifyPointCloud openmvs/scene.mvs --resolution-level 2 --number-views 4 …`

Output: **`openmvs/scene_dense.ply`** (+ `scene_dense.mvs`, depth maps).

Smoke-friendly defaults keep RAM/time sane (`--resolution-level 2` = ¼ linear). Drop to `1` / `0` once it works.

## Failures

| Situation | Behavior |
|-----------|----------|
| OpenMVS not on PATH | Exit 1 + install / AGPL message (no crash loop) |
| No posed sparse under `recon/colmap/sparse_posed` | Fail loud — run `colmap_posed` first |
| Empty densify PLY | Fail loud |

## Seed tips

| Seed | Use |
|------|-----|
| Posed COLMAP triangulated | **Preferred** |
| Filtered flow cloud | Only if street/ground dominance stripped |
| Photo-plane inliers | Optional extra points with visibility |

Cross-pano images only (same rule as posed triangulator). Same-center orbit mates waste densify.

## CPU note

Stock / non-CUDA OpenMVS densifies on **CPU** — correct for AMD boxes without CUDA PatchMatch. Do not expect CUDA-only postprocess defaults.

## Links

- https://github.com/cdcseacave/openMVS  
- https://github.com/cdcseacave/openMVS/wiki/Usage  
- AGPL-3.0: https://www.gnu.org/licenses/agpl-3.0.html  
