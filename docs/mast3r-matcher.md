# MASt3R as matcher only → posed COLMAP (TC3)

**Hard rule:** one ENU world frame. MASt3R supplies **2D–2D matches**, not
extrinsics. Free-pose sparse GA / GLOMAP / `--ignore_pose` are **not** the
product path.

Follows the spirit of naver/mast3r `kapture_mast3r_mapping.py`:
`has_pose` → write matches into a COLMAP DB → **`point_triangulator`** with
fixed priors. We skip the kapture dependency: `bootstrap_posed_database` +
`write_known_pose_model` already pin IMAGE_IDs and ENU `t = -R @ C`.

## CLI

```bash
# Equivalent forms — both lock ENU and call point_triangulator:
uv run ps1hood reconstruct <run> --backend mast3r
uv run ps1hood reconstruct <run> --backend colmap_posed --matcher mast3r

# CPU / no mast3r package — classic SIFT matches_importer (default):
uv run ps1hood reconstruct <run> --backend colmap_posed --matcher sift
```

Without the `mast3r`/`dust3r` packages or a CUDA/HIP GPU, the mast3r matcher
**fails loud** after exporting `recon/colmap/images/` + `sparse_prior/` +
`cross_pano_pairs.txt` (no silent free-pose fallback).

## Pipeline

1. Select posed sparse frames (keyframes + optional strided FILM mids with
   `lerp_pose` ENU) — same as `colmap_posed`.
2. Export images; write `sparse_prior` cameras/images/empty points3D from
   align ENU + FoV PINHOLE.
3. Build cross-pano **forward** pair list (`n_forward=3`).
4. `database_creator` + insert cameras/images with IMAGE_IDs matching the
   text model (no feature_extractor remap).
5. MASt3R pairwise matches on that list → keypoints + matches +
   `two_view_geometries` (config=CALIBRATED when skipping SIFT GV).
6. `point_triangulator` with `--clear_points 1`, two-view tracks allowed,
   intrinsics BA refine off. Extrinsics come from the ENU prior model.
7. PLY → `recon/cloud_photo.ply` + `recon/cloud.ply`.

## What we deliberately do **not** do

- `sparse_global_alignment` / MASt3R-SfM as the Studio hero cloud
- `demo_glomap.py` / free mapper as the default
- `--ignore_pose` / discarding align ENU
- CPU MASt3R (unsupported / useless)

Experimental free-pose remains as `run_mast3r_free_pose()` for research only
— not wired to the CLI.

## License / install

MASt3R code+weights are **CC BY-NC-SA 4.0** (hobby OK; flag before product).
Not vendored. See `pyproject.toml` `[project.optional-dependencies] mast3r`
and [`gpu-mast3r-ubuntu.md`](gpu-mast3r-ubuntu.md) for CUDA vs ROCm reality
on RX 6900 XT.

```bash
git clone --recursive https://github.com/naver/mast3r
# torch+CUDA, pip install -e ., download .pth — then reconstruct --backend mast3r
```

## Related

- [`rd-interp-and-v2pc.md`](rd-interp-and-v2pc.md) §B+ / TC3
- [`longer-tracks-no-oom.md`](longer-tracks-no-oom.md) (pair graph; MASt3R as §4)
- [`mapanything-densify.md`](mapanything-densify.md) (TC2 densify; separate)
