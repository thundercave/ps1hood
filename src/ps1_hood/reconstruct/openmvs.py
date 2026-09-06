"""Optional OpenMVS densify from posed COLMAP sparse + known poses.

OpenMVS is an **external AGPL-3.0 binary** — not vendored into this MIT tree.
We only shell out to ``InterfaceCOLMAP`` + ``DensifyPointCloud`` when they are
on PATH. Seed must be the posed COLMAP triangulated model (not raw flow street
cloud).

Path:
  posed sparse → colmap image_undistorter → InterfaceCOLMAP → DensifyPointCloud
  (CPU PatchMatch defaults; --resolution-level 2 for smoke-friendly RAM/time)

See docs/openmvs-densify.md and https://github.com/cdcseacave/openMVS/wiki/Usage
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ps1_hood.reconstruct.colmap import find_sparse_model

log = logging.getLogger(__name__)

INSTALL_HINT = """
Optional OpenMVS densify is AGPL-3.0 and is NOT redistributed in this MIT repo.

Install OpenMVS so these are on PATH:
  InterfaceCOLMAP
  DensifyPointCloud

Build / packages:
  https://github.com/cdcseacave/openMVS
  https://github.com/cdcseacave/openMVS/wiki

Also needs ``colmap`` (for image_undistorter).

Then:
  ps1hood densify <run> --backend openmvs

Requires a posed COLMAP triangulated sparse model under recon/colmap/sparse_posed
(seed from ``ps1hood reconstruct <run> --backend colmap_posed``). Do not densify
from an unfiltered flow street cloud.
""".strip()

# Smoke-friendly densify defaults (wiring note Milestone B).
DEFAULT_RESOLUTION_LEVEL = 2
DEFAULT_NUMBER_VIEWS = 4
DEFAULT_MIN_RESOLUTION = 640
DEFAULT_MAX_RESOLUTION = 1600
DEFAULT_NUMBER_VIEWS_FUSE = 2


def which_openmvs() -> dict[str, str] | None:
    """Return ``{InterfaceCOLMAP, DensifyPointCloud}`` paths, or None if missing."""
    interface = shutil.which("InterfaceCOLMAP")
    densify = shutil.which("DensifyPointCloud")
    if not interface or not densify:
        return None
    return {"InterfaceCOLMAP": interface, "DensifyPointCloud": densify}


def require_openmvs() -> dict[str, str]:
    bins = which_openmvs()
    if bins is None:
        raise RuntimeError(
            "OpenMVS not on PATH (need InterfaceCOLMAP + DensifyPointCloud).\n"
            + INSTALL_HINT
        )
    return bins


def require_colmap() -> str:
    colmap = shutil.which("colmap")
    if not colmap:
        raise RuntimeError(
            "colmap is not on PATH (needed for image_undistorter before OpenMVS).\n"
            + INSTALL_HINT
        )
    return colmap


def resolve_posed_inputs(
    run_root: Path,
    *,
    images_path: Path | None = None,
    sparse_path: Path | None = None,
) -> tuple[Path, Path]:
    """Locate COLMAP images + posed triangulated sparse model.

    Defaults:
      images  → ``<run>/recon/colmap/images``
      sparse  → ``find_sparse_model(<run>/recon/colmap/sparse_posed)``
    """
    root = run_root.resolve()
    images = Path(images_path) if images_path else root / "recon" / "colmap" / "images"
    if not images.is_dir() or not any(images.iterdir()):
        raise RuntimeError(
            f"no COLMAP images under {images}. "
            "Run: ps1hood reconstruct <run> --backend colmap_posed first."
        )

    if sparse_path is not None:
        sparse = Path(sparse_path)
        if sparse.is_dir() and not (
            (sparse / "points3D.bin").is_file() or (sparse / "points3D.txt").is_file()
        ):
            found = find_sparse_model(sparse)
            if found is None:
                raise RuntimeError(f"no points3D under posed sparse path {sparse}")
            sparse = found
    else:
        posed_root = root / "recon" / "colmap" / "sparse_posed"
        sparse = find_sparse_model(posed_root) if posed_root.is_dir() else None
        if sparse is None and posed_root.is_dir() and (
            (posed_root / "points3D.bin").is_file()
            or (posed_root / "points3D.txt").is_file()
        ):
            sparse = posed_root
        if sparse is None:
            raise RuntimeError(
                f"no posed COLMAP triangulated model under {posed_root}. "
                "Seed densify from posed sparse (colmap_posed), not raw flow street cloud.\n"
                "Run: ps1hood reconstruct <run> --backend colmap_posed"
            )
    return images, sparse


def densify_argv(
    densify_bin: str,
    scene_mvs: Path,
    *,
    resolution_level: int = DEFAULT_RESOLUTION_LEVEL,
    number_views: int = DEFAULT_NUMBER_VIEWS,
    min_resolution: int = DEFAULT_MIN_RESOLUTION,
    max_resolution: int = DEFAULT_MAX_RESOLUTION,
    number_views_fuse: int = DEFAULT_NUMBER_VIEWS_FUSE,
) -> list[str]:
    """Build DensifyPointCloud argv (smoke-friendly CPU flags)."""
    return [
        densify_bin,
        str(scene_mvs),
        "--resolution-level",
        str(int(resolution_level)),
        "--min-resolution",
        str(int(min_resolution)),
        "--max-resolution",
        str(int(max_resolution)),
        "--number-views",
        str(int(number_views)),
        "--number-views-fuse",
        str(int(number_views_fuse)),
    ]


def interface_colmap_argv(
    interface_bin: str,
    dense_workspace: Path,
    scene_mvs: Path,
    *,
    image_folder: Path | None = None,
) -> list[str]:
    """Build InterfaceCOLMAP argv for a COLMAP dense workspace."""
    images = image_folder if image_folder is not None else dense_workspace / "images"
    return [
        interface_bin,
        "-i",
        str(dense_workspace),
        "-o",
        str(scene_mvs),
        "--image-folder",
        str(images),
    ]


def undistorter_argv(
    colmap: str,
    *,
    image_path: Path,
    input_path: Path,
    output_path: Path,
) -> list[str]:
    return [
        colmap,
        "image_undistorter",
        "--image_path",
        str(image_path),
        "--input_path",
        str(input_path),
        "--output_path",
        str(output_path),
        "--output_type",
        "COLMAP",
    ]


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    log.info("OpenMVS/COLMAP: %s", " ".join(cmd[:4]))
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def run_openmvs_densify(
    run_root: Path,
    *,
    images_path: Path | None = None,
    sparse_path: Path | None = None,
    out_dir: Path | None = None,
    resolution_level: int = DEFAULT_RESOLUTION_LEVEL,
    number_views: int = DEFAULT_NUMBER_VIEWS,
    min_resolution: int = DEFAULT_MIN_RESOLUTION,
    max_resolution: int = DEFAULT_MAX_RESOLUTION,
    number_views_fuse: int = DEFAULT_NUMBER_VIEWS_FUSE,
) -> dict[str, Any]:
    """Densify posed COLMAP sparse → ``openmvs/scene_dense.ply``.

    Fails loud if OpenMVS / colmap missing, sparse seed missing, or PLY empty.
    Does not vendor OpenMVS; does not use OSM/BAG meshes.
    """
    bins = require_openmvs()
    colmap = require_colmap()
    images, sparse = resolve_posed_inputs(
        run_root, images_path=images_path, sparse_path=sparse_path
    )

    openmvs_dir = Path(out_dir) if out_dir is not None else run_root.resolve() / "openmvs"
    openmvs_dir.mkdir(parents=True, exist_ok=True)
    dense_ws = openmvs_dir / "dense"
    scene_mvs = openmvs_dir / "scene.mvs"
    dense_ply = openmvs_dir / "scene_dense.ply"

    if dense_ws.exists():
        shutil.rmtree(dense_ws)

    _run(
        undistorter_argv(
            colmap,
            image_path=images,
            input_path=sparse,
            output_path=dense_ws,
        )
    )
    if not (dense_ws / "images").is_dir():
        raise RuntimeError(
            f"image_undistorter did not write images under {dense_ws}. "
            "Check posed sparse model + COLMAP install."
        )

    _run(
        interface_colmap_argv(
            bins["InterfaceCOLMAP"],
            dense_ws,
            scene_mvs,
            image_folder=dense_ws / "images",
        )
    )
    if not scene_mvs.is_file():
        raise RuntimeError(f"InterfaceCOLMAP did not write {scene_mvs}")

    _run(
        densify_argv(
            bins["DensifyPointCloud"],
            scene_mvs,
            resolution_level=resolution_level,
            number_views=number_views,
            min_resolution=min_resolution,
            max_resolution=max_resolution,
            number_views_fuse=number_views_fuse,
        ),
        cwd=openmvs_dir,
    )

    # OpenMVS may write scene_dense.ply next to scene.mvs or under working folder.
    candidates = [
        dense_ply,
        openmvs_dir / "scene_dense.ply",
        scene_mvs.with_name("scene_dense.ply"),
    ]
    found = next((p for p in candidates if p.is_file() and p.stat().st_size > 64), None)
    if found is None:
        raise RuntimeError(
            f"DensifyPointCloud produced no usable PLY under {openmvs_dir}. "
            "Try lowering --resolution-level or check OpenMVS logs."
        )
    if found.resolve() != dense_ply.resolve():
        shutil.copy2(found, dense_ply)

    n_pts = _ply_vertex_count(dense_ply)
    if n_pts < 1:
        raise RuntimeError(f"OpenMVS densify PLY empty at {dense_ply}")

    log.info(
        "OpenMVS densify → %s (%s points; seed=%s; resolution-level=%s)",
        dense_ply,
        n_pts,
        sparse,
        resolution_level,
    )
    return {
        "path": str(dense_ply),
        "points": n_pts,
        "backend": "openmvs",
        "seed": str(sparse),
        "scene_mvs": str(scene_mvs),
        "resolution_level": int(resolution_level),
        "number_views": int(number_views),
        "agpl": True,
    }


def _ply_vertex_count(path: Path) -> int:
    # ASCII or binary PLY — vertex count is always in the header.
    with path.open("rb") as fh:
        for _ in range(64):
            line = fh.readline()
            if not line:
                break
            try:
                text = line.decode("ascii", errors="ignore").strip()
            except Exception:  # noqa: BLE001
                continue
            if text.startswith("element vertex"):
                parts = text.split()
                return int(parts[-1])
            if text == "end_header":
                break
    return 0
