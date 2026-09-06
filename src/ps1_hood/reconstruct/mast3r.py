"""Optional MASt3R / DUSt3R reconstruction backend.

Torch + MASt3R weights are *not* default dependencies. Install separately
(see RuntimeError message / README) and call ``run_mast3r``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ps1_hood.reconstruct.colmap import export_colmap_images
from ps1_hood.reconstruct.unproject import write_ply

log = logging.getLogger(__name__)

INSTALL_HINT = """
MASt3R is optional and not vendored. Install on a CUDA machine:

  git clone --recursive https://github.com/naver/mast3r
  cd mast3r
  # install torch matching your CUDA, then:
  pip install -e .
  # download checkpoint weights (see mast3r README)

Then re-run:
  ps1hood reconstruct <run> --backend mast3r

Needs a GPU; CPU-only runs are unsupported / extremely slow.
""".strip()


def _try_import_mast3r() -> bool:
    try:
        import dust3r  # noqa: F401
        import mast3r  # noqa: F401

        return True
    except ImportError:
        return False


def run_mast3r(
    frames: list[dict[str, Any]],
    recon_dir: Path,
    *,
    cloud_name: str = "cloud.ply",
) -> dict[str, Any]:
    """Run MASt3R sparse global alignment on keyframe images.

    Always exports images under ``recon/mast3r_input/images``. If mast3r /
    dust3r are not importable, raises ``RuntimeError`` with install steps.
    When present, writes ``recon/cloud.ply`` (RGB when available).
    """
    if len(frames) < 2:
        raise RuntimeError("MASt3R needs at least 2 keyframes with images")

    input_ws, _names = export_colmap_images(frames, recon_dir / "mast3r_input")
    images_dir = input_ws / "images"

    if not _try_import_mast3r():
        raise RuntimeError(
            f"MASt3R / DUSt3R not installed. Keyframes exported to {images_dir}.\n"
            + INSTALL_HINT
        )

    cloud_path = recon_dir / cloud_name
    meta = _run_sparse_ga(images_dir, cloud_path)
    return meta


def _run_sparse_ga(images_dir: Path, dest_ply: Path) -> dict[str, Any]:
    """Best-effort MASt3R-SfM / demo-style sparse alignment → PLY."""
    import numpy as np

    image_paths = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if len(image_paths) < 2:
        raise RuntimeError(f"need ≥2 images in {images_dir}")

    # Lazy imports — only when packages exist.
    from dust3r.utils.image import load_images  # type: ignore[import-untyped]
    from mast3r.model import AsymmetricMASt3R  # type: ignore[import-untyped]

    device = _pick_device()
    weights = _find_weights()
    log.info("MASt3R: loading model on %s (%s)", device, weights or "hub/default")
    if weights:
        model = AsymmetricMASt3R.from_pretrained(weights).to(device)
    else:
        # fall through to HF hub id used by upstream demos
        model = AsymmetricMASt3R.from_pretrained(
            "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
        ).to(device)
    model.eval()

    imgs = load_images([str(p) for p in image_paths], size=512)
    try:
        from mast3r.cloud_opt.sparse_ga import sparse_global_alignment  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "mast3r is installed but sparse_global_alignment is missing. "
            "Use a full naver/mast3r checkout with submodules.\n" + INSTALL_HINT
        ) from exc

    cache = images_dir.parent / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    scene = sparse_global_alignment(
        [str(p) for p in image_paths],
        imgs,
        cache,
        model,
        device=device,
    )

    xyz, rgb = _scene_points(scene)
    if xyz is None or len(xyz) == 0:
        raise RuntimeError("MASt3R produced an empty point cloud")

    dest_ply.parent.mkdir(parents=True, exist_ok=True)
    if rgb is None:
        rgb = np.full((len(xyz), 3), 180, dtype=np.uint8)
    write_ply(dest_ply, np.asarray(xyz, dtype=np.float64), np.asarray(rgb, dtype=np.uint8))
    log.info("MASt3R wrote %s points → %s", len(xyz), dest_ply)
    return {"path": str(dest_ply), "points": int(len(xyz)), "backend": "mast3r"}


def _pick_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    log.warning("CUDA not available — MASt3R will be very slow on CPU")
    return "cpu"


def _find_weights() -> str | None:
    """Look for a local checkpoint if the user already downloaded one."""
    candidates = [
        Path.home() / ".cache" / "torch" / "hub" / "checkpoints",
        Path.home() / ".cache" / "huggingface" / "hub",
        Path("/workspace/mast3r"),
        Path.cwd() / "mast3r",
    ]
    names = (
        "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
        "*MASt3R*.pth",
    )
    for base in candidates:
        if not base.exists():
            continue
        for pattern in names:
            hits = list(base.rglob(pattern)) if "*" in pattern else [base / pattern]
            for hit in hits:
                if hit.is_file():
                    return str(hit)
    return None


def _scene_points(scene: Any) -> tuple[Any, Any]:
    """Extract Nx3 xyz (+ optional RGB) from a MASt3R/DUSt3R scene object."""
    import numpy as np

    # Newer sparse_ga scenes expose get_dense_pts3d / get_pts3d.
    for method in ("get_dense_pts3d", "get_pts3d"):
        fn = getattr(scene, method, None)
        if not callable(fn):
            continue
        try:
            pts = fn()
        except TypeError:
            pts = fn(raw=False)
        xyz_list: list[np.ndarray] = []
        rgb_list: list[np.ndarray] = []
        # pts may be list of tensors / arrays per view
        if isinstance(pts, (list, tuple)):
            for p in pts:
                arr = _to_numpy(p).reshape(-1, 3)
                xyz_list.append(arr)
        else:
            xyz_list.append(_to_numpy(pts).reshape(-1, 3))
        xyz = np.concatenate(xyz_list, axis=0) if xyz_list else None
        # colours if present on scene
        cols = getattr(scene, "imgs", None) or getattr(scene, "rgb", None)
        if cols is not None and xyz is not None:
            try:
                for im in cols:
                    arr = _to_numpy(im)
                    if arr.ndim == 3 and arr.shape[-1] >= 3:
                        rgb_list.append((arr[..., :3].reshape(-1, 3) * (255.0 if arr.max() <= 1.5 else 1.0)).astype(np.uint8))
                if rgb_list and sum(len(r) for r in rgb_list) == len(xyz):
                    return xyz, np.concatenate(rgb_list, axis=0)
            except Exception:  # noqa: BLE001
                pass
        return xyz, None

    # Fallback: attrs used by some demo dumps
    for attr in ("pts3d", "points", "xyz"):
        if hasattr(scene, attr):
            return _to_numpy(getattr(scene, attr)).reshape(-1, 3), None
    raise RuntimeError("could not extract points from MASt3R scene")


def _to_numpy(x: Any) -> Any:
    import numpy as np

    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)
