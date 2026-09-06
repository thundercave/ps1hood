"""Optional MapAnything densify with locked ENU extrinsics.

MapAnything (Meta, Apache-2.0 code; prefer ``facebook/map-anything-apache``
weights) accepts images + intrinsics + OpenCV cam2world poses. We feed our
align / lerp_pose ENU and never enable ``ignore_pose_inputs``.

Product PLY hard rule (Studio frame-fix):
  WORLD_ENU = R_c2w @ X_cam + C_enu
using **our** cam2world from ``camera_rotation_cv`` + (e,n,u) — never MA
predicted poses. Prefer ``pts3d_cam`` / ``depth_z``; refuse dumping raw
``pts3d`` (model world) as ENU. Fail-loud if cloud centroid / z p50 diverge
from camera centers (the ~+50 m float symptom).

Two paths:
  A) Posed COLMAP folder → upstream
     ``scripts/demo_inference_on_colmap_outputs.py --apache …``
     (never ``--ignore_pose_inputs``).
  B) Bundle from align/interp frames → ``preprocess_inputs`` + ``model.infer``
     with ``ignore_pose_inputs=False``.

Binary / weights are NOT vendored. Without CUDA the densify runner fails
loud; export-only works on CPU.

See docs/mapanything-densify.md and /workspace/mapanything-enu-recipe.md.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ps1_hood.geo import camera_rotation_cv
from ps1_hood.reconstruct.colmap import export_colmap_images, find_sparse_model
from ps1_hood.reconstruct.unproject import write_ply

log = logging.getLogger(__name__)

BUNDLE_FORMAT = "ps1hood-mapanything-bundle-v1"
DEFAULT_HF_MODEL = "facebook/map-anything-apache"
DEFAULT_STRIDE = 2
DEFAULT_MAX_VIEWS = 48

INSTALL_HINT = """
Optional MapAnything densify is NOT redistributed in this MIT repo.

Install on a CUDA machine (cloud preferred; AMD 6900 XT has no CUDA):

  git clone https://github.com/facebookresearch/map-anything.git
  cd map-anything
  pip install -e ".[colmap]"
  # Prefer Apache weights: facebook/map-anything-apache

Then either:
  ps1hood densify <run> --backend mapanything --apache --stride 2
  # or export a pose-locked bundle and run the helper:
  ps1hood export mapanything-bundle <run>
  python scripts/run_mapanything_bundle.py runs/<run>/mapanything/bundle

NEVER pass --ignore_pose_inputs / ignore_pose_inputs=True — ENU poses must stay locked.

Docs: docs/mapanything-densify.md
Upstream: https://github.com/facebookresearch/map-anything
""".strip()


def K_from_fov(width: int, height: int, fov_deg: float) -> np.ndarray:
    """Pinhole K from horizontal FoV at image resolution (OpenCV)."""
    f = 0.5 * float(width) / math.tan(math.radians(float(fov_deg)) / 2.0)
    cx = 0.5 * float(width)
    cy = 0.5 * float(height)
    return np.array(
        [[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def cam2world_from_frame(frame: dict[str, Any]) -> np.ndarray:
    """OpenCV cam2world 4×4 from ENU centre + ``camera_rotation_cv``.

    World = ENU metres (X east, Y north, Z up). Camera = OpenCV
    (+X right, +Y down, +Z forward). Centre C = (e, n, u).
    """
    R = np.asarray(
        camera_rotation_cv(float(frame["heading"]), float(frame.get("pitch") or 0.0)),
        dtype=np.float64,
    )
    C = np.array(
        [float(frame["e"]), float(frame["n"]), float(frame["u"])],
        dtype=np.float64,
    )
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = C
    return T


def _frame_size(frame: dict[str, Any]) -> tuple[int, int]:
    w = frame.get("width")
    h = frame.get("height")
    if w and h:
        return int(w), int(h)
    img = cv2.imread(str(frame["path"]), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"cannot read image for MapAnything export: {frame['path']}")
    hh, ww = img.shape[:2]
    return int(ww), int(hh)


def assert_frame_pose_lock(frame: dict[str, Any], *, index: int = 0) -> None:
    """Fail loud if a frame lacks ENU + FoV needed for locked poses."""
    missing = [k for k in ("e", "n", "u", "heading", "fov", "path") if frame.get(k) is None]
    if missing:
        raise RuntimeError(
            f"MapAnything frame {index} missing {missing}; "
            "poses must come from align / lerp_pose ENU, not invented"
        )
    for key in ("e", "n", "u", "heading", "fov"):
        val = float(frame[key])
        if not math.isfinite(val):
            raise RuntimeError(f"MapAnything frame {index}: {key}={val} is not finite")
    pitch = float(frame.get("pitch") or 0.0)
    if not math.isfinite(pitch):
        raise RuntimeError(f"MapAnything frame {index}: pitch={pitch} is not finite")


def view_record_from_frame(
    frame: dict[str, Any],
    *,
    image_rel: str,
    index: int = 0,
) -> dict[str, Any]:
    """Serializable MapAnything view + locked ENU metadata."""
    assert_frame_pose_lock(frame, index=index)
    w, h = _frame_size(frame)
    K = K_from_fov(w, h, float(frame["fov"]))
    T = cam2world_from_frame(frame)
    return {
        "index": int(index),
        "image": image_rel,
        "width": w,
        "height": h,
        "intrinsics": K.tolist(),
        "camera_poses": T.tolist(),
        "is_metric_scale": True,
        "enu": {
            "e": float(frame["e"]),
            "n": float(frame["n"]),
            "u": float(frame["u"]),
            "heading": float(frame["heading"]),
            "pitch": float(frame.get("pitch") or 0.0),
            "fov": float(frame["fov"]),
        },
        "pano_id": frame.get("pano_id"),
        "interpolated": bool(frame.get("interpolated", False)),
    }


def assert_bundle_pose_lock(manifest: dict[str, Any]) -> None:
    """Verify every view has cam2world matching enu metadata (pose lock)."""
    views = manifest.get("views") or []
    if len(views) < 2:
        raise RuntimeError(
            f"MapAnything bundle needs ≥2 views with locked poses (got {len(views)})"
        )
    if not manifest.get("pose_lock", True):
        raise RuntimeError("MapAnything bundle pose_lock must be true")
    for i, view in enumerate(views):
        if not view.get("is_metric_scale", False):
            raise RuntimeError(f"view {i}: is_metric_scale must be True")
        enu = view.get("enu") or {}
        for key in ("e", "n", "u", "heading", "fov"):
            if key not in enu:
                raise RuntimeError(f"view {i}: missing enu.{key}")
        T = np.asarray(view["camera_poses"], dtype=np.float64)
        if T.shape != (4, 4):
            raise RuntimeError(f"view {i}: camera_poses must be 4×4, got {T.shape}")
        expected = cam2world_from_frame(
            {
                "e": enu["e"],
                "n": enu["n"],
                "u": enu["u"],
                "heading": enu["heading"],
                "pitch": enu.get("pitch") or 0.0,
            }
        )
        if not np.allclose(T, expected, atol=1e-5, rtol=0):
            raise RuntimeError(
                f"view {i}: camera_poses drifted from locked ENU "
                f"(max|Δ|={float(np.max(np.abs(T - expected))):.3e})"
            )
        K = np.asarray(view["intrinsics"], dtype=np.float64)
        if K.shape != (3, 3):
            raise RuntimeError(f"view {i}: intrinsics must be 3×3, got {K.shape}")
        w, h = int(view["width"]), int(view["height"])
        K_exp = K_from_fov(w, h, float(enu["fov"]))
        if not np.allclose(K, K_exp, atol=1e-4, rtol=0):
            raise RuntimeError(f"view {i}: intrinsics do not match FoV lock")


def subsample_frames(
    frames: list[dict[str, Any]],
    *,
    stride: int = DEFAULT_STRIDE,
    max_views: int | None = DEFAULT_MAX_VIEWS,
) -> list[dict[str, Any]]:
    """Even subsample by index stride, then optional max_views cap."""
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    chosen = list(frames[:: int(stride)]) if stride > 1 else list(frames)
    if max_views is not None and max_views > 0 and len(chosen) > max_views:
        from ps1_hood.capture.discover import subsample_evenly

        chosen = subsample_evenly(chosen, int(max_views))
    return chosen


def resolve_mapanything_frames(
    project: Any,
    *,
    stride: int = DEFAULT_STRIDE,
    max_views: int | None = DEFAULT_MAX_VIEWS,
    prefer_interp: bool = True,
) -> list[dict[str, Any]]:
    """Load keyframes (+ optional densify-stride interp) with ENU poses."""
    from ps1_hood.interpolate.sequence import select_densify_frames
    from ps1_hood.reconstruct.keyframes import load_keyframes

    keyframes = load_keyframes(project)
    frames: list[dict[str, Any]] = list(keyframes)

    if prefer_interp:
        frames_path = project.interp_dir / "frames.json"
        if frames_path.is_file():
            raw = project.read_json(frames_path)
            if isinstance(raw, list) and raw:
                try:
                    densify = select_densify_frames(raw)
                    # Prefer densify list when it has real image paths.
                    usable = [f for f in densify if f.get("path") and Path(f["path"]).is_file()]
                    if len(usable) >= 2:
                        frames = usable
                except RuntimeError as exc:
                    log.warning("interp densify frames skipped: %s", exc)

    if len(frames) < 2:
        raise RuntimeError(
            "MapAnything needs ≥2 posed frames with images "
            f"(keyframes={len(keyframes)}). Run align (and optionally interpolate) first."
        )
    return subsample_frames(frames, stride=stride, max_views=max_views)


def export_mapanything_bundle(
    frames: list[dict[str, Any]],
    out_dir: Path,
    *,
    stride: int = 1,
    max_views: int | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    """Write images/ + manifest.json in the format MapAnything expects.

    Manifest fields per view match upstream multi-modal infer:
    ``img`` path, ``intrinsics`` 3×3, ``camera_poses`` 4×4 OpenCV cam2world,
    ``is_metric_scale`` True. ENU extras are for pose-lock asserts only.
    """
    chosen = subsample_frames(frames, stride=stride, max_views=max_views)
    if len(chosen) < 2:
        raise RuntimeError(f"need ≥2 frames after subsample (got {len(chosen)})")

    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    images_dir = out / "images"
    images_dir.mkdir(parents=True)

    # Reuse COLMAP-style unique names for stable relative paths.
    _, names = export_colmap_images(chosen, out)
    views: list[dict[str, Any]] = []
    for i, (frame, name) in enumerate(zip(chosen, names, strict=True)):
        views.append(
            view_record_from_frame(frame, image_rel=f"images/{name}", index=i)
        )

    # Stacked arrays for runner / tests (pose lock).
    poses = np.stack(
        [np.asarray(v["camera_poses"], dtype=np.float64) for v in views], axis=0
    )
    Ks = np.stack(
        [np.asarray(v["intrinsics"], dtype=np.float64) for v in views], axis=0
    )
    np.savez_compressed(out / "poses_cam2world.npz", poses=poses, intrinsics=Ks)

    manifest: dict[str, Any] = {
        "format": BUNDLE_FORMAT,
        "run": run_name,
        "pose_lock": True,
        "ignore_pose_inputs": False,
        "hf_model_preferred": DEFAULT_HF_MODEL,
        "convention": {
            "camera_poses": "OpenCV cam2world (+X right, +Y down, +Z forward)",
            "world": "ENU metres (X east, Y north, Z up)",
            "intrinsics": "3x3 pinhole K at source image WxH",
            "is_metric_scale": True,
        },
        "n_views": len(views),
        "views": views,
        "infer_flags": {
            "ignore_pose_inputs": False,
            "ignore_calibration_inputs": False,
            "ignore_pose_scale_inputs": False,
            "ignore_depth_inputs": True,
            "memory_efficient_inference": True,
            "minibatch_size": 1,
            "apply_mask": True,
            "mask_edges": True,
            "apply_confidence_mask": True,
            "confidence_percentile": 10,
        },
    }
    assert_bundle_pose_lock(manifest)
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (out / "README.txt").write_text(
        "ps1hood MapAnything bundle (pose-locked ENU).\n"
        "Load manifest.json views → preprocess_inputs → model.infer(\n"
        "  ignore_pose_inputs=False, ignore_calibration_inputs=False, …\n"
        "). Prefer facebook/map-anything-apache. NEVER --ignore_pose_inputs.\n"
        "See docs/mapanything-densify.md\n",
        encoding="utf-8",
    )
    log.info("MapAnything bundle → %s (%s views)", out, len(views))
    return {
        "path": str(out),
        "manifest": str(out / "manifest.json"),
        "n_views": len(views),
        "pose_lock": True,
        "format": BUNDLE_FORMAT,
    }


def which_mapanything() -> dict[str, Any] | None:
    """Return import/runtime info if ``mapanything`` is importable, else None."""
    try:
        import mapanything  # noqa: F401
    except ImportError:
        return None
    info: dict[str, Any] = {"package": True}
    try:
        import torch

        info["torch"] = True
        info["cuda"] = bool(torch.cuda.is_available())
    except ImportError:
        info["torch"] = False
        info["cuda"] = False
    demo = shutil.which("demo_inference_on_colmap_outputs.py")
    info["colmap_demo"] = demo
    return info


def require_cuda_for_densify() -> None:
    """Fail loud when densify is requested without CUDA."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "MapAnything densify needs PyTorch + CUDA.\n" + INSTALL_HINT
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "MapAnything densify requires CUDA (this box/GPU has none). "
            "Export a pose-locked bundle on CPU, then run on cloud CUDA:\n"
            "  ps1hood export mapanything-bundle <run>\n"
            "  # on CUDA host:\n"
            "  python scripts/run_mapanything_bundle.py "
            "runs/<run>/mapanything/bundle --apache\n"
            + INSTALL_HINT
        )


def default_amp_dtype() -> str:
    """AMP dtype for MapAnything ``model.infer``.

    NVIDIA CUDA: ``bf16`` (upstream recipe default).
    HIP/ROCm (``torch.version.hip`` set, e.g. RX 6900 XT / gfx1030): ``fp16`` —
    bf16 and flash-attn often break or NaN on RDNA2.
    """
    try:
        import torch
    except ImportError:
        return "bf16"
    hip = getattr(getattr(torch, "version", None), "hip", None)
    if hip:  # non-empty string on ROCm builds
        return "fp16"
    return "bf16"


def resolve_posed_colmap(run_root: Path) -> tuple[Path, Path] | None:
    """Locate posed COLMAP sparse + images if present (Path A)."""
    root = run_root.resolve()
    images = root / "recon" / "colmap" / "images"
    posed_root = root / "recon" / "colmap" / "sparse_posed"
    if not images.is_dir():
        return None
    sparse = find_sparse_model(posed_root) if posed_root.is_dir() else None
    if sparse is None and posed_root.is_dir() and (
        (posed_root / "points3D.bin").is_file() or (posed_root / "points3D.txt").is_file()
    ):
        sparse = posed_root
    if sparse is None:
        return None
    return images, sparse


def colmap_demo_argv(
    demo_script: Path,
    *,
    colmap_path: Path,
    output_directory: Path,
    stride: int = DEFAULT_STRIDE,
    apache: bool = True,
    save_glb: bool = True,
    save_colmap: bool = True,
    ext: str | None = None,
) -> list[str]:
    """Build upstream COLMAP demo argv — never includes --ignore_pose_inputs."""
    cmd = [
        sys.executable,
        str(demo_script),
        "--colmap_path",
        str(colmap_path),
        "--stride",
        str(int(stride)),
        "--output_directory",
        str(output_directory),
    ]
    if apache:
        cmd.append("--apache")
    if save_glb:
        cmd.append("--save_glb")
    if save_colmap:
        cmd.append("--save_colmap")
    if ext:
        cmd.extend(["--ext", ext])
    # Hard rule: do not append --ignore_pose_inputs.
    if "--ignore_pose_inputs" in cmd:
        raise RuntimeError("refusing to pass --ignore_pose_inputs (ENU pose lock)")
    return cmd


def find_mapanything_colmap_demo() -> Path | None:
    """Locate ``demo_inference_on_colmap_outputs.py`` if installed/cloned."""
    which = shutil.which("demo_inference_on_colmap_outputs.py")
    if which:
        return Path(which)
    try:
        import mapanything

        pkg = Path(mapanything.__file__).resolve().parent
    except ImportError:
        pkg = None
    candidates: list[Path] = []
    if pkg is not None:
        # site-packages/mapanything → repo root may be parents[1] when editable
        candidates.extend(
            [
                pkg.parent / "scripts" / "demo_inference_on_colmap_outputs.py",
                pkg.parents[1] / "scripts" / "demo_inference_on_colmap_outputs.py",
            ]
        )
    for env in ("MAPANYTHING_ROOT", "MAP_ANYTHING_ROOT"):
        root = __import__("os").environ.get(env)
        if root:
            candidates.append(Path(root) / "scripts" / "demo_inference_on_colmap_outputs.py")
    for home in (Path.home() / "map-anything", Path("/opt/map-anything"), Path("/workspace/map-anything")):
        candidates.append(home / "scripts" / "demo_inference_on_colmap_outputs.py")
    for c in candidates:
        if c.is_file():
            return c
    return None


def import_dense_ply_to_recon(
    ply_src: Path,
    run_root: Path,
    *,
    also_studio_cloud: bool = True,
) -> dict[str, Any]:
    """Copy densify PLY into recon/ for Studio load path."""
    recon = run_root.resolve() / "recon"
    recon.mkdir(parents=True, exist_ok=True)
    dest = recon / "cloud_mapanything.ply"
    shutil.copy2(ply_src, dest)
    studio = None
    if also_studio_cloud:
        studio = recon / "cloud.ply"
        shutil.copy2(ply_src, studio)
    n = _ply_vertex_count(dest)
    return {
        "cloud_mapanything": str(dest),
        "cloud": str(studio) if studio else None,
        "points": n,
    }


def _ply_vertex_count(path: Path) -> int:
    with path.open("rb") as fh:
        for _ in range(64):
            line = fh.readline()
            if not line:
                break
            text = line.decode("ascii", errors="ignore").strip()
            if text.startswith("element vertex"):
                return int(text.split()[-1])
            if text == "end_header":
                break
    return 0


def _median_pose_drift_m(
    locked: list[np.ndarray], predicted: list[np.ndarray]
) -> float | None:
    if not locked or len(locked) != len(predicted):
        return None
    deltas = [
        float(np.linalg.norm(np.asarray(p)[:3, 3] - np.asarray(l)[:3, 3]))
        for l, p in zip(locked, predicted, strict=True)
    ]
    return float(np.median(deltas)) if deltas else None


def load_bundle_views_for_infer(
    bundle_dir: Path,
    *,
    device: str = "cpu",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load bundle manifest into MapAnything view dicts (numpy / torch tensors)."""
    import torch

    manifest_path = Path(bundle_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert_bundle_pose_lock(manifest)
    views: list[dict[str, Any]] = []
    for i, rec in enumerate(manifest["views"]):
        img_path = Path(bundle_dir) / rec["image"]
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"cannot read bundle image {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        if (w, h) != (int(rec["width"]), int(rec["height"])):
            log.warning(
                "view %s size on disk %sx%s != manifest %sx%s — recompute K from FoV",
                i,
                w,
                h,
                rec["width"],
                rec["height"],
            )
            K = K_from_fov(w, h, float(rec["enu"]["fov"]))
        else:
            K = np.asarray(rec["intrinsics"], dtype=np.float32)
        T = np.asarray(rec["camera_poses"], dtype=np.float32)
        views.append(
            {
                "img": rgb,  # HxWx3 uint8 [0,255]
                "intrinsics": K,
                "camera_poses": T,
                "is_metric_scale": torch.tensor([True], device=device),
            }
        )
    return views, manifest


# Sanity gates for product PLY (Studio: MA cloud floated at z~50 when raw pts3d dumped).
DEFAULT_CENTROID_DELTA_MAX_M = 25.0
DEFAULT_Z_P50_DELTA_MAX_M = 15.0


def cam_points_to_enu(
    pts_cam: np.ndarray,
    R_c2w: np.ndarray,
    C_enu: np.ndarray,
) -> np.ndarray:
    """WORLD_ENU = R_c2w @ X_cam + C_enu (our poses — never MA predicted)."""
    pts = np.asarray(pts_cam, dtype=np.float64).reshape(-1, 3)
    R = np.asarray(R_c2w, dtype=np.float64).reshape(3, 3)
    C = np.asarray(C_enu, dtype=np.float64).reshape(3)
    return (pts @ R.T) + C


def _squeeze_hw3(arr: np.ndarray) -> np.ndarray:
    """Normalize MapAnything (B,H,W,3) / (H,W,3) tensors to HxWx3."""
    a = np.asarray(arr)
    while a.ndim > 3 and a.shape[0] == 1:
        a = a[0]
    if a.ndim != 3 or a.shape[-1] != 3:
        raise RuntimeError(f"expected HxWx3 geometry, got shape {a.shape}")
    return a


def _squeeze_hw1(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    while a.ndim > 2 and a.shape[0] == 1:
        a = a[0]
    if a.ndim == 3 and a.shape[-1] == 1:
        a = a[..., 0]
    if a.ndim != 2:
        raise RuntimeError(f"expected HxW depth/mask, got shape {arr.shape}")
    return a


def pts_cam_from_prediction(pred: dict[str, Any]) -> np.ndarray:
    """Camera-frame points per pixel (HxWx3). Prefer ``pts3d_cam``, else depth_z.

    Hard rule: product export must transform these with **our** cam2world.
    Raw ``pts3d`` (model "world") is intentionally not returned — Studio saw
    floating islands when it was dumped as ENU while predicted |ΔC|~50m.
    """
    if pred.get("pts3d_cam") is not None:
        return _squeeze_hw3(_as_numpy(pred["pts3d_cam"])).astype(np.float64)

    depth = pred.get("depth_z")
    rays = pred.get("ray_directions")
    if depth is not None and rays is not None:
        d = _squeeze_hw1(_as_numpy(depth)).astype(np.float64)
        r = _squeeze_hw3(_as_numpy(rays)).astype(np.float64)
        # ray_directions are camera-frame; MapAnything depth_z is Z-depth.
        # Prefer depth_along_ray if present for ray * t; else scale rays so Z=depth_z.
        along = pred.get("depth_along_ray")
        if along is not None:
            t = _squeeze_hw1(_as_numpy(along)).astype(np.float64)
            return r * t[..., None]
        # Scale each ray so its Z component equals depth_z (pinhole Z-depth).
        z_comp = np.clip(r[..., 2], 1e-8, None)
        return r * (d / z_comp)[..., None]

    if depth is not None and pred.get("intrinsics") is not None:
        d = _squeeze_hw1(_as_numpy(depth)).astype(np.float64)
        K = _as_numpy(pred["intrinsics"])
        while K.ndim > 2 and K.shape[0] == 1:
            K = K[0]
        K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        h, w = d.shape
        us, vs = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        z = d
        x = (us - cx) / fx * z
        y = (vs - cy) / fy * z
        return np.stack([x, y, z], axis=-1)

    raise RuntimeError(
        "MapAnything prediction lacks pts3d_cam (and depth_z fallback). "
        "Refusing to export raw pts3d as ENU — that caused the Studio frame bug "
        "(centroid z~50m vs street cameras)."
    )


def _colors_for_pred(
    pred: dict[str, Any], mask_flat: np.ndarray | None, n_pts: int
) -> np.ndarray:
    img = pred.get("img_no_norm")
    if img is None:
        return np.full((n_pts, 3), 200, dtype=np.uint8)
    img_np = _as_numpy(img)
    if img_np.ndim == 4:
        img_np = img_np[0]
    flat = (img_np.reshape(-1, 3) * 255.0).clip(0, 255).astype(np.uint8)
    if mask_flat is not None and flat.shape[0] == mask_flat.shape[0]:
        return flat[mask_flat]
    return flat[:n_pts] if flat.shape[0] >= n_pts else np.full((n_pts, 3), 200, dtype=np.uint8)


def assert_cloud_near_cameras(
    xyz: np.ndarray,
    camera_centers: np.ndarray,
    *,
    centroid_delta_max_m: float = DEFAULT_CENTROID_DELTA_MAX_M,
    z_p50_delta_max_m: float = DEFAULT_Z_P50_DELTA_MAX_M,
) -> dict[str, float]:
    """Fail loud if fused cloud is not in the same ENU street frame as cameras.

    Studio symptom of the raw-pts3d bug: cloud z p50~60 while cameras u~3–4,
    centroid ΔC tens of metres.
    """
    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    cams = np.asarray(camera_centers, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        raise RuntimeError("ENU check: empty cloud")
    if len(cams) == 0:
        raise RuntimeError("ENU check: no camera centers")

    cloud_c = pts.mean(axis=0)
    cam_c = cams.mean(axis=0)
    delta_c = float(np.linalg.norm(cloud_c - cam_c))
    z_p50 = float(np.median(pts[:, 2]))
    cam_u_p50 = float(np.median(cams[:, 2]))
    z_delta = abs(z_p50 - cam_u_p50)

    stats = {
        "cloud_centroid_e": float(cloud_c[0]),
        "cloud_centroid_n": float(cloud_c[1]),
        "cloud_centroid_u": float(cloud_c[2]),
        "cam_centroid_e": float(cam_c[0]),
        "cam_centroid_n": float(cam_c[1]),
        "cam_centroid_u": float(cam_c[2]),
        "centroid_delta_m": delta_c,
        "cloud_z_p50": z_p50,
        "cam_u_p50": cam_u_p50,
        "z_p50_delta_m": float(z_delta),
    }

    problems: list[str] = []
    if delta_c > centroid_delta_max_m:
        problems.append(
            f"cloud centroid |ΔC|={delta_c:.1f}m > {centroid_delta_max_m}m vs cameras "
            f"(cloud={cloud_c.round(2).tolist()}, cams={cam_c.round(2).tolist()})"
        )
    if z_delta > z_p50_delta_max_m:
        problems.append(
            f"cloud z p50={z_p50:.1f}m vs camera u p50={cam_u_p50:.1f}m "
            f"(|Δ|={z_delta:.1f}m > {z_p50_delta_max_m}m) — likely camera-frame "
            "pts dumped as world / wrong cam2world (Studio frame bug)"
        )
    if problems:
        raise RuntimeError(
            "MapAnything ENU frame check failed (refusing product PLY): "
            + "; ".join(problems)
        )
    return stats


def fuse_predictions_to_ply(
    predictions: list[dict[str, Any]],
    dest_ply: Path,
    *,
    locked_poses: list[np.ndarray],
    max_points: int = 2_000_000,
    centroid_delta_max_m: float = DEFAULT_CENTROID_DELTA_MAX_M,
    z_p50_delta_max_m: float = DEFAULT_Z_P50_DELTA_MAX_M,
    skip_enu_check: bool = False,
) -> dict[str, Any]:
    """Fuse masked camera-frame geometry → ENU PLY with **our** cam2world.

    Hard rule: ``WORLD = R_c2w @ X_cam + C_enu`` using locked poses from
    align/lerp — never MapAnything predicted ``camera_poses`` for product.
    Prefer ``pts3d_cam`` / ``depth_z``; refuse raw ``pts3d`` as ENU.
    """
    if not locked_poses or len(locked_poses) != len(predictions):
        raise RuntimeError(
            f"fuse requires locked_poses matching predictions "
            f"(got {0 if not locked_poses else len(locked_poses)} poses, "
            f"{len(predictions)} preds)"
        )

    xyz_parts: list[np.ndarray] = []
    rgb_parts: list[np.ndarray] = []
    predicted_poses: list[np.ndarray] = []
    export_source = "pts3d_cam"

    for i, (pred, T_lock) in enumerate(zip(predictions, locked_poses, strict=True)):
        T = np.asarray(T_lock, dtype=np.float64)
        if T.shape != (4, 4):
            raise RuntimeError(f"view {i}: locked pose must be 4×4 cam2world, got {T.shape}")
        R = T[:3, :3]
        C = T[:3, 3]

        try:
            pts_hw = pts_cam_from_prediction(pred)
        except RuntimeError as exc:
            keys = sorted(str(k) for k in pred.keys())
            raise RuntimeError(
                f"view {i}: cannot build camera-frame points (keys={keys}): {exc}"
            ) from exc

        if pred.get("pts3d_cam") is None and pred.get("depth_z") is not None:
            export_source = "depth_z"

        flat_cam = pts_hw.reshape(-1, 3)
        mask = pred.get("mask")
        mask_flat: np.ndarray | None = None
        if mask is not None:
            mask_flat = _as_numpy(mask).reshape(-1).astype(bool)
            if mask_flat.shape[0] == flat_cam.shape[0]:
                flat_cam = flat_cam[mask_flat]
            else:
                mask_flat = None

        if flat_cam.shape[0] == 0:
            cam = pred.get("camera_poses")
            if cam is not None:
                cam_np = _as_numpy(cam)
                if cam_np.ndim == 3:
                    cam_np = cam_np[0]
                predicted_poses.append(cam_np.astype(np.float64))
            continue

        xyz_enu = cam_points_to_enu(flat_cam, R, C)
        colors = _colors_for_pred(pred, mask_flat, len(xyz_enu))
        xyz_parts.append(xyz_enu)
        rgb_parts.append(colors[: len(xyz_enu)])

        # Capture predicted pose only for drift log — never write as authority.
        cam = pred.get("camera_poses")
        if cam is not None:
            cam_np = _as_numpy(cam)
            if cam_np.ndim == 3:
                cam_np = cam_np[0]
            predicted_poses.append(cam_np.astype(np.float64))

    if not xyz_parts:
        raise RuntimeError("MapAnything produced no camera-frame points under mask")

    xyz = np.concatenate(xyz_parts, axis=0)
    rgb = np.concatenate(rgb_parts, axis=0)
    if len(xyz) > max_points:
        idx = np.linspace(0, len(xyz) - 1, max_points).astype(np.int64)
        xyz, rgb = xyz[idx], rgb[idx]

    centers = np.stack(
        [np.asarray(T, dtype=np.float64)[:3, 3] for T in locked_poses], axis=0
    )
    enu_stats: dict[str, float] | None = None
    if not skip_enu_check:
        enu_stats = assert_cloud_near_cameras(
            xyz,
            centers,
            centroid_delta_max_m=centroid_delta_max_m,
            z_p50_delta_max_m=z_p50_delta_max_m,
        )
        log.info(
            "MapAnything ENU check OK: centroid|ΔC|=%.2fm z_p50=%.2f (cam_u_p50=%.2f)",
            enu_stats["centroid_delta_m"],
            enu_stats["cloud_z_p50"],
            enu_stats["cam_u_p50"],
        )

    dest_ply = Path(dest_ply)
    dest_ply.parent.mkdir(parents=True, exist_ok=True)
    # write_ply expects BGR-ish channel order historically (r,g,b swapped in writer)
    # Pass OpenCV-style BGR so writer emits correct RGB labels.
    bgr = rgb[:, ::-1].copy()
    write_ply(dest_ply, xyz, bgr)

    drift = None
    if predicted_poses:
        drift = _median_pose_drift_m(list(locked_poses), predicted_poses)
        if drift is not None:
            log.info(
                "MapAnything predicted-pose median |ΔC|=%.3f m (discarded; ENU lock kept)",
                drift,
            )

    return {
        "path": str(dest_ply),
        "points": int(len(xyz)),
        "predicted_pose_median_delta_m": drift,
        "predicted_poses_discarded": True,
        "export_frame": "ENU",
        "export_source": export_source,
        "transform": "R_c2w @ pts_cam + C_enu (locked poses)",
        "enu_check": enu_stats,
    }


def _as_numpy(t: Any) -> np.ndarray:
    if isinstance(t, np.ndarray):
        return t
    if hasattr(t, "detach"):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def run_mapanything_on_bundle(
    bundle_dir: Path,
    *,
    out_ply: Path,
    apache: bool = True,
    device: str | None = None,
    minibatch_size: int = 1,
    amp_dtype: str | None = None,
) -> dict[str, Any]:
    """Path B: load pose-locked bundle → MapAnything infer → PLY."""
    require_cuda_for_densify()
    import torch
    from mapanything.models import MapAnything
    from mapanything.utils.image import preprocess_inputs

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device != "cuda":
        raise RuntimeError("MapAnything densify requires device=cuda")

    views, manifest = load_bundle_views_for_infer(bundle_dir, device=device)
    # Hard rule: every view must carry poses; first view especially.
    if any("camera_poses" not in v for v in views):
        raise RuntimeError("refusing infer: missing camera_poses on a view")
    if "camera_poses" not in views[0]:
        raise RuntimeError("refusing infer: reference view[0] lacks camera_poses")

    amp = amp_dtype or default_amp_dtype()
    model_id = DEFAULT_HF_MODEL if apache else "facebook/map-anything"
    log.info(
        "MapAnything: loading %s on %s (%s views, amp_dtype=%s)",
        model_id,
        device,
        len(views),
        amp,
    )
    model = MapAnything.from_pretrained(model_id).to(device)

    processed = preprocess_inputs(views)
    predictions = model.infer(
        processed,
        memory_efficient_inference=True,
        minibatch_size=int(minibatch_size),
        use_amp=True,
        amp_dtype=amp,
        apply_mask=True,
        mask_edges=True,
        apply_confidence_mask=True,
        confidence_percentile=10,
        ignore_calibration_inputs=False,
        ignore_depth_inputs=True,
        ignore_pose_inputs=False,  # KEEP POSES ON
        ignore_depth_scale_inputs=False,
        ignore_pose_scale_inputs=False,
    )

    locked = [np.asarray(v["camera_poses"], dtype=np.float64) for v in views]
    meta = fuse_predictions_to_ply(predictions, out_ply, locked_poses=locked)
    meta["backend"] = "mapanything"
    meta["hf_model"] = model_id
    meta["n_views"] = len(views)
    meta["pose_lock"] = True
    meta["ignore_pose_inputs"] = False
    meta["amp_dtype"] = amp
    meta["bundle"] = str(bundle_dir)
    meta["manifest_format"] = manifest.get("format")
    return meta


def run_mapanything_colmap_demo(
    run_root: Path,
    *,
    stride: int = DEFAULT_STRIDE,
    apache: bool = True,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Path A: shell out to upstream COLMAP demo (poses kept)."""
    require_cuda_for_densify()
    resolved = resolve_posed_colmap(run_root)
    if resolved is None:
        raise RuntimeError(
            "no posed COLMAP under recon/colmap/sparse_posed + images. "
            "Run: ps1hood reconstruct <run> --backend colmap_posed "
            "or use Path B (export mapanything-bundle)."
        )
    _images, sparse = resolved
    demo = find_mapanything_colmap_demo()
    if demo is None:
        raise RuntimeError(
            "demo_inference_on_colmap_outputs.py not found. "
            "Clone map-anything and set MAPANYTHING_ROOT, or use Path B bundle.\n"
            + INSTALL_HINT
        )
    out = Path(out_dir) if out_dir else run_root.resolve() / "mapanything" / "colmap_out"
    out.mkdir(parents=True, exist_ok=True)
    # Prefer sparse model dir that contains cameras/images/points3D.
    colmap_path = sparse
    argv = colmap_demo_argv(
        demo,
        colmap_path=colmap_path,
        output_directory=out,
        stride=stride,
        apache=apache,
        save_glb=True,
        save_colmap=True,
    )
    log.info("MapAnything COLMAP demo: %s", " ".join(argv))
    subprocess.run(argv, check=True)
    # Find a PLY / points under output
    ply_candidates = list(out.rglob("*.ply"))
    ply = next((p for p in ply_candidates if p.stat().st_size > 64), None)
    if ply is None:
        raise RuntimeError(f"MapAnything COLMAP demo wrote no PLY under {out}")
    imported = import_dense_ply_to_recon(ply, run_root)
    return {
        "backend": "mapanything",
        "path": imported["cloud_mapanything"],
        "points": imported["points"],
        "colmap_out": str(out),
        "seed": str(sparse),
        "pose_lock": True,
        "ignore_pose_inputs": False,
        "apache": apache,
        "path_kind": "colmap_demo",
    }


def run_mapanything_densify(
    run_root: Path,
    project: Any | None = None,
    *,
    stride: int = DEFAULT_STRIDE,
    max_views: int | None = DEFAULT_MAX_VIEWS,
    apache: bool = True,
    prefer_colmap: bool = True,
    also_studio_cloud: bool = True,
) -> dict[str, Any]:
    """Densify with MapAnything (CUDA). Exports bundle always; runs when possible.

    Prefer Path A (posed COLMAP demo) when sparse exists and demo script is
    found; else Path B (align/interp bundle + Python API).
    """
    root = Path(run_root).resolve()
    ma_dir = root / "mapanything"
    ma_dir.mkdir(parents=True, exist_ok=True)

    # Always ensure a pose-locked bundle exists for cloud hand-off / tests.
    if project is not None:
        frames = resolve_mapanything_frames(
            project, stride=1, max_views=None, prefer_interp=True
        )
        bundle_meta = export_mapanything_bundle(
            frames,
            ma_dir / "bundle",
            stride=stride,
            max_views=max_views,
            run_name=getattr(project, "name", root.name),
        )
    else:
        bundle_meta = None

    require_cuda_for_densify()

    if prefer_colmap and resolve_posed_colmap(root) is not None:
        demo = find_mapanything_colmap_demo()
        if demo is not None:
            try:
                return run_mapanything_colmap_demo(
                    root, stride=stride, apache=apache, out_dir=ma_dir / "colmap_out"
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("Path A COLMAP demo failed (%s); trying Path B bundle", exc)

    if bundle_meta is None:
        raise RuntimeError(
            "MapAnything densify needs a Project for Path B (align frames) "
            "or posed COLMAP + upstream demo for Path A.\n" + INSTALL_HINT
        )

    out_ply = ma_dir / "cloud.ply"
    meta = run_mapanything_on_bundle(
        Path(bundle_meta["path"]),
        out_ply=out_ply,
        apache=apache,
        minibatch_size=1,
    )
    imported = import_dense_ply_to_recon(
        out_ply, root, also_studio_cloud=also_studio_cloud
    )
    meta["cloud_mapanything"] = imported["cloud_mapanything"]
    meta["studio_cloud"] = imported["cloud"]
    meta["path_kind"] = "bundle_infer"
    meta["bundle_export"] = bundle_meta
    return meta
