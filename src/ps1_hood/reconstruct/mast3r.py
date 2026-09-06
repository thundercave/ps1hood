"""MASt3R as **matcher only** → posed COLMAP ``point_triangulator``.

Product path (TC3): cross-pano pair list → MASt3R pairwise matches → COLMAP DB
→ ``point_triangulator`` with **fixed ENU** priors. Never free-pose sparse GA /
GLOMAP / ``--ignore_pose`` as the hero.

Patterns follow naver/mast3r ``kapture_mast3r_mapping.py`` (has_pose →
triangulator) without requiring kapture: we bootstrap the COLMAP DB ourselves
and keep ``write_known_pose_model`` ENU extrinsics.

Optional / experimental: ``run_mast3r_free_pose`` keeps the old sparse-GA toy
for research — not wired as the CLI default.

Torch + MASt3R weights are *not* default dependencies. Fail loud without the
package or a CUDA/HIP GPU.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from ps1_hood.reconstruct.colmap import (
    bootstrap_posed_database,
    cross_pano_forward_pairs,
    export_colmap_images,
    import_keypoints_and_matches,
    run_colmap_posed,
    write_cross_pano_match_list,
    write_known_pose_model,
)
from ps1_hood.reconstruct.unproject import write_ply

log = logging.getLogger(__name__)

PairMatchFn = Callable[[Path, Path], tuple[np.ndarray, np.ndarray]]

INSTALL_HINT = """
MASt3R is optional and not vendored (CC BY-NC-SA 4.0 weights/code).
Install on a CUDA (or working ROCm) machine:

  git clone --recursive https://github.com/naver/mast3r
  cd mast3r
  # install torch matching your CUDA/ROCm, then:
  pip install -e .
  # download checkpoint weights (see mast3r README)

Then re-run matcher→posed triangulator (ENU locked):

  ps1hood reconstruct <run> --backend mast3r
  # equivalent: --backend colmap_posed --matcher mast3r

Needs a GPU. CPU-only is unsupported. Do NOT use free-pose MASt3R / GLOMAP
as the product path — poses stay in align ENU.

Docs: docs/mast3r-matcher.md · upstream toys: kapture_mast3r_mapping.py
""".strip()

DEFAULT_HF_WEIGHTS = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
DEFAULT_CONF_THR = 1.001
DEFAULT_MATCH_SIZE = 512
# SV forward pairs often yield short tracks; keep 2-view (posed triangulator
# already allows them). Raise toward 3+ when the graph densifies.
DEFAULT_MIN_TRACK_LEN = 2


def _try_import_mast3r() -> bool:
    try:
        import dust3r  # noqa: F401
        import mast3r  # noqa: F401

        return True
    except ImportError:
        return False


def which_mast3r() -> dict[str, Any] | None:
    """Return import/runtime info if mast3r+dust3r importable, else None."""
    if not _try_import_mast3r():
        return None
    info: dict[str, Any] = {"package": True}
    try:
        import torch

        info["torch"] = True
        info["cuda"] = bool(torch.cuda.is_available())
        hip = getattr(getattr(torch, "version", None), "hip", None)
        info["hip"] = bool(hip)
    except ImportError:
        info["torch"] = False
        info["cuda"] = False
        info["hip"] = False
    return info


def pick_mast3r_device() -> str:
    """Prefer CUDA/HIP device; fail loud on CPU-only for the product path."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "MASt3R matcher needs PyTorch + CUDA/HIP GPU.\n" + INSTALL_HINT
        ) from exc
    if torch.cuda.is_available():
        # ROCm builds also expose the HIP device via the torch.cuda API.
        return "cuda"
    raise RuntimeError(
        "MASt3R matcher requires a CUDA or ROCm GPU (torch.cuda.is_available() "
        "is False). Export keyframes and run on cloud CUDA, or use "
        "--backend colmap_posed --matcher sift on CPU.\n" + INSTALL_HINT
    )


def require_mast3r_runtime() -> str:
    """Fail loud unless mast3r/dust3r import and a GPU device exist."""
    if not _try_import_mast3r():
        raise RuntimeError("MASt3R / DUSt3R not installed.\n" + INSTALL_HINT)
    return pick_mast3r_device()


def _find_weights() -> str | None:
    """Look for a local checkpoint if the user already downloaded one."""
    candidates = [
        Path.home() / ".cache" / "torch" / "hub" / "checkpoints",
        Path.home() / ".cache" / "huggingface" / "hub",
        Path("/workspace/mast3r"),
        Path.home() / "mast3r",
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


def _load_mast3r_model(device: str) -> Any:
    from mast3r.model import AsymmetricMASt3R  # type: ignore[import-untyped]

    weights = _find_weights()
    log.info("MASt3R matcher: loading model on %s (%s)", device, weights or "hub/default")
    if weights:
        model = AsymmetricMASt3R.from_pretrained(weights).to(device)
    else:
        model = AsymmetricMASt3R.from_pretrained(DEFAULT_HF_WEIGHTS).to(device)
    model.eval()
    return model


def _scale_xy_to_original(
    xy: np.ndarray, *, resized_hw: tuple[int, int], orig_wh: tuple[int, int]
) -> np.ndarray:
    """Map matches from MASt3R resize (H,W) back to original (W,H) pixels."""
    rh, rw = resized_hw
    ow, oh = orig_wh
    out = np.asarray(xy, dtype=np.float64).reshape(-1, 2).copy()
    if rw <= 0 or rh <= 0:
        return out.astype(np.float32)
    out[:, 0] = out[:, 0] * (float(ow) / float(rw))
    out[:, 1] = out[:, 1] * (float(oh) / float(rh))
    out[:, 0] = np.clip(out[:, 0], 0.0, max(ow - 0.01, 0.0))
    out[:, 1] = np.clip(out[:, 1], 0.0, max(oh - 0.01, 0.0))
    return out.astype(np.float32)


def match_image_pair_xy(
    model: Any,
    path_a: Path,
    path_b: Path,
    *,
    device: str,
    conf_thr: float = DEFAULT_CONF_THR,
    size: int = DEFAULT_MATCH_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Run MASt3R on one image pair → (xy_a, xy_b) in **original** pixel coords."""
    from dust3r.inference import inference  # type: ignore[import-untyped]
    from dust3r.utils.image import load_images  # type: ignore[import-untyped]
    from mast3r.fast_nn import extract_correspondences_nonsym  # type: ignore[import-untyped]
    from PIL import Image

    path_a, path_b = Path(path_a), Path(path_b)
    with Image.open(path_a) as im:
        ow_a, oh_a = im.size
    with Image.open(path_b) as im:
        ow_b, oh_b = im.size

    imgs = load_images([str(path_a), str(path_b)], size=int(size))
    if len(imgs) < 2:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
        )
    pairs = [(imgs[0], imgs[1])]
    output = inference(pairs, model, device, batch_size=1, verbose=False)
    pred1, pred2 = output["pred1"], output["pred2"]

    if "desc" not in pred1:
        raise RuntimeError(
            "MASt3R forward pass returned no descriptors — use AsymmetricMASt3R weights"
        )

    descs = [pred1["desc"][0], pred2["desc"][0]]
    confidences = [pred1["desc_conf"][0], pred2["desc_conf"][0]]
    corres = extract_correspondences_nonsym(
        descs[0],
        descs[1],
        confidences[0],
        confidences[1],
        device=device,
        subsample=8,
        pixel_tol=0,
    )
    conf = corres[2]
    mask = conf >= float(conf_thr)
    matches_a = corres[0][mask].detach().cpu().numpy()
    matches_b = corres[1][mask].detach().cpu().numpy()
    if len(matches_a) == 0:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
        )

    shape_a = np.asarray(imgs[0]["true_shape"]).reshape(-1)
    shape_b = np.asarray(imgs[1]["true_shape"]).reshape(-1)
    rh_a, rw_a = int(shape_a[0]), int(shape_a[1])
    rh_b, rw_b = int(shape_b[0]), int(shape_b[1])
    xy_a = _scale_xy_to_original(matches_a, resized_hw=(rh_a, rw_a), orig_wh=(ow_a, oh_a))
    xy_b = _scale_xy_to_original(matches_b, resized_hw=(rh_b, rw_b), orig_wh=(ow_b, oh_b))
    return xy_a, xy_b


def _quantize_key(xy: np.ndarray, quant: float = 0.5) -> tuple[int, int]:
    return (int(round(float(xy[0]) / quant)), int(round(float(xy[1]) / quant)))


def aggregate_pair_matches(
    pair_xy: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
    n_images: int,
    *,
    min_track_len: int = DEFAULT_MIN_TRACK_LEN,
    quant: float = 0.5,
) -> tuple[dict[int, np.ndarray], dict[tuple[int, int], np.ndarray]]:
    """Union keypoints across pairs; drop tracks shorter than ``min_track_len``.

    Returns COLMAP-ready ``{image_id(1-based): Nx2}`` and
    ``{(id1,id2): Mx2 indices}`` with id1 < id2.
    """
    # per-image: quant key → list of (x,y) samples + provisional track ids later
    buckets: list[dict[tuple[int, int], list[np.ndarray]]] = [
        {} for _ in range(n_images)
    ]
    # edges: (i, key_i, j, key_j)
    edges: list[tuple[int, tuple[int, int], int, tuple[int, int]]] = []

    for (i, j), (xy_i, xy_j) in pair_xy.items():
        if i < 0 or j < 0 or i >= n_images or j >= n_images or i == j:
            continue
        a = np.asarray(xy_i, dtype=np.float32).reshape(-1, 2)
        b = np.asarray(xy_j, dtype=np.float32).reshape(-1, 2)
        n = min(len(a), len(b))
        for k in range(n):
            ki = _quantize_key(a[k], quant)
            kj = _quantize_key(b[k], quant)
            buckets[i].setdefault(ki, []).append(a[k])
            buckets[j].setdefault(kj, []).append(b[k])
            edges.append((i, ki, j, kj))

    # Build disjoint-set over (image, quant_key) nodes that appear in edges.
    parent: dict[tuple[int, tuple[int, int]], tuple[int, tuple[int, int]]] = {}

    def find(x: tuple[int, tuple[int, int]]) -> tuple[int, tuple[int, int]]:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: tuple[int, tuple[int, int]], b: tuple[int, tuple[int, int]]) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, ki, j, kj in edges:
        union((i, ki), (j, kj))

    # track root → set of (image, key)
    tracks: dict[tuple[int, tuple[int, int]], set[tuple[int, tuple[int, int]]]] = {}
    for i, bucket in enumerate(buckets):
        for key in bucket:
            node = (i, key)
            root = find(node)
            tracks.setdefault(root, set()).add(node)

    valid_nodes: set[tuple[int, tuple[int, int]]] = set()
    for members in tracks.values():
        images_in_track = {im for im, _ in members}
        if len(images_in_track) >= int(min_track_len):
            valid_nodes.update(members)

    keypoints: dict[int, np.ndarray] = {}
    index_of: dict[tuple[int, tuple[int, int]], int] = {}
    for i, bucket in enumerate(buckets):
        kept: list[np.ndarray] = []
        for key, samples in bucket.items():
            if (i, key) not in valid_nodes:
                continue
            mean = np.mean(np.stack(samples, axis=0), axis=0).astype(np.float32)
            index_of[(i, key)] = len(kept)
            kept.append(mean)
        if kept:
            keypoints[i + 1] = np.stack(kept, axis=0)  # COLMAP image_id 1-based
        else:
            keypoints[i + 1] = np.zeros((0, 2), dtype=np.float32)

    pair_matches: dict[tuple[int, int], np.ndarray] = {}
    for i, ki, j, kj in edges:
        if (i, ki) not in index_of or (j, kj) not in index_of:
            continue
        id1, id2 = i + 1, j + 1
        m1, m2 = index_of[(i, ki)], index_of[(j, kj)]
        if id1 > id2:
            id1, id2 = id2, id1
            m1, m2 = m2, m1
        pair_matches.setdefault((id1, id2), []).append((m1, m2))  # type: ignore[arg-type]

    out_matches: dict[tuple[int, int], np.ndarray] = {}
    for key, rows in pair_matches.items():
        arr = np.unique(np.asarray(rows, dtype=np.uint32).reshape(-1, 2), axis=0)
        if len(arr):
            out_matches[key] = arr
    return keypoints, out_matches


def match_pairs_for_images(
    image_paths: list[Path],
    pair_indices: Iterable[tuple[int, int]],
    *,
    device: str | None = None,
    conf_thr: float = DEFAULT_CONF_THR,
    size: int = DEFAULT_MATCH_SIZE,
    match_fn: PairMatchFn | None = None,
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    """Match listed index pairs. ``match_fn`` injects mocks (no GPU)."""
    paths = [Path(p) for p in image_paths]
    pairs = [(int(i), int(j)) for i, j in pair_indices]
    out: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    if match_fn is not None:
        for i, j in pairs:
            xy_i, xy_j = match_fn(paths[i], paths[j])
            out[(i, j)] = (
                np.asarray(xy_i, dtype=np.float32).reshape(-1, 2),
                np.asarray(xy_j, dtype=np.float32).reshape(-1, 2),
            )
        return out

    device = device or require_mast3r_runtime()
    model = _load_mast3r_model(device)
    for i, j in pairs:
        log.info("MASt3R match %s ↔ %s", paths[i].name, paths[j].name)
        xy_i, xy_j = match_image_pair_xy(
            model, paths[i], paths[j], device=device, conf_thr=conf_thr, size=size
        )
        out[(i, j)] = (xy_i, xy_j)
        log.info("  → %s correspondences", len(xy_i))
    return out


def fill_database_with_mast3r_matches(
    database: Path,
    images_dir: Path,
    frames: list[dict[str, Any]],
    image_names: list[str],
    *,
    pair_indices: list[tuple[int, int]] | None = None,
    conf_thr: float = DEFAULT_CONF_THR,
    size: int = DEFAULT_MATCH_SIZE,
    min_track_len: int = DEFAULT_MIN_TRACK_LEN,
    match_fn: PairMatchFn | None = None,
    skip_geometric_verification: bool = True,
) -> dict[str, Any]:
    """Run MASt3R on cross-pano pairs and import keypoints/matches into ``database``.

    Database must already contain cameras/images (see ``bootstrap_posed_database``).
    """
    if len(frames) != len(image_names):
        raise ValueError("frames / image_names mismatch")
    pairs = pair_indices or cross_pano_forward_pairs(frames, n_forward=3)
    if not pairs:
        raise RuntimeError("no cross-pano pairs for MASt3R matcher")

    image_paths = [Path(images_dir) / name for name in image_names]
    missing = [p for p in image_paths if not p.is_file()]
    if missing:
        raise RuntimeError(f"MASt3R matcher missing images e.g. {missing[0]}")

    # Export-only / fail-loud before GPU work when package missing and no mock.
    if match_fn is None:
        require_mast3r_runtime()

    pair_xy = match_pairs_for_images(
        image_paths,
        pairs,
        conf_thr=conf_thr,
        size=size,
        match_fn=match_fn,
    )
    keypoints, pair_matches = aggregate_pair_matches(
        pair_xy, len(image_names), min_track_len=min_track_len
    )
    n_kp = sum(len(v) for v in keypoints.values())
    if n_kp == 0 or not pair_matches:
        raise RuntimeError(
            "MASt3R matcher produced no usable tracks "
            f"(keypoints={n_kp}, pairs_with_matches={len(pair_matches)}). "
            "Try denser drive spacing / lower conf_thr, or fall back to "
            "--matcher sift."
        )
    n_imported = import_keypoints_and_matches(
        database,
        keypoints,
        pair_matches,
        skip_geometric_verification=skip_geometric_verification,
    )
    log.info(
        "MASt3R → COLMAP DB: %s keypoints across %s images, %s pair geometries",
        n_kp,
        len(image_names),
        n_imported,
    )
    return {
        "n_keypoints": int(n_kp),
        "n_pair_geometries": int(n_imported),
        "n_pairs_requested": len(pairs),
        "min_track_len": int(min_track_len),
        "matcher": "mast3r",
        "pose_lock": True,
    }


def run_mast3r_matcher_posed(
    frames: list[dict[str, Any]],
    recon_dir: Path,
    *,
    cloud_name: str = "cloud_photo.ply",
    n_forward: int = 3,
    match_fn: PairMatchFn | None = None,
    min_points: int = 200,
) -> dict[str, Any]:
    """Full TC3 path: export → ENU prior → MASt3R matches → point_triangulator."""
    if len(frames) < 2:
        raise RuntimeError("MASt3R matcher needs at least 2 keyframes with images")

    recon_dir = Path(recon_dir)
    recon_dir.mkdir(parents=True, exist_ok=True)
    ws, names = export_colmap_images(frames, recon_dir / "colmap")

    # Always write priors + pair list for debugging / cloud hand-off even if
    # mast3r is missing (fail after export).
    sparse_prior = ws / "sparse_prior"
    if sparse_prior.exists():
        import shutil

        shutil.rmtree(sparse_prior)
    write_known_pose_model(frames, names, sparse_prior)
    match_list = ws / "cross_pano_pairs.txt"
    write_cross_pano_match_list(frames, names, match_list, n_forward=n_forward)

    ply_out = recon_dir / cloud_name

    # Injected match_fn (unit tests) bypasses the live package/GPU requirement.
    if match_fn is not None:
        return _run_posed_with_match_fn(
            ws,
            frames,
            names,
            ply_out=ply_out,
            n_forward=n_forward,
            match_fn=match_fn,
            min_points=min_points,
        )

    if not _try_import_mast3r():
        raise RuntimeError(
            f"MASt3R / DUSt3R not installed. Keyframes + ENU priors exported under {ws}.\n"
            + INSTALL_HINT
        )
    # Fail loud on CPU before COLMAP work.
    require_mast3r_runtime()

    ply_path = run_colmap_posed(
        ws,
        frames,
        names,
        ply_out=ply_out,
        n_forward=n_forward,
        matcher="mast3r",
        min_points=min_points,
    )

    cloud_ply = recon_dir / "cloud.ply"
    import shutil

    shutil.copy2(ply_path, cloud_ply)
    n_pts = _ply_vertex_count(cloud_ply)
    return {
        "path": str(cloud_ply),
        "photo_ply": str(ply_path),
        "points": n_pts,
        "backend": "mast3r",
        "matcher": "mast3r",
        "pose_lock": True,
        "source": "mast3r_matcher_posed",
    }


def _run_posed_with_match_fn(
    workspace: Path,
    frames: list[dict[str, Any]],
    image_names: list[str],
    *,
    ply_out: Path,
    n_forward: int,
    match_fn: PairMatchFn,
    min_points: int,
) -> dict[str, Any]:
    """Test/helper path: injected match_fn → DB → point_triangulator."""
    import shutil

    from ps1_hood.reconstruct.colmap import (
        _colmap_bin,
        _convert_model_to_ply,
        _count_model_points,
        _run,
        filter_frames_registered_in_matches,
        find_sparse_model,
        point_triangulator_argv,
    )

    images = workspace / "images"
    db = workspace / "database.db"
    sparse_prior = workspace / "sparse_prior"
    if sparse_prior.exists():
        shutil.rmtree(sparse_prior)
    write_known_pose_model(frames, image_names, sparse_prior)
    write_cross_pano_match_list(
        frames, image_names, workspace / "cross_pano_pairs.txt", n_forward=n_forward
    )
    bootstrap_posed_database(db, frames, image_names)
    fill_database_with_mast3r_matches(
        db,
        images,
        frames,
        image_names,
        pair_indices=cross_pano_forward_pairs(frames, n_forward=n_forward),
        match_fn=match_fn,
    )
    frames, image_names = filter_frames_registered_in_matches(
        frames, image_names, db, model_dir=sparse_prior
    )
    sparse_out = workspace / "sparse_posed"
    if sparse_out.exists():
        shutil.rmtree(sparse_out)
    sparse_out.mkdir(parents=True, exist_ok=True)
    colmap = _colmap_bin()
    _run(point_triangulator_argv(colmap, db, images, sparse_prior, sparse_out))
    model = find_sparse_model(sparse_out) or sparse_out
    _convert_model_to_ply(colmap, model, ply_out)
    n = _count_model_points(model)
    cloud_ply = ply_out.parent / "cloud.ply"
    shutil.copy2(ply_out, cloud_ply)
    if n < min_points:
        raise RuntimeError(
            f"MASt3R matcher posed triangulator too thin ({n} points; need ≥{min_points})"
        )
    return {
        "path": str(cloud_ply),
        "photo_ply": str(ply_out),
        "points": int(n),
        "backend": "mast3r",
        "matcher": "mast3r",
        "pose_lock": True,
        "source": "mast3r_matcher_posed",
    }


def run_mast3r(
    frames: list[dict[str, Any]],
    recon_dir: Path,
    *,
    cloud_name: str = "cloud.ply",
) -> dict[str, Any]:
    """CLI entry: MASt3R **matcher** → posed ``point_triangulator`` (ENU locked).

    Exports under ``recon/colmap/`` (images + sparse_prior + matches). Raises
    ``RuntimeError`` with install steps if mast3r is missing / no GPU.
    """
    # Prefer cloud_photo.ply naming used by colmap_posed; also write cloud.ply.
    photo = "cloud_photo.ply" if cloud_name == "cloud.ply" else cloud_name
    meta = run_mast3r_matcher_posed(frames, recon_dir, cloud_name=photo)
    # Ensure requested cloud_name exists when caller asked for cloud.ply.
    dest = Path(recon_dir) / cloud_name
    src = Path(meta["path"])
    if dest.resolve() != src.resolve() and src.is_file():
        import shutil

        shutil.copy2(src, dest)
        meta["path"] = str(dest)
    return meta


# ----- experimental free-pose (NOT product) ---------------------------------


def run_mast3r_free_pose(
    frames: list[dict[str, Any]],
    recon_dir: Path,
    *,
    cloud_name: str = "cloud_mast3r_free.ply",
) -> dict[str, Any]:
    """Experimental sparse global alignment — **not** the product path.

    Invents extrinsics (ghost risk). Prefer ``run_mast3r`` / ``--matcher mast3r``.
    """
    log.warning(
        "run_mast3r_free_pose: free-pose MASt3R alignment is experimental and "
        "NOT the product path — ENU poses are ignored"
    )
    if len(frames) < 2:
        raise RuntimeError("MASt3R needs at least 2 keyframes with images")

    input_ws, _names = export_colmap_images(frames, recon_dir / "mast3r_input")
    images_dir = input_ws / "images"
    device = require_mast3r_runtime()
    cloud_path = Path(recon_dir) / cloud_name
    meta = _run_sparse_ga(images_dir, cloud_path, device=device)
    meta["pose_lock"] = False
    meta["product_path"] = False
    return meta


def _run_sparse_ga(images_dir: Path, dest_ply: Path, *, device: str) -> dict[str, Any]:
    """Best-effort MASt3R-SfM sparse alignment → PLY (research only)."""
    image_paths = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if len(image_paths) < 2:
        raise RuntimeError(f"need ≥2 images in {images_dir}")

    from dust3r.utils.image import load_images  # type: ignore[import-untyped]

    model = _load_mast3r_model(device)
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
        raise RuntimeError("MASt3R free-pose produced an empty point cloud")

    dest_ply.parent.mkdir(parents=True, exist_ok=True)
    if rgb is None:
        rgb = np.full((len(xyz), 3), 180, dtype=np.uint8)
    write_ply(dest_ply, np.asarray(xyz, dtype=np.float64), np.asarray(rgb, dtype=np.uint8))
    log.info("MASt3R free-pose wrote %s points → %s", len(xyz), dest_ply)
    return {
        "path": str(dest_ply),
        "points": int(len(xyz)),
        "backend": "mast3r_free_pose",
        "pose_lock": False,
    }


def _scene_points(scene: Any) -> tuple[Any, Any]:
    """Extract Nx3 xyz (+ optional RGB) from a MASt3R/DUSt3R scene object."""
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
        if isinstance(pts, (list, tuple)):
            for p in pts:
                arr = _to_numpy(p).reshape(-1, 3)
                xyz_list.append(arr)
        else:
            xyz_list.append(_to_numpy(pts).reshape(-1, 3))
        xyz = np.concatenate(xyz_list, axis=0) if xyz_list else None
        cols = getattr(scene, "imgs", None) or getattr(scene, "rgb", None)
        if cols is not None and xyz is not None:
            try:
                for im in cols:
                    arr = _to_numpy(im)
                    if arr.ndim == 3 and arr.shape[-1] >= 3:
                        rgb_list.append(
                            (
                                arr[..., :3].reshape(-1, 3)
                                * (255.0 if arr.max() <= 1.5 else 1.0)
                            ).astype(np.uint8)
                        )
                if rgb_list and sum(len(r) for r in rgb_list) == len(xyz):
                    return xyz, np.concatenate(rgb_list, axis=0)
            except Exception:  # noqa: BLE001
                pass
        return xyz, None

    for attr in ("pts3d", "points", "xyz"):
        if hasattr(scene, attr):
            return _to_numpy(getattr(scene, attr)).reshape(-1, 3), None
    raise RuntimeError("could not extract points from MASt3R scene")


def _to_numpy(x: Any) -> Any:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _ply_vertex_count(path: Path) -> int:
    with Path(path).open("rb") as fh:
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
