"""De-ghost chunked MapAnything PLYs under locked ENU.

Sacred: locked ENU only — never ICP/Umeyama between chunks, never invent
poses, never average ghost XYZ in a voxel. Keep-one / hybrid append only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree

from ps1_hood.reconstruct.facades import _read_ply_xyz
from ps1_hood.reconstruct.unproject import write_ply

# Diagnose verdict thresholds (metres), matching overnight fusion policy.
FUSION_MEDIAN_MAX_M = 0.12
STOP_TRANSFORM_MEDIAN_M = 0.20


@dataclass(frozen=True)
class PairDiagnose:
    """NN(A→B) stats on the overlap AABB subsample."""

    path_a: Path
    path_b: Path
    n_a: int
    n_b: int
    n_overlap_a: int
    n_queried: int
    median_m: float
    p90_m: float
    verdict: str


def resolve_chunk_paths(patterns: Sequence[str | Path]) -> list[Path]:
    """Expand globs / paths into an ordered list of existing PLY files.

    Glob matches are sorted; distinct positional args keep caller order.
    """
    out: list[Path] = []
    seen: set[Path] = set()
    for raw in patterns:
        p = Path(raw)
        if any(ch in str(raw) for ch in "*?["):
            if p.is_absolute():
                matches = sorted(p.parent.glob(p.name))
            else:
                matches = sorted(Path().glob(str(raw)))
        elif p.is_dir():
            matches = sorted(p.glob("*.ply"))
        else:
            matches = [p]
        for m in matches:
            rp = m.resolve() if m.exists() else m
            if rp in seen:
                continue
            if not m.exists():
                raise FileNotFoundError(f"chunk PLY not found: {m}")
            seen.add(rp)
            out.append(m)
    if not out:
        raise FileNotFoundError(f"no chunk PLYs matched: {list(patterns)}")
    return out


def _aabb(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return xyz.min(axis=0), xyz.max(axis=0)


def _overlap_mask(xyz: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.all((xyz >= lo) & (xyz <= hi), axis=1)


def _subsample(xyz: np.ndarray, max_points: int | None, seed: int | None) -> np.ndarray:
    if max_points is None or len(xyz) <= max_points:
        return xyz
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(xyz), size=int(max_points), replace=False)
    return xyz[idx]


def verdict_from_median(median_m: float) -> str:
    """One-line fusion gate from overlap NN median."""
    if not np.isfinite(median_m):
        return "STOP-no-overlap"
    if median_m >= STOP_TRANSFORM_MEDIAN_M:
        return "STOP-transform-bug"
    if median_m <= FUSION_MEDIAN_MAX_M:
        return "fusion-policy"
    return "caution-gap"


def diagnose_pair(
    xyz_a: np.ndarray,
    xyz_b: np.ndarray,
    *,
    max_points: int | None = None,
    seed: int | None = None,
    path_a: Path | None = None,
    path_b: Path | None = None,
) -> PairDiagnose:
    """Subsample A∩B AABB and report NN(A→B) median + p90."""
    pa = path_a or Path("A")
    pb = path_b or Path("B")
    n_a, n_b = int(len(xyz_a)), int(len(xyz_b))
    if n_a == 0 or n_b == 0:
        return PairDiagnose(
            path_a=pa,
            path_b=pb,
            n_a=n_a,
            n_b=n_b,
            n_overlap_a=0,
            n_queried=0,
            median_m=float("inf"),
            p90_m=float("inf"),
            verdict="STOP-no-overlap",
        )

    lo_a, hi_a = _aabb(xyz_a)
    lo_b, hi_b = _aabb(xyz_b)
    lo = np.maximum(lo_a, lo_b)
    hi = np.minimum(hi_a, hi_b)
    if np.any(lo > hi):
        return PairDiagnose(
            path_a=pa,
            path_b=pb,
            n_a=n_a,
            n_b=n_b,
            n_overlap_a=0,
            n_queried=0,
            median_m=float("inf"),
            p90_m=float("inf"),
            verdict="STOP-no-overlap",
        )

    mask = _overlap_mask(xyz_a, lo, hi)
    overlap_a = xyz_a[mask]
    n_overlap = int(len(overlap_a))
    if n_overlap == 0:
        return PairDiagnose(
            path_a=pa,
            path_b=pb,
            n_a=n_a,
            n_b=n_b,
            n_overlap_a=0,
            n_queried=0,
            median_m=float("inf"),
            p90_m=float("inf"),
            verdict="STOP-no-overlap",
        )

    query = _subsample(overlap_a, max_points, seed)
    tree = cKDTree(xyz_b)
    dists, _ = tree.query(query, k=1, workers=-1)
    dists = np.asarray(dists, dtype=np.float64)
    median_m = float(np.median(dists))
    p90_m = float(np.percentile(dists, 90))
    return PairDiagnose(
        path_a=pa,
        path_b=pb,
        n_a=n_a,
        n_b=n_b,
        n_overlap_a=n_overlap,
        n_queried=int(len(query)),
        median_m=median_m,
        p90_m=p90_m,
        verdict=verdict_from_median(median_m),
    )


def diagnose_paths(
    paths: Sequence[Path],
    *,
    max_points: int | None = None,
    seed: int | None = None,
) -> list[PairDiagnose]:
    """Diagnose each consecutive pair (or single pair if only two paths)."""
    if len(paths) < 2:
        raise ValueError("diagnose needs at least two chunk PLYs")
    clouds = [_read_ply_xyz(p) for p in paths]
    results: list[PairDiagnose] = []
    for i in range(len(paths) - 1):
        results.append(
            diagnose_pair(
                clouds[i],
                clouds[i + 1],
                max_points=max_points,
                seed=None if seed is None else seed + i,
                path_a=paths[i],
                path_b=paths[i + 1],
            )
        )
    return results


def keep_one(
    xyz: np.ndarray,
    radius: float,
    *,
    seed: int | None = None,
) -> np.ndarray:
    """Greedy radius keep-one via KDTree of kept points (no XYZ averaging)."""
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"expected (N,3) xyz, got {xyz.shape}")
    n = len(xyz)
    if n == 0:
        return xyz.copy()
    if radius < 0:
        raise ValueError(f"radius must be >= 0, got {radius}")
    if radius == 0:
        return xyz.copy()

    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    pts = xyz[order]
    kept_idx = [0]
    tree = cKDTree(pts[0:1])
    for i in range(1, n):
        d, _ = tree.query(pts[i], k=1, workers=1)
        if float(d) > radius:
            kept_idx.append(i)
            tree = cKDTree(pts[kept_idx])
    return pts[kept_idx]


def keep_one_paths(
    paths: Sequence[Path],
    radius: float,
    *,
    seed: int | None = None,
) -> np.ndarray:
    """Concat all chunk XYZ then greedy keep-one (no averaging)."""
    parts = [_read_ply_xyz(p) for p in paths]
    if not parts:
        return np.zeros((0, 3), dtype=np.float64)
    xyz = np.vstack(parts) if len(parts) > 1 else parts[0]
    return keep_one(xyz, radius, seed=seed)


def parse_chunk_nn(overrides: Iterable[str] | None) -> dict[str, float]:
    """Parse ``CHUNK=NN`` / ``CHUNK:NN`` tokens into stem→metres map."""
    out: dict[str, float] = {}
    if not overrides:
        return out
    for raw in overrides:
        tok = raw.strip()
        if not tok:
            continue
        if "=" in tok:
            key, val = tok.split("=", 1)
        elif ":" in tok:
            key, val = tok.split(":", 1)
        else:
            raise ValueError(f"bad --chunk-nn {raw!r}; expected CHUNK=NN")
        key = key.strip()
        # Allow basename or stem.
        stem = Path(key).stem if key.lower().endswith(".ply") else key
        out[stem] = float(val)
    return out


def _nn_for_chunk(path: Path, default_nn: float, overrides: Mapping[str, float]) -> float:
    stem = path.stem
    name = path.name
    if stem in overrides:
        return float(overrides[stem])
    if name in overrides:
        return float(overrides[name])
    # Prefix match e.g. chunk_04 in chunk_04_something.ply
    for key, val in overrides.items():
        if stem == key or stem.startswith(key) or key in stem:
            return float(val)
    return float(default_nn)


def hybrid_merge(
    base_xyz: np.ndarray,
    chunk_xyzs: Sequence[np.ndarray],
    *,
    nn_m: float = 0.10,
    chunk_nn: Sequence[float] | None = None,
) -> np.ndarray:
    """Append chunk points whose NN to the *current* merged cloud exceeds nn.

    ``chunk_nn[i]`` overrides ``nn_m`` for that chunk. No XYZ averaging.
    """
    merged = np.asarray(base_xyz, dtype=np.float64).reshape(-1, 3).copy()
    tree = cKDTree(merged) if len(merged) else None
    for i, chunk in enumerate(chunk_xyzs):
        pts = np.asarray(chunk, dtype=np.float64).reshape(-1, 3)
        if len(pts) == 0:
            continue
        thr = float(chunk_nn[i]) if chunk_nn is not None else float(nn_m)
        if tree is None or len(merged) == 0:
            merged = pts.copy()
            tree = cKDTree(merged)
            continue
        dists, _ = tree.query(pts, k=1, workers=-1)
        mask = np.asarray(dists, dtype=np.float64) > thr
        added = pts[mask]
        if len(added) == 0:
            continue
        merged = np.vstack([merged, added])
        tree = cKDTree(merged)
    return merged


def hybrid_merge_paths(
    base_path: Path,
    chunk_paths: Sequence[Path],
    *,
    nn_m: float = 0.10,
    chunk_nn_overrides: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Load base + chunks from disk and hybrid-append (locked ENU, no ICP)."""
    overrides = dict(chunk_nn_overrides or {})
    base = _read_ply_xyz(base_path)
    chunks = [_read_ply_xyz(p) for p in chunk_paths]
    thr = [_nn_for_chunk(p, nn_m, overrides) for p in chunk_paths]
    return hybrid_merge(base, chunks, nn_m=nn_m, chunk_nn=thr)


def write_xyz_ply(path: Path, xyz: np.ndarray, *, gray: int = 180) -> None:
    """Write XYZ PLY via unproject.write_ply with neutral gray RGB."""
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    rgb = np.full((len(xyz), 3), int(gray), dtype=np.uint8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_ply(path, xyz, rgb)


def format_diagnose_line(r: PairDiagnose) -> str:
    """Human one-liner for a diagnose pair result."""
    med = "inf" if not np.isfinite(r.median_m) else f"{r.median_m:.4f}"
    p90 = "inf" if not np.isfinite(r.p90_m) else f"{r.p90_m:.4f}"
    return (
        f"{r.path_a.name} → {r.path_b.name}: "
        f"median={med}m p90={p90}m "
        f"overlap_A={r.n_overlap_a} queried={r.n_queried} "
        f"verdict={r.verdict}"
    )
