"""Tests for locked-ENU MapAnything chunk merge (diagnose / keep-one / hybrid)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ps1_hood.reconstruct.facades import _read_ply_xyz
from ps1_hood.reconstruct.merge_ma_chunks import (
    diagnose_pair,
    diagnose_paths,
    hybrid_merge,
    hybrid_merge_paths,
    keep_one,
    keep_one_paths,
    parse_chunk_nn,
    write_xyz_ply,
)


def _write_ascii_xyz_ply(path: Path, xyz: np.ndarray) -> None:
    """Minimal ascii XYZ PLY (no RGB) for synthetic fixtures."""
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(xyz)}",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
    ]
    for x, y, z in xyz:
        lines.append(f"{x:.6f} {y:.6f} {z:.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def test_diagnose_identical_median_near_zero(tmp_path: Path) -> None:
    pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.5, 0.5, 0.1],
        ],
        dtype=np.float64,
    )
    a = tmp_path / "a.ply"
    b = tmp_path / "b.ply"
    _write_ascii_xyz_ply(a, pts)
    _write_ascii_xyz_ply(b, pts.copy())
    rows = diagnose_paths([a, b], seed=0)
    assert len(rows) == 1
    assert rows[0].median_m == 0.0
    assert rows[0].p90_m == 0.0
    assert rows[0].verdict == "fusion-policy"


def test_diagnose_offset_08m(tmp_path: Path) -> None:
    base = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.25, 0.75, 0.0],
            [0.75, 0.25, 0.0],
            [0.2, 0.2, 0.0],
        ],
        dtype=np.float64,
    )
    offset = base + np.array([0.08, 0.0, 0.0])
    a = tmp_path / "a.ply"
    b = tmp_path / "b.ply"
    _write_ascii_xyz_ply(a, base)
    _write_ascii_xyz_ply(b, offset)
    r = diagnose_pair(_read_ply_xyz(a), _read_ply_xyz(b), path_a=a, path_b=b)
    assert abs(r.median_m - 0.08) < 1e-6
    assert r.verdict == "fusion-policy"


def test_keep_one_near_and_far() -> None:
    near = np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]], dtype=np.float64)
    kept_near = keep_one(near, radius=0.12, seed=0)
    assert len(kept_near) == 1

    far = np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]], dtype=np.float64)
    kept_far = keep_one(far, radius=0.12, seed=0)
    assert len(kept_far) == 2


def test_keep_one_paths_roundtrip(tmp_path: Path) -> None:
    pts = np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    ply = tmp_path / "c.ply"
    _write_ascii_xyz_ply(ply, pts)
    out = keep_one_paths([ply], radius=0.12, seed=1)
    assert len(out) == 2  # origin+1.0 kept; 0.05 collapsed


def test_hybrid_base_plus_near_and_far() -> None:
    base = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    chunk = np.array([[0.05, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    merged = hybrid_merge(base, [chunk], nn_m=0.1)
    assert len(merged) == 2
    # Far point appended; near discarded.
    assert any(np.allclose(p, [1.0, 0.0, 0.0]) for p in merged)
    assert not any(np.allclose(p, [0.05, 0.0, 0.0]) for p in merged)
    assert any(np.allclose(p, [0.0, 0.0, 0.0]) for p in merged)


def test_hybrid_chunk_nn_override(tmp_path: Path) -> None:
    base_p = tmp_path / "base.ply"
    c0 = tmp_path / "chunk_00.ply"
    c4 = tmp_path / "chunk_04.ply"
    _write_ascii_xyz_ply(base_p, np.array([[0.0, 0.0, 0.0]], dtype=np.float64))
    # 0.12m from base: kept with default 0.10, dropped with 0.15 override.
    _write_ascii_xyz_ply(c0, np.array([[0.12, 0.0, 0.0]], dtype=np.float64))
    _write_ascii_xyz_ply(c4, np.array([[0.12, 0.1, 0.0]], dtype=np.float64))
    merged = hybrid_merge_paths(
        base_p,
        [c0, c4],
        nn_m=0.10,
        chunk_nn_overrides=parse_chunk_nn(["chunk_04=0.15"]),
    )
    # base + chunk_00 point; chunk_04 point rejected (0.12 < 0.15 to current merge)
    assert len(merged) == 2
    assert any(np.allclose(p, [0.12, 0.0, 0.0]) for p in merged)
    assert not any(np.allclose(p, [0.12, 0.1, 0.0]) for p in merged)


def test_write_xyz_ply_roundtrip(tmp_path: Path) -> None:
    xyz = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    out = tmp_path / "out.ply"
    write_xyz_ply(out, xyz)
    got = _read_ply_xyz(out)
    assert got.shape == (2, 3)
    np.testing.assert_allclose(got, xyz, atol=1e-3)


def test_parse_chunk_nn() -> None:
    assert parse_chunk_nn(["chunk_04=0.15", "chunk_01:0.2"]) == {
        "chunk_04": 0.15,
        "chunk_01": 0.2,
    }
