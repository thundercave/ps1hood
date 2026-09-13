"""Path α planarize: synthetic walls → segment → quads; tilt / empty fail."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from ps1_hood.reconstruct.facades import extract_facades
from ps1_hood.reconstruct.planarize import (
    DEFAULT_VERTICAL_DOT,
    HAS_OPEN3D,
    inliers_to_quad,
    planes_from_mapanything_ply,
    resolve_dense_ply,
    segment_vertical_planes_numpy,
    write_planes_json,
)


def _synthetic_street_cloud(
    *,
    n_wall: int = 800,
    n_ground: int = 600,
    noise: float = 0.02,
    seed: int = 0,
) -> np.ndarray:
    """Two vertical walls (x=5 and y=8) + ground z=0."""
    rng = np.random.default_rng(seed)
    # Wall A: x=5, y in [-4,4], z in [0, 8]
    ya = rng.uniform(-4, 4, n_wall)
    za = rng.uniform(0.2, 8.0, n_wall)
    wall_a = np.stack([np.full(n_wall, 5.0) + rng.normal(0, noise, n_wall), ya, za], axis=1)
    # Wall B: y=8, x in [-4,4], z in [0, 7]
    xb = rng.uniform(-4, 4, n_wall)
    zb = rng.uniform(0.2, 7.0, n_wall)
    wall_b = np.stack([xb, np.full(n_wall, 8.0) + rng.normal(0, noise, n_wall), zb], axis=1)
    # Ground
    xg = rng.uniform(-6, 6, n_ground)
    yg = rng.uniform(-6, 10, n_ground)
    ground = np.stack([xg, yg, rng.normal(0, noise, n_ground)], axis=1)
    return np.vstack([wall_a, wall_b, ground]).astype(np.float64)


def _write_xyz_ply(path: Path, xyz: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as fh:
        fh.write("ply\nformat ascii 1.0\n")
        fh.write(f"element vertex {len(xyz)}\n")
        fh.write("property float x\nproperty float y\nproperty float z\n")
        fh.write("end_header\n")
        for p in xyz:
            fh.write(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\n")


def test_segment_synthetic_two_walls() -> None:
    xyz = _synthetic_street_cloud()
    cam = np.array([0.0, 0.0, 2.0])
    hyps, ground, residual = segment_vertical_planes_numpy(
        xyz,
        cam,
        distance_threshold=0.08,
        min_inliers=80,
        max_planes=8,
        residual_stop=100,
        num_iterations=600,
    )
    assert len(hyps) >= 2
    for h in hyps:
        assert abs(float(h["n"][2])) <= DEFAULT_VERTICAL_DOT + 1e-6
        assert h["width_m"] * h["height_m"] >= 4.0
        assert abs(float(h["n"] @ h["center"] + h["d"])) < 0.2
    # Normals roughly axis-aligned
    dots = [abs(float(h["n"][0])) for h in hyps] + [abs(float(h["n"][1])) for h in hyps]
    assert max(dots) > 0.8


def test_inliers_to_quad_rejects_tiny() -> None:
    n = np.array([1.0, 0.0, 0.0])
    d = -1.0
    pts = np.array([[1.0, 0.0, 0.5], [1.0, 0.1, 0.6], [1.0, -0.1, 0.4]])
    assert inliers_to_quad(pts, n, d) is None


def test_inliers_to_quad_vertical_wall() -> None:
    rng = np.random.default_rng(1)
    ys = rng.uniform(-3, 3, 200)
    zs = rng.uniform(0.5, 6.0, 200)
    pts = np.stack([np.full(200, 4.0), ys, zs], axis=1)
    n = np.array([1.0, 0.0, 0.0])
    d = -4.0
    q = inliers_to_quad(pts, n, d, ground_z=0.0)
    assert q is not None
    assert q["width_m"] >= 1.5
    assert q["height_m"] >= 1.5
    assert len(q["corners"]) == 4
    assert abs(float(q["n"][2])) < 0.15


def test_planes_from_ply_numpy_fallback(tmp_path: Path) -> None:
    xyz = _synthetic_street_cloud(n_wall=600, n_ground=400)
    ply = tmp_path / "cloud.ply"
    _write_xyz_ply(ply, xyz)
    hyps, ground, n_res = planes_from_mapanything_ply(
        ply,
        np.array([0.0, 0.0, 2.0]),
        voxel=0.15,
        distance_threshold=0.10,
        min_inliers=60,
        max_planes=8,
        residual_stop=80,
        prefer_open3d=False,
    )
    assert len(hyps) >= 1
    assert n_res >= 0


def test_tilt_reject_via_vertical_filter() -> None:
    """Slanted roof-like plane should not become a façade hyp."""
    rng = np.random.default_rng(3)
    # Plane with strong up component: z = 0.5*x
    xs = rng.uniform(-2, 2, 500)
    ys = rng.uniform(-2, 2, 500)
    zs = 0.5 * xs + 3.0 + rng.normal(0, 0.02, 500)
    xyz = np.stack([xs, ys, zs], axis=1)
    hyps, ground, _ = segment_vertical_planes_numpy(
        xyz,
        np.array([0.0, 0.0, 2.0]),
        min_inliers=80,
        max_planes=4,
        residual_stop=50,
        distance_threshold=0.08,
    )
    # May pick ground-ish or nothing; no vertical façade expected
    for h in hyps:
        assert abs(float(h["n"][2])) < 0.15


def test_resolve_dense_ply_mapanything(tmp_path: Path) -> None:
    ply = tmp_path / "mapanything" / "cloud.ply"
    _write_xyz_ply(ply, np.zeros((10, 3)))
    assert resolve_dense_ply(tmp_path, "mapanything") == ply
    with pytest.raises(FileNotFoundError):
        resolve_dense_ply(tmp_path / "missing", "mapanything")


def test_extract_facades_planarize_branch_writes_planes_json(tmp_path: Path) -> None:
    """Dense synthetic PLY + planarize=True; wipe-on-fail still writes empty product."""
    xyz = _synthetic_street_cloud(n_wall=500, n_ground=300)
    ply = tmp_path / "cloud.ply"
    _write_xyz_ply(ply, xyz)
    # Uncorrelated images → ZNCC fail-loud
    frames = []
    rng = np.random.default_rng(4)
    for i in range(2):
        img = rng.integers(0, 255, (180, 240, 3), dtype=np.uint8)
        p = tmp_path / f"f{i}.jpg"
        cv2.imwrite(str(p), img)
        frames.append(
            {
                "path": str(p),
                "e": float(i * 5),
                "n": 0.0,
                "u": 2.0,
                "heading": 90.0,
                "pitch": 0.0,
                "fov": 90.0,
                "pano_id": f"p{i}",
            }
        )
    dest = tmp_path / "facades.obj"
    meta = extract_facades(
        ply,
        dest,
        frames=frames,
        n_planes=8,
        zncc_accept=0.90,
        planarize=True,
        voxel_m=0.15,
        plane_dist_m=0.10,
        keep_previous_on_fail=False,
        fallback_heading=False,
    )
    assert meta["source"] == "mapanything_planarize"
    assert meta["path_alpha"] is True
    assert meta.get("ok") is False
    assert (tmp_path / "planes.json").is_file()
    assert dest.is_file()


def test_extract_facades_legacy_unchanged_small_ply(tmp_path: Path) -> None:
    """Small PLY + planarize=None → Milestone A path (not auto planarize)."""
    rng = np.random.default_rng(5)
    frames = []
    for i in range(2):
        img = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
        p = tmp_path / f"s{i}.jpg"
        cv2.imwrite(str(p), img)
        frames.append(
            {
                "path": str(p),
                "e": float(i * 10),
                "n": 0.0,
                "u": 2.5,
                "heading": 0.0,
                "pitch": 0.0,
                "fov": 90.0,
                "pano_id": f"s{i}",
            }
        )
    dest = tmp_path / "facades.obj"
    meta = extract_facades(
        None, dest, frames=frames, zncc_accept=0.9, n_planes=4, planarize=False
    )
    assert meta["source"] == "photo_consistency"
    assert meta.get("path_alpha") is False


def test_write_planes_json(tmp_path: Path) -> None:
    path = tmp_path / "planes.json"
    write_planes_json(
        path,
        planes=[
            {
                "nx": 1.0,
                "ny": 0.0,
                "d": 5.0,
                "quad": [(5, -1, 0), (5, 1, 0), (5, 1, 4), (5, -1, 4)],
                "width_m": 2.0,
                "height_m": 4.0,
                "zncc": 0.45,
                "count": 100,
            }
        ],
        ground_z=0.0,
        residual_points=50,
        textured_maps=["textures/facade_00.jpg"],
    )
    body = path.read_text()
    assert "mapanything_planarize" in body or "photo_consistency" in body or "source" in body
    assert "facade_00" in body


def test_open3d_flag_documented() -> None:
    # Soft: just ensure constant exists; Open3D optional
    assert isinstance(HAS_OPEN3D, bool)

def test_fail_loud_preserves_prior_facade_artifacts(tmp_path: Path) -> None:
    """0 accepts + keep_previous: do not clobber non-empty product; write *.failed."""
    xyz = _synthetic_street_cloud(n_wall=400, n_ground=200)
    ply = tmp_path / "cloud.ply"
    _write_xyz_ply(ply, xyz)

    dest = tmp_path / "facades.obj"
    mtl = tmp_path / "facades.mtl"
    planes_json = tmp_path / "planes.json"
    tex_dir = tmp_path / "textures"
    tex_dir.mkdir()
    prior_obj = "\n".join(
        [
            "# prior good photo facades",
            "mtllib facades.mtl",
            "v 0 0 0",
            "v 1 0 0",
            "v 1 1 0",
            "v 0 1 0",
            "v 0 0 1",
            "v 1 0 1",
            "v 1 1 1",
            "v 0 1 1",
            "v 2 0 0",
            "v 3 0 0",
            "v 3 1 0",
            "v 2 1 0",
            "usemtl ground",
            "f 1 2 3 4",
            "usemtl facade_00",
            "f 5 6 7 8",
            "usemtl facade_01",
            "f 9 10 11 12",
        ]
        + ["# pad"] * 40
    ) + "\n"
    dest.write_text(prior_obj, encoding="ascii")
    mtl.write_text("newmtl facade_00\nKd 0.5 0.5 0.5\n", encoding="ascii")
    planes_json.write_text(
        '{"frame":"ENU","source":"photo_consistency","planes":[{"id":"facade_00"}],'
        '"ground_z":0,"residual_points":0,"gates":{"plane_count":1}}',
        encoding="utf-8",
    )
    tex = tex_dir / "facade_00.jpg"
    cv2.imwrite(str(tex), np.zeros((32, 32, 3), dtype=np.uint8))
    prior_obj_bytes = dest.read_bytes()
    prior_tex_bytes = tex.read_bytes()
    prior_json = planes_json.read_text(encoding="utf-8")

    frames = []
    rng = np.random.default_rng(7)
    for i in range(2):
        img = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
        p = tmp_path / f"g{i}.jpg"
        cv2.imwrite(str(p), img)
        frames.append(
            {
                "path": str(p),
                "e": float(i * 4),
                "n": 0.0,
                "u": 2.0,
                "heading": 90.0,
                "pitch": 0.0,
                "fov": 90.0,
                "pano_id": f"g{i}",
            }
        )

    meta = extract_facades(
        ply,
        dest,
        frames=frames,
        n_planes=6,
        zncc_accept=0.95,
        planarize=True,
        voxel_m=0.15,
        plane_dist_m=0.10,
        keep_previous_on_fail=True,
        fallback_heading=False,
    )
    assert meta["planes"] == 0
    assert meta.get("preserved_previous") is True
    assert meta.get("ok") is False
    assert dest.read_bytes() == prior_obj_bytes
    assert tex.read_bytes() == prior_tex_bytes
    assert planes_json.read_text(encoding="utf-8") == prior_json
    assert (tmp_path / "planes.failed.json").is_file()
    assert (tmp_path / "facades.failed.obj").is_file()


def test_inliers_to_quad_percentile_clamps_wide_slab() -> None:
    """Street-slab inliers → width clamped to DEFAULT_MAX_WIDTH_M."""
    from ps1_hood.reconstruct.planarize import DEFAULT_MAX_WIDTH_M

    rng = np.random.default_rng(9)
    ys = rng.uniform(-40, 40, 800)
    zs = rng.uniform(0.5, 8.0, 800)
    pts = np.stack([np.full(800, 5.0), ys, zs], axis=1)
    n = np.array([1.0, 0.0, 0.0])
    d = -5.0
    q = inliers_to_quad(pts, n, d, ground_z=0.0)
    assert q is not None
    assert q["width_m"] <= DEFAULT_MAX_WIDTH_M + 1e-6
    assert q["height_m"] >= 2.0

