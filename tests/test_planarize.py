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
    expand_hyps_for_scoring,
    inliers_to_quad,
    is_a_source,
    nms_keep_planes,
    planes_from_mapanything_ply,
    planes_from_product_json,
    resolve_a_ply,
    resolve_dense_ply,
    resolve_ma_ply,
    score_planar_hyps,
    segment_vertical_planes_numpy,
    split_long_hyp,
    union_keep_planes,
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



def test_is_strictly_better_textured_then_zncc() -> None:
    from ps1_hood.reconstruct.facades import _is_strictly_better

    prev = {"textured": 5, "plane_count": 7, "mean_zncc": 0.42}
    # 2-plane fallback must not beat 7-plane / 5-textured product
    ok, why = _is_strictly_better(
        {"textured": 2, "plane_count": 2, "mean_zncc": 0.377}, prev
    )
    assert ok is False
    assert "textured" in why

    # equal textured+planes needs higher mean_zncc
    ok2, _ = _is_strictly_better(
        {"textured": 5, "plane_count": 7, "mean_zncc": 0.42}, prev
    )
    assert ok2 is False

    ok3, why3 = _is_strictly_better(
        {"textured": 5, "plane_count": 7, "mean_zncc": 0.50}, prev
    )
    assert ok3 is True
    assert "clause3" in why3 and "mean_zncc" in why3

    # textured↑ alone with planes↓ / mean regress must NOT promote
    ok4, why4 = _is_strictly_better(
        {"textured": 6, "plane_count": 6, "mean_zncc": 0.30}, prev
    )
    assert ok4 is False
    assert "fewer planes" in why4 or "mean_zncc" in why4


def test_is_strictly_better_quality_keep_no_mean_regress() -> None:
    """PC hole after PR #23: 6/6/0.377 must not beat live 7/5/0.42."""
    from ps1_hood.reconstruct.facades import _is_strictly_better

    prev = {"textured": 5, "plane_count": 7, "mean_zncc": 0.42}

    # Incident: more textured but fewer planes + mean regress → False
    ok_hole, why_hole = _is_strictly_better(
        {"textured": 6, "plane_count": 6, "mean_zncc": 0.377}, prev
    )
    assert ok_hole is False
    assert "fewer planes" in why_hole

    # more textured, ≥planes, mean within 0.02 → True (clause1)
    ok1, why1 = _is_strictly_better(
        {"textured": 6, "plane_count": 8, "mean_zncc": 0.41}, prev
    )
    assert ok1 is True
    assert "clause1" in why1

    # same textured, mean +0.03 → True (clause3)
    ok3, why3 = _is_strictly_better(
        {"textured": 5, "plane_count": 7, "mean_zncc": 0.45}, prev
    )
    assert ok3 is True
    assert "clause3" in why3

    # same textured, more planes, mean flat → True (clause2)
    ok2, why2 = _is_strictly_better(
        {"textured": 5, "plane_count": 8, "mean_zncc": 0.42}, prev
    )
    assert ok2 is True
    assert "clause2" in why2

    # textured↑ but planes↓ (even with great mean) → False
    ok_down, why_down = _is_strictly_better(
        {"textured": 6, "plane_count": 5, "mean_zncc": 0.50}, prev
    )
    assert ok_down is False
    assert "fewer planes" in why_down


def test_weaker_fallback_does_not_clobber_better_product(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """2-plane fallback @ low ZNCC must keep 7-plane product; write *.candidate."""
    import json

    import ps1_hood.reconstruct.photo_planes as pp
    import ps1_hood.reconstruct.planarize as pl

    dest = tmp_path / "facades.obj"
    mtl = tmp_path / "facades.mtl"
    planes_json = tmp_path / "planes.json"
    tex_dir = tmp_path / "textures"
    tex_dir.mkdir()

    prior_planes = []
    for i in range(7):
        prior_planes.append(
            {
                "id": f"facade_{i:02d}",
                "n": [1.0, 0.0, 0.0],
                "d": -5.0,
                "zncc": 0.42,
                "texture": f"textures/facade_{i:02d}.jpg" if i < 5 else None,
                "width_m": 8.0,
                "height_m": 6.0,
                "inliers": 100,
                "source": "photo_consistency",
            }
        )
    planes_json.write_text(
        json.dumps(
            {
                "frame": "ENU",
                "source": "photo_consistency",
                "ground_z": 0.0,
                "planes": prior_planes,
                "residual_points": 0,
                "gates": {"mean_zncc": 0.42, "plane_count": 7},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for i in range(5):
        cv2.imwrite(
            str(tex_dir / f"facade_{i:02d}.jpg"),
            np.full((32, 32, 3), 40 + i, dtype=np.uint8),
        )
    dest.write_text(
        "\n".join(
            ["# prior good", "mtllib facades.mtl"]
            + [f"v {i} 0 0" for i in range(16)]
            + ["usemtl ground", "f 1 2 3 4"]
            + [
                f"usemtl facade_{i:02d}\nf {i * 4 + 1} {i * 4 + 2} {i * 4 + 3} {i * 4 + 4}"
                for i in range(3)
            ]
            + ["# pad"] * 40
        )
        + "\n",
        encoding="ascii",
    )
    mtl.write_text("newmtl facade_00\nKd 0.5 0.5 0.5\n", encoding="ascii")
    prior_obj = dest.read_bytes()
    prior_json = planes_json.read_text(encoding="utf-8")
    prior_tex = (tex_dir / "facade_00.jpg").read_bytes()

    xyz = _synthetic_street_cloud(n_wall=400, n_ground=200)
    ply = tmp_path / "cloud.ply"
    _write_xyz_ply(ply, xyz)

    frames = []
    rng = np.random.default_rng(11)
    for i in range(2):
        img = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
        p = tmp_path / f"w{i}.jpg"
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
                "pano_id": f"w{i}",
            }
        )

    def _zero_alpha(hyps, frames, **kwargs):  # noqa: ANN001
        return []

    def _weak_fallback(frames, xyz=None, **kwargs):  # noqa: ANN001
        out = []
        for dxy in (5.0, 8.0):
            out.append(
                {
                    "n": np.array([1.0, 0.0, 0.0]),
                    "d": -dxy,
                    "center": np.array([dxy, 0.0, 3.0]),
                    "width_m": 6.0,
                    "height_m": 5.0,
                    "zncc": 0.377,
                    "ok": True,
                    "source": "heading_distance",
                    "count": 50,
                }
            )
        return out

    monkeypatch.setattr(pl, "score_planar_hyps", _zero_alpha)
    monkeypatch.setattr(pp, "search_photo_consistent_planes", _weak_fallback)

    meta = extract_facades(
        ply,
        dest,
        frames=frames,
        n_planes=8,
        zncc_accept=0.40,
        planarize=True,
        voxel_m=0.15,
        plane_dist_m=0.10,
        keep_previous_on_fail=True,
        fallback_heading=True,
    )

    assert meta.get("preserved_previous") is True
    assert meta.get("rejected_weaker") is True
    assert meta.get("candidate_planes") == 2
    assert dest.read_bytes() == prior_obj
    assert planes_json.read_text(encoding="utf-8") == prior_json
    assert (tex_dir / "facade_00.jpg").read_bytes() == prior_tex
    assert (tmp_path / "planes.candidate.json").is_file()
    assert (tmp_path / "facades.candidate.obj").is_file()
    assert len(list(tex_dir.glob("facade_*.jpg"))) == 5


def test_score_planar_hyps_fail_loud_marks_sentinel(caplog: pytest.LogCaptureFixture) -> None:
    """When every hyp is skipped pre-score, FAIL-LOUD must say SENTINEL + scored=0."""
    import logging

    from ps1_hood.reconstruct.planarize import score_planar_hyps

    # Hyp far away → xy40 or depth skip; no frames that can score
    hyps = [
        {
            "n": np.array([1.0, 0.0, 0.0]),
            "d": -200.0,
            "center": np.array([200.0, 0.0, 3.0]),
            "width_m": 8.0,
            "height_m": 6.0,
            "source": "ma_segment",
        }
    ]
    frames = [
        {
            "path": "/nonexistent.jpg",
            "e": 0.0,
            "n": 0.0,
            "u": 2.0,
            "heading": 0.0,
            "pitch": 0.0,
            "fov": 90.0,
            "pano_id": "a",
        },
        {
            "path": "/nonexistent2.jpg",
            "e": 5.0,
            "n": 0.0,
            "u": 2.0,
            "heading": 0.0,
            "pitch": 0.0,
            "fov": 90.0,
            "pano_id": "b",
        },
    ]
    with caplog.at_level(logging.ERROR, logger="ps1_hood.reconstruct.planarize"):
        kept = score_planar_hyps(hyps, frames, zncc_accept=0.40)
    assert kept == []
    joined = " ".join(r.message for r in caplog.records)
    assert "SENTINEL" in joined
    assert "scored=0" in joined


def test_split_long_hyp_overlapping_windows() -> None:
    """Wide peel → overlapping ~10 m windows with same n,d."""
    n = np.array([1.0, 0.0, 0.0])
    d = -5.0
    # 24 m wide wall along +Y at x=5
    corners = np.array(
        [
            [5.0, -12.0, 0.5],
            [5.0, 12.0, 0.5],
            [5.0, 12.0, 8.0],
            [5.0, -12.0, 8.0],
        ],
        dtype=np.float64,
    )
    hyp = {
        "n": n,
        "d": d,
        "center": corners.mean(axis=0),
        "width_m": 24.0,
        "height_m": 7.5,
        "corners": corners,
        "source": "ma_segment",
    }
    parts = split_long_hyp(hyp)
    assert len(parts) >= 2
    for p in parts:
        assert abs(float(p["n"] @ n)) > 0.99
        assert abs(float(p["d"]) - d) < 1e-6
        assert 1.5 <= float(p["width_m"]) <= 8.0 + 1e-6
        # Centers closer to ends / mid than original mid-block-only crop
        assert abs(float(p["center"][0]) - 5.0) < 0.2
    expanded = expand_hyps_for_scoring([hyp])
    assert len(expanded) == len(parts)


def test_score_planar_hyps_street_parallel_not_depth_sentinel(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Cams beside a façade (small |n·C+d|, ok Euclidean) must not all skip as SENTINEL.

    PR#16 signed-depth min-over-cams gate zeroed scored on street-parallel peels.
    Soft Euclidean + deferred refine must still reach score_vertical_plane (or
    fail for views/load — not hard depth) when images are missing.
    """
    import logging

    # Wall at x=8; cams at x=0 along +Y (beside wall) — signed |n·C+d|=8,
    # Euclidean center→cam depends on Y. Place a LONG peel whose AABB center
    # is far in Y so old euclidean gate at mid-block would fail; split brings
    # windows near cams.
    n = np.array([1.0, 0.0, 0.0])
    d = -8.0
    corners = np.array(
        [
            [8.0, -20.0, 0.5],
            [8.0, 20.0, 0.5],
            [8.0, 20.0, 9.0],
            [8.0, -20.0, 9.0],
        ],
        dtype=np.float64,
    )
    hyp = {
        "n": n,
        "d": d,
        "center": corners.mean(axis=0),  # (8, 0, ~4.75)
        "width_m": 40.0,
        "height_m": 8.5,
        "corners": corners,
        "source": "ma_segment",
        "inliers": 5000,
    }
    # Missing images → load skip after views picked; but must NOT be depth-only sentinel
    frames = []
    for i, y in enumerate((-5.0, 5.0)):
        frames.append(
            {
                "path": str(tmp_path / f"missing_{i}.jpg"),
                "e": 0.0,
                "n": y,
                "u": 2.0,
                "heading": 90.0,  # look +E toward wall
                "pitch": 0.0,
                "fov": 90.0,
                "pano_id": f"p{i}",
            }
        )
    with caplog.at_level(logging.ERROR, logger="ps1_hood.reconstruct.planarize"):
        kept = score_planar_hyps([hyp], frames, zncc_accept=0.40)
    assert kept == []
    joined = " ".join(r.message for r in caplog.records)
    assert "SENTINEL" in joined
    # Hard depth should not consume the long peel after split (windows near cams)
    assert "depth=0" in joined or "depth=0;" in joined or "depth=0 " in joined
    # Failure is views/load (no images), not systematic depth wipe of all windows
    assert ("load=" in joined) or ("views=" in joined)


def _write_textured_views_for_wall(tmp_path: Path) -> tuple[list[dict], np.ndarray, float, np.ndarray]:
    """Two cams looking +E at vertical wall x=8 with photo-consistent textures."""
    from ps1_hood.reconstruct.photo_planes import (
        View,
        plane_homography,
        Rt_from_frame,
        K_from_frame,
        score_vertical_plane,
    )

    rng = np.random.default_rng(7)
    # Distinctive façade pattern
    img0 = np.zeros((480, 640, 3), dtype=np.uint8)
    noise = rng.integers(40, 220, (480, 640), dtype=np.uint8)
    img0[:, :, 0] = noise
    img0[:, :, 1] = np.roll(noise, 11, axis=1)
    img0[:, :, 2] = np.roll(noise, 5, axis=0)
    cv2.rectangle(img0, (120, 60), (520, 420), (40, 160, 230), -1)
    cv2.putText(img0, "WALL", (200, 260), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 255), 4)

    n = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    d = -8.0
    center = np.array([8.0, 0.0, 4.0], dtype=np.float64)

    frames = []
    for i, (e, north, heading) in enumerate(
        [(0.0, -1.5, 90.0), (0.5, 1.5, 90.0)]
    ):
        fr = {
            "e": e,
            "n": north,
            "u": 2.0,
            "heading": heading,
            "pitch": 0.0,
            "fov": 90.0,
            "pano_id": f"tex{i}",
        }
        frames.append(fr)

    # Build consistent src image via plane H from cam0 pattern
    R0, t0 = Rt_from_frame({**frames[0], "path": "x"})
    R1, t1 = Rt_from_frame({**frames[1], "path": "x"})
    K0 = K_from_frame(640, 480, 90.0)
    K1 = K_from_frame(640, 480, 90.0)
    H = plane_homography(K0, R0, t0, K1, R1, t1, n, d)
    img1 = cv2.warpPerspective(img0, H, (640, 480), flags=cv2.INTER_LINEAR)

    paths = []
    for i, im in enumerate((img0, img1)):
        p = tmp_path / f"tex_wall_{i}.jpg"
        cv2.imwrite(str(p), im)
        frames[i]["path"] = str(p)
        paths.append(p)

    # Sanity: direct score_vertical_plane should accept
    v0 = View(img0, K0, R0, t0, 0, "tex0")
    v1 = View(img1, K1, R1, t1, 1, "tex1")
    direct = score_vertical_plane(
        [v0, v1], n, d, center, width_m=6.0, height_m=6.0, zncc_accept=0.35, patch=64
    )
    assert direct["ok"], f"sanity direct score failed: {direct}"
    return frames, n, d, center


def test_score_planar_hyps_synthetic_facade_finite_zncc(tmp_path: Path) -> None:
    """Plausible MA peel + photo-consistent views → finite ZNCC ≥ 0.35 (not sentinel)."""
    frames, n, d, center = _write_textured_views_for_wall(tmp_path)
    hyp = {
        "n": n,
        "d": d,
        "center": center,
        "width_m": 8.0,
        "height_m": 7.0,
        "corners": np.array(
            [
                [8.0, -4.0, 0.5],
                [8.0, 4.0, 0.5],
                [8.0, 4.0, 7.5],
                [8.0, -4.0, 7.5],
            ],
            dtype=np.float64,
        ),
        "source": "ma_segment",
        "inliers": 800,
    }
    kept = score_planar_hyps([hyp], frames, zncc_accept=0.35, refine=True)
    assert len(kept) >= 1, "expected ≥1 accept on synthetic photo-consistent façade"
    assert float(kept[0]["zncc"]) >= 0.35
    assert not (isinstance(kept[0]["zncc"], float) and kept[0]["zncc"] == -1.0)


def test_score_planar_hyps_refine_rescues_depth_offset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """±n refine must try depth offsets (Milestone A parity) and can beat as-is seed."""
    frames, n, d, center = _write_textured_views_for_wall(tmp_path)
    bad_center = center - n * 2.0
    bad_d = float(-n @ bad_center)
    hyp = {
        "n": n,
        "d": bad_d,
        "center": bad_center,
        "width_m": 8.0,
        "height_m": 7.0,
        "source": "ma_segment",
        "inliers": 800,
    }

    # Instrument: true depth candidate returns higher ZNCC than seed.
    import ps1_hood.reconstruct.photo_planes as pp

    real = pp.score_vertical_plane
    calls: list[float] = []

    def wrapped(views, n_c, d_c, c_c, **kwargs):  # noqa: ANN001
        # Signed offset from true wall along original +n
        depth_err = abs(float(n @ c_c) - float(n @ center))
        calls.append(depth_err)
        out = real(views, n_c, d_c, c_c, **kwargs)
        # Force clear preference for near-true depth so refine wins over seed.
        if isinstance(out.get("zncc"), float) and not __import__("math").isnan(out["zncc"]):
            if depth_err < 0.25:
                out = {**out, "zncc": 0.95, "ok": True, "reason": "accept"}
            elif depth_err > 1.5:
                out = {**out, "zncc": 0.20, "ok": False, "reason": "zncc below threshold"}
        return out

    monkeypatch.setattr(pp, "score_vertical_plane", wrapped)
    kept = score_planar_hyps([hyp], frames, zncc_accept=0.35, refine=True)
    assert len(kept) >= 1
    assert float(kept[0]["zncc"]) >= 0.35
    assert abs(float(kept[0]["refine_delta_m"])) >= 1.0
    # Refine must have scored multiple depth offsets (not seed-only).
    assert len(calls) >= 3
    assert min(calls) < 0.5  # at least one candidate near true wall


def test_plane_homography_rejects_near_camera() -> None:
    """Ill-conditioned |c|<0.5 must raise (anti-fold / sentinel-ish warps)."""
    import pytest
    from ps1_hood.reconstruct.photo_planes import plane_homography

    K = np.eye(3, dtype=np.float64)
    R = np.eye(3, dtype=np.float64)
    t = np.zeros(3, dtype=np.float64)  # cam at origin
    n = np.array([0.0, 0.0, 1.0])
    d = -0.2  # plane 0.2 m in front → |c|=0.2
    with pytest.raises(ValueError, match="near reference"):
        plane_homography(K, R, t, K, R, t - np.array([1.0, 0, 0]), n, d)


def test_split_long_hyp_8m_windows() -> None:
    """30 m peel, trigger=8, window=8, overlap=2 → ≥3 pieces; same n,d; ~6–8 m centers."""
    n = np.array([1.0, 0.0, 0.0])
    d = -5.0
    corners = np.array(
        [
            [5.0, -15.0, 0.5],
            [5.0, 15.0, 0.5],
            [5.0, 15.0, 8.5],
            [5.0, -15.0, 8.5],
        ],
        dtype=np.float64,
    )
    hyp = {
        "n": n,
        "d": d,
        "center": corners.mean(axis=0),
        "width_m": 30.0,
        "height_m": 8.0,
        "corners": corners,
        "source": "ma_segment",
    }
    parts = split_long_hyp(
        hyp, trigger_width_m=8.0, window_m=8.0, overlap_m=2.0
    )
    assert len(parts) >= 3
    ys = sorted(float(p["center"][1]) for p in parts)
    # step = window - overlap = 6 m; last remainder window may tighten the gap
    gaps = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
    assert all(4.0 <= g <= 8.5 for g in gaps)
    assert ys[-1] - ys[0] >= 12.0
    for p in parts:
        assert abs(float(p["n"] @ n)) > 0.99
        assert abs(float(p["d"]) - d) < 1e-6
        assert p.get("split_parent") is True


def test_split_no_op_below_trigger() -> None:
    """Width 7 m → single hyp when trigger=8."""
    n = np.array([0.0, 1.0, 0.0])
    d = -3.0
    hyp = {
        "n": n,
        "d": d,
        "center": np.array([0.0, 3.0, 4.0]),
        "width_m": 7.0,
        "height_m": 6.0,
        "source": "ma_segment",
    }
    parts = split_long_hyp(
        hyp, trigger_width_m=8.0, window_m=8.0, overlap_m=2.0
    )
    assert len(parts) == 1
    assert parts[0] is hyp or float(parts[0]["width_m"]) == 7.0


def test_split_uses_raw_width_when_clamped() -> None:
    """Pre-clamp raw_width_m > clamped width_m still expands façade windows."""
    n = np.array([1.0, 0.0, 0.0])
    d = -5.0
    # Clamped 25 m corners, but raw was 40 m
    corners = np.array(
        [
            [5.0, -12.5, 0.5],
            [5.0, 12.5, 0.5],
            [5.0, 12.5, 8.0],
            [5.0, -12.5, 8.0],
        ],
        dtype=np.float64,
    )
    hyp = {
        "n": n,
        "d": d,
        "center": corners.mean(axis=0),
        "width_m": 25.0,
        "raw_width_m": 40.0,
        "height_m": 7.5,
        "corners": corners,
        "source": "ma_segment",
    }
    parts = split_long_hyp(
        hyp, trigger_width_m=8.0, window_m=8.0, overlap_m=2.0
    )
    # 40 m / step 6 → more windows than 25 m alone
    parts_clamped = split_long_hyp(
        {**hyp, "raw_width_m": 25.0},
        trigger_width_m=8.0,
        window_m=8.0,
        overlap_m=2.0,
    )
    assert len(parts) > len(parts_clamped)


def test_union_promote_beats_product() -> None:
    from ps1_hood.reconstruct.facades import _is_strictly_better

    prev = {"textured": 5, "plane_count": 7, "mean_zncc": 0.42}
    ok, why = _is_strictly_better(
        {"textured": 6, "plane_count": 8, "mean_zncc": 0.41}, prev
    )
    assert ok is True
    assert "clause1" in why and "textured" in why


def test_union_no_promote_weaker() -> None:
    """Mirrors PC post-#20: 1 textured @ 0.44 must not beat 7/5/0.42."""
    from ps1_hood.reconstruct.facades import _is_strictly_better

    prev = {"textured": 5, "plane_count": 7, "mean_zncc": 0.42}
    ok, why = _is_strictly_better(
        {"textured": 1, "plane_count": 1, "mean_zncc": 0.44}, prev
    )
    assert ok is False
    assert "textured" in why


def test_hybrid_seeds_scored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """MA peel + A heading seeds share one score_planar_hyps / NMS pass."""
    import logging

    import ps1_hood.reconstruct.photo_planes as pp

    # Two well-separated façades: MA at y=-20, A at y=+20
    ma_hyp = {
        "n": np.array([1.0, 0.0, 0.0]),
        "d": -8.0,
        "center": np.array([8.0, -20.0, 4.0]),
        "width_m": 8.0,
        "height_m": 8.0,
        "source": "ma_segment",
    }
    a_hyp = {
        "n": np.array([1.0, 0.0, 0.0]),
        "d": -8.0,
        "center": np.array([8.0, 20.0, 4.0]),
        "width_m": 8.0,
        "height_m": 9.0,
        "source": "heading_distance",
    }
    frames = []
    for i, y in enumerate((-20.0, 20.0)):
        img = np.full((64, 64, 3), 80, dtype=np.uint8)
        p = tmp_path / f"h{i}.jpg"
        cv2.imwrite(str(p), img)
        frames.append(
            {
                "path": str(p),
                "e": 0.0,
                "n": y,
                "u": 2.0,
                "heading": 90.0,
                "pitch": 0.0,
                "fov": 90.0,
                "pano_id": f"h{i}",
            }
        )

    def _fake_score(views, n, d, center, **kwargs):  # noqa: ANN001
        cy = float(np.asarray(center)[1])
        # Accept both; slightly higher ZNCC for MA so sort is stable
        z = 0.50 if cy < 0 else 0.45
        return {
            "ok": True,
            "zncc": z,
            "n": np.asarray(n, dtype=np.float64),
            "d": float(d),
            "center": np.asarray(center, dtype=np.float64),
        }

    monkeypatch.setattr(pp, "score_vertical_plane", _fake_score)
    monkeypatch.setattr(
        pp,
        "_pick_scoring_views",
        lambda frames, n, c, max_views=4, min_frontal=0.25: [0, 1],
    )
    monkeypatch.setattr(
        pp,
        "load_view",
        lambda fr, i: type(
            "V",
            (),
            {
                "Rcw": np.eye(3),
                "t": np.array([0.0, 0.0, 0.0]),
                "K": np.eye(3),
                "image": np.zeros((64, 64), dtype=np.float32),
                "gray": np.zeros((64, 64), dtype=np.float32),
            },
        )(),
    )

    with caplog.at_level(logging.INFO, logger="ps1_hood.reconstruct.planarize"):
        kept = score_planar_hyps(
            [ma_hyp],
            frames,
            zncc_accept=0.35,
            max_keep=8,
            refine=False,
            seed_hyps=[a_hyp],
            split_long=False,
        )
    assert len(kept) >= 2
    sources = {str(p.get("source")) for p in kept}
    assert "ma_segment" in sources
    assert "heading_distance" in sources
    joined = " ".join(r.message for r in caplog.records)
    assert "hybrid" in joined.lower() or "a_kept" in joined


def test_extract_facades_hybrid_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Path α hybrid default: dual-arm full A search + MA peels (no A seed inject)."""
    import ps1_hood.reconstruct.photo_planes as pp
    import ps1_hood.reconstruct.planarize as pl

    xyz = _synthetic_street_cloud(n_wall=400, n_ground=200)
    ply = tmp_path / "cloud.ply"
    _write_xyz_ply(ply, xyz)
    frames = []
    rng = np.random.default_rng(21)
    for i in range(2):
        img = rng.integers(0, 255, (80, 100, 3), dtype=np.uint8)
        p = tmp_path / f"hy{i}.jpg"
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
                "pano_id": f"hy{i}",
            }
        )

    called = {"search": 0, "seed_hyps": None}

    def _search(*args, **kwargs):  # noqa: ANN001
        called["search"] += 1
        return [
            {
                "n": np.array([1.0, 0.0, 0.0]),
                "d": -8.0,
                "center": np.array([8.0, 20.0, 4.0]),
                "width_m": 8.0,
                "height_m": 9.0,
                "zncc": 0.42,
                "ok": True,
                "source": "heading_distance",
            }
        ]

    def _score(hyps, frames, **kwargs):  # noqa: ANN001
        called["seed_hyps"] = kwargs.get("seed_hyps")
        return [
            {
                "n": np.array([1.0, 0.0, 0.0]),
                "d": -5.0,
                "center": np.array([5.0, 0.0, 3.0]),
                "width_m": 8.0,
                "height_m": 6.0,
                "zncc": 0.439,
                "ok": True,
                "source": "ma_segment",
                "count": 100,
            }
        ]

    monkeypatch.setattr(pp, "search_photo_consistent_planes", _search)
    monkeypatch.setattr(pl, "score_planar_hyps", _score)

    dest = tmp_path / "facades.obj"
    meta = extract_facades(
        ply,
        dest,
        frames=frames,
        n_planes=8,
        zncc_accept=0.35,
        planarize=True,
        voxel_m=0.15,
        plane_dist_m=0.10,
        keep_previous_on_fail=False,
        fallback_heading=False,
        hybrid_heading=True,
    )
    assert called["search"] >= 1
    assert called["seed_hyps"] is None  # dual-arm: no A inject into MA scorer
    assert meta.get("source") == "mapanything_hybrid"
    assert int(meta.get("planes") or 0) >= 2


def _plane(center_xy, *, zncc=0.5, split=False, d=-8.0):
    x, y = center_xy
    return {
        "n": np.array([1.0, 0.0, 0.0]),
        "d": float(d),
        "center": np.array([float(x), float(y), 4.0]),
        "width_m": 8.0,
        "height_m": 8.0,
        "zncc": float(zncc),
        "source": "ma_segment",
        "split_parent": bool(split),
    }


def test_nms_split_siblings_keep_beyond_4m() -> None:
    """Split siblings ~5 m apart survive at split_xy=4; closer pair collapses."""
    far = [
        _plane((8.0, 0.0), zncc=0.55, split=True),
        _plane((8.0, 5.0), zncc=0.50, split=True),  # 5 m > 4 m
    ]
    kept_far = nms_keep_planes(far, max_keep=8)
    assert len(kept_far) == 2

    near = [
        _plane((8.0, 0.0), zncc=0.55, split=True),
        _plane((8.0, 3.0), zncc=0.50, split=True),  # 3 m < 4 m
    ]
    kept_near = nms_keep_planes(near, max_keep=8)
    assert len(kept_near) == 1
    assert float(kept_near[0]["zncc"]) == pytest.approx(0.55)


def test_nms_non_split_still_collapses_under_6m() -> None:
    """Without split_parent, default XY=6 still merges a 5 m pair."""
    pair = [
        _plane((8.0, 0.0), zncc=0.55, split=False),
        _plane((8.0, 5.0), zncc=0.50, split=False),
    ]
    kept = nms_keep_planes(pair, max_keep=8)
    assert len(kept) == 1

    # Same pair with one marked split uses 4 m → both keep
    mixed = [
        _plane((8.0, 0.0), zncc=0.55, split=True),
        _plane((8.0, 5.0), zncc=0.50, split=False),
    ]
    kept_mixed = nms_keep_planes(mixed, max_keep=8)
    assert len(kept_mixed) == 2


def test_extract_facades_peel_knobs_forwarded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLI peel knobs reach planes_from_mapanything_ply + score_planar_hyps."""
    import ps1_hood.reconstruct.photo_planes as pp
    import ps1_hood.reconstruct.planarize as pl

    xyz = _synthetic_street_cloud(n_wall=400, n_ground=200)
    ply = tmp_path / "cloud.ply"
    _write_xyz_ply(ply, xyz)
    frames = []
    rng = np.random.default_rng(7)
    for i in range(2):
        img = rng.integers(0, 255, (80, 100, 3), dtype=np.uint8)
        p = tmp_path / f"pk{i}.jpg"
        cv2.imwrite(str(p), img)
        frames.append(
            {
                "path": str(p),
                "e": 0.0,
                "n": float(i),
                "u": 2.0,
                "heading": 90.0,
                "pitch": 0.0,
                "fov": 90.0,
                "pano_id": f"pk{i}",
            }
        )

    seen: dict = {}

    def _fake_planes(ply_path, cam_c, **kwargs):  # noqa: ANN001
        seen["planes_kwargs"] = dict(kwargs)
        return (
            [
                {
                    "n": np.array([1.0, 0.0, 0.0]),
                    "d": -5.0,
                    "center": np.array([5.0, 0.0, 3.0]),
                    "width_m": 8.0,
                    "height_m": 6.0,
                    "source": "ma_segment",
                }
            ],
            {"z": 0.0},
            10,
        )

    def _fake_score(hyps, frames, **kwargs):  # noqa: ANN001
        seen["score_kwargs"] = dict(kwargs)
        return [
            {
                "n": np.array([1.0, 0.0, 0.0]),
                "d": -5.0,
                "center": np.array([5.0, 0.0, 3.0]),
                "width_m": 8.0,
                "height_m": 6.0,
                "zncc": 0.5,
                "ok": True,
                "source": "ma_segment",
                "count": 100,
            }
        ]

    monkeypatch.setattr(pl, "planes_from_mapanything_ply", _fake_planes)
    monkeypatch.setattr(pl, "score_planar_hyps", _fake_score)
    monkeypatch.setattr(pp, "hypothesize_vertical_planes", lambda *a, **k: [])

    extract_facades(
        ply,
        tmp_path / "facades.obj",
        frames=frames,
        n_planes=8,
        zncc_accept=0.35,
        planarize=True,
        keep_previous_on_fail=False,
        fallback_heading=False,
        hybrid_heading=False,
        min_inliers=300,
        residual_stop=1000,
        vertical_dot=0.18,
        peel_max_planes=32,
        nms_xy_m=6.0,
        nms_xy_split_m=4.0,
    )
    pk = seen["planes_kwargs"]
    assert pk["min_inliers"] == 300
    assert pk["residual_stop"] == 1000
    assert pk["vertical_dot_max"] == pytest.approx(0.18)
    assert pk["max_planes"] == 32
    sk = seen["score_kwargs"]
    assert sk["nms_xy_m"] == pytest.approx(6.0)
    assert sk["nms_xy_split_m"] == pytest.approx(4.0)
    assert sk["union_strategy"] == "a_priority"


def test_facades_cli_exposes_peel_knobs() -> None:
    from click.testing import CliRunner

    from ps1_hood.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["facades", "--help"])
    assert result.exit_code == 0
    help_text = result.output
    for flag in (
        "--min-inliers",
        "--residual-stop",
        "--vertical-dot",
        "--peel-max-planes",
        "--nms-xy",
        "--nms-xy-split",
        "--union-strategy",
        "--max-planes",
        "--hybrid-a-full-search",
        "--no-hybrid-a-full-search",
        "--a-source",
        "--ma-source",
        "--control-out",
    ):
        assert flag in help_text
    assert "a_priority" in help_text
    assert "default: 16" in help_text


def _a_plane(center_xy, *, zncc=0.40, d=-8.0, source="heading_distance", split=False):
    x, y = center_xy
    return {
        "n": np.array([1.0, 0.0, 0.0]),
        "d": float(d),
        "center": np.array([float(x), float(y), 4.0]),
        "width_m": 8.0,
        "height_m": 8.0,
        "zncc": float(zncc),
        "source": source,
        "split_parent": bool(split),
    }


def test_a_priority_keeps_a_drops_overlapping_ma() -> None:
    """Many A + overlapping MA → A kept, overlapping MA dropped, far MA added."""
    a_planes = [
        _a_plane((8.0, 0.0), zncc=0.40),
        _a_plane((8.0, 20.0), zncc=0.42),
        _a_plane((8.0, 40.0), zncc=0.41),
        _a_plane((8.0, 60.0), zncc=0.39),
        _a_plane((8.0, 80.0), zncc=0.38),
    ]
    overlapping_ma = [
        _plane((8.0, 1.0), zncc=0.55),  # 1 m from first A — dup
        _plane((8.0, 21.0), zncc=0.52),  # 1 m from second A — dup
    ]
    far_ma = [_plane((8.0, 200.0), zncc=0.45)]
    accepted = a_planes + overlapping_ma + far_ma
    tel: dict = {}
    kept = union_keep_planes(
        accepted, strategy="a_priority", max_keep=16, telemetry=tel
    )
    sources = [str(p.get("source")) for p in kept]
    assert sources.count("heading_distance") == 5
    assert sources.count("ma_segment") == 1
    assert tel["strategy"] == "a_priority"
    assert tel["a_pre_nms"] == 5
    assert tel["a_kept"] == 5
    assert tel["ma_pre_nms"] == 3
    assert tel["ma_added"] == 1
    assert tel["union_kept"] == 6
    assert tel["pre_nms"] == 8
    # Surviving MA is the far one, not a same-wall overlap
    ma = [p for p in kept if p["source"] == "ma_segment"]
    assert float(ma[0]["center"][1]) == pytest.approx(200.0)


def test_union_strategy_nms_regression() -> None:
    """``nms`` still ZNCC-sorts: higher-ZNCC MA on the same wall evicts A."""
    accepted = [
        _a_plane((8.0, 0.0), zncc=0.40),
        _plane((8.0, 1.0), zncc=0.55),  # overlapping, better ZNCC
        _a_plane((8.0, 20.0), zncc=0.42),
        _plane((8.0, 40.0), zncc=0.45),  # far MA
    ]
    tel: dict = {}
    kept = union_keep_planes(accepted, strategy="nms", max_keep=16, telemetry=tel)
    sources = {str(p.get("source")) for p in kept}
    assert "ma_segment" in sources
    # A at y=0 is a dup of MA at y=1 under ZNCC-first NMS
    a_ys = [float(p["center"][1]) for p in kept if p["source"] == "heading_distance"]
    ma_ys = [float(p["center"][1]) for p in kept if p["source"] == "ma_segment"]
    assert 0.0 not in a_ys
    assert 1.0 in ma_ys
    assert tel["strategy"] == "nms"
    assert tel["a_pre_nms"] == 2
    assert tel["ma_pre_nms"] == 2
    # Contrast: a_priority keeps both A and drops overlapping MA
    tel_a: dict = {}
    kept_a = union_keep_planes(
        accepted, strategy="a_priority", max_keep=16, telemetry=tel_a
    )
    a_ys_p = [float(p["center"][1]) for p in kept_a if p["source"] == "heading_distance"]
    ma_ys_p = [float(p["center"][1]) for p in kept_a if p["source"] == "ma_segment"]
    assert sorted(a_ys_p) == pytest.approx([0.0, 20.0])
    assert 1.0 not in ma_ys_p
    assert 40.0 in ma_ys_p
    assert tel_a["a_kept"] == 2
    assert tel_a["ma_added"] == 1


def test_a_priority_max_keep_16_allows_over_12() -> None:
    """Hybrid keep cap 16 can retain more than 12 well-separated A planes."""
    accepted = [
        _a_plane((8.0, float(i * 20)), zncc=0.40 + i * 0.001)
        for i in range(14)
    ]
    tel: dict = {}
    kept = union_keep_planes(
        accepted, strategy="a_priority", max_keep=16, telemetry=tel
    )
    assert len(kept) == 14
    assert len(kept) > 12
    assert tel["a_kept"] == 14
    assert tel["union_kept"] == 14
    capped = union_keep_planes(accepted, strategy="a_priority", max_keep=12)
    assert len(capped) == 12


def test_extract_facades_forwards_union_strategy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ps1_hood.reconstruct.photo_planes as pp
    import ps1_hood.reconstruct.planarize as pl

    xyz = _synthetic_street_cloud(n_wall=400, n_ground=200)
    ply = tmp_path / "cloud.ply"
    _write_xyz_ply(ply, xyz)
    frames = []
    rng = np.random.default_rng(3)
    for i in range(2):
        img = rng.integers(0, 255, (80, 100, 3), dtype=np.uint8)
        pth = tmp_path / f"us{i}.jpg"
        cv2.imwrite(str(pth), img)
        frames.append(
            {
                "path": str(pth),
                "e": 0.0,
                "n": float(i),
                "u": 2.0,
                "heading": 90.0,
                "pitch": 0.0,
                "fov": 90.0,
                "pano_id": f"us{i}",
            }
        )

    seen: dict = {}

    def _fake_planes(ply_path, cam_c, **kwargs):  # noqa: ANN001
        return (
            [
                {
                    "n": np.array([1.0, 0.0, 0.0]),
                    "d": -5.0,
                    "center": np.array([5.0, 0.0, 3.0]),
                    "width_m": 8.0,
                    "height_m": 6.0,
                    "source": "ma_segment",
                }
            ],
            {"z": 0.0},
            10,
        )

    def _fake_score(hyps, frames, **kwargs):  # noqa: ANN001
        seen["score_kwargs"] = dict(kwargs)
        return [
            {
                "n": np.array([1.0, 0.0, 0.0]),
                "d": -5.0,
                "center": np.array([5.0, 0.0, 3.0]),
                "width_m": 8.0,
                "height_m": 6.0,
                "zncc": 0.5,
                "ok": True,
                "source": "ma_segment",
                "count": 100,
            }
        ]

    def _fake_search(*a, **k):  # noqa: ANN001
        seen["search_called"] = True
        return [
            {
                "n": np.array([1.0, 0.0, 0.0]),
                "d": -8.0,
                "center": np.array([8.0, 40.0, 4.0]),
                "width_m": 8.0,
                "height_m": 9.0,
                "zncc": 0.41,
                "ok": True,
                "source": "heading_distance",
            }
        ]

    def _fake_union(accepted, **kwargs):  # noqa: ANN001
        seen["union_kwargs"] = dict(kwargs)
        return list(accepted)

    monkeypatch.setattr(pl, "planes_from_mapanything_ply", _fake_planes)
    monkeypatch.setattr(pl, "score_planar_hyps", _fake_score)
    monkeypatch.setattr(pp, "search_photo_consistent_planes", _fake_search)
    monkeypatch.setattr(pl, "union_keep_planes", _fake_union)

    extract_facades(
        ply,
        tmp_path / "facades.obj",
        frames=frames,
        n_planes=16,
        zncc_accept=0.35,
        planarize=True,
        keep_previous_on_fail=False,
        fallback_heading=False,
        hybrid_heading=True,
        union_strategy="nms",
    )
    assert seen.get("search_called") is True
    # MA arm uses internal nms; final union gets the CLI strategy
    assert seen["score_kwargs"]["union_strategy"] == "nms"
    assert seen["score_kwargs"]["seed_hyps"] is None
    assert seen["score_kwargs"]["max_keep"] == 16
    assert seen["union_kwargs"]["strategy"] == "nms"
    assert seen["union_kwargs"]["max_keep"] == 16


def test_hybrid_a_arm_calls_full_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Dual arm calls search_photo_consistent_planes when hybrid is on."""
    import logging

    import ps1_hood.reconstruct.photo_planes as pp
    import ps1_hood.reconstruct.planarize as pl

    xyz = _synthetic_street_cloud(n_wall=400, n_ground=200)
    ply = tmp_path / "cloud.ply"
    _write_xyz_ply(ply, xyz)
    frames = []
    rng = np.random.default_rng(11)
    for i in range(2):
        img = rng.integers(0, 255, (80, 100, 3), dtype=np.uint8)
        p = tmp_path / f"da{i}.jpg"
        cv2.imwrite(str(p), img)
        frames.append(
            {
                "path": str(p),
                "e": 0.0,
                "n": float(i),
                "u": 2.0,
                "heading": 90.0,
                "pitch": 0.0,
                "fov": 90.0,
                "pano_id": f"da{i}",
            }
        )

    called = {"search": 0, "score_seed": "unset"}

    def _fake_planes(ply_path, cam_c, **kwargs):  # noqa: ANN001
        return (
            [
                {
                    "n": np.array([1.0, 0.0, 0.0]),
                    "d": -5.0,
                    "center": np.array([5.0, 0.0, 3.0]),
                    "width_m": 8.0,
                    "height_m": 6.0,
                    "source": "ma_segment",
                }
            ],
            {"z": 0.0},
            10,
        )

    def _fake_search(*a, **k):  # noqa: ANN001
        called["search"] += 1
        return [
            {
                "n": np.array([1.0, 0.0, 0.0]),
                "d": -8.0,
                "center": np.array([8.0, 20.0, 4.0]),
                "width_m": 8.0,
                "height_m": 9.0,
                "zncc": 0.42,
                "ok": True,
                "source": "heading_distance",
            }
        ]

    def _fake_score(hyps, frames, **kwargs):  # noqa: ANN001
        called["score_seed"] = kwargs.get("seed_hyps")
        return [
            {
                "n": np.array([1.0, 0.0, 0.0]),
                "d": -5.0,
                "center": np.array([5.0, 0.0, 3.0]),
                "width_m": 8.0,
                "height_m": 6.0,
                "zncc": 0.50,
                "ok": True,
                "source": "ma_segment",
                "count": 100,
            }
        ]

    monkeypatch.setattr(pl, "planes_from_mapanything_ply", _fake_planes)
    monkeypatch.setattr(pp, "search_photo_consistent_planes", _fake_search)
    monkeypatch.setattr(pl, "score_planar_hyps", _fake_score)

    with caplog.at_level(logging.INFO, logger="ps1_hood.reconstruct.facades"):
        extract_facades(
            ply,
            tmp_path / "facades.obj",
            frames=frames,
            n_planes=16,
            zncc_accept=0.35,
            planarize=True,
            keep_previous_on_fail=False,
            fallback_heading=False,
            hybrid_heading=True,
            hybrid_a_full_search=True,
        )
    assert called["search"] == 1
    assert called["score_seed"] is None
    joined = " ".join(r.message for r in caplog.records)
    assert "dual_arm" in joined
    assert "search_photo_consistent_planes" in joined


def test_union_a_priority_after_dual_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A accepts from full search survive a_priority when MA overlaps."""
    import ps1_hood.reconstruct.photo_planes as pp
    import ps1_hood.reconstruct.planarize as pl

    xyz = _synthetic_street_cloud(n_wall=400, n_ground=200)
    ply = tmp_path / "cloud.ply"
    _write_xyz_ply(ply, xyz)
    frames = []
    rng = np.random.default_rng(13)
    for i in range(2):
        img = rng.integers(0, 255, (80, 100, 3), dtype=np.uint8)
        p = tmp_path / f"ua{i}.jpg"
        cv2.imwrite(str(p), img)
        frames.append(
            {
                "path": str(p),
                "e": 0.0,
                "n": float(i),
                "u": 2.0,
                "heading": 90.0,
                "pitch": 0.0,
                "fov": 90.0,
                "pano_id": f"ua{i}",
            }
        )

    def _fake_planes(ply_path, cam_c, **kwargs):  # noqa: ANN001
        return ([], {"z": 0.0}, 0)

    def _fake_search(*a, **k):  # noqa: ANN001
        # Four well-separated A planes from full search
        return [
            _a_plane((8.0, float(i * 20)), zncc=0.40 + i * 0.01)
            for i in range(4)
        ]

    def _fake_score(hyps, frames, **kwargs):  # noqa: ANN001
        assert kwargs.get("seed_hyps") is None
        # Overlapping MA (higher ZNCC) + one far MA
        return [
            _plane((8.0, 1.0), zncc=0.55),  # overlaps A at y=0
            _plane((8.0, 200.0), zncc=0.45),
        ]

    monkeypatch.setattr(pl, "planes_from_mapanything_ply", _fake_planes)
    monkeypatch.setattr(pp, "search_photo_consistent_planes", _fake_search)
    monkeypatch.setattr(pl, "score_planar_hyps", _fake_score)

    # Spy real union to assert telemetry (extract imports from planarize each call)
    tel_out: dict = {}
    real_union = pl.union_keep_planes

    def _spy_union(accepted, **kwargs):  # noqa: ANN001
        result = real_union(accepted, **kwargs)
        if kwargs.get("telemetry") is not None:
            tel_out.update(kwargs["telemetry"])
        return result

    monkeypatch.setattr(pl, "union_keep_planes", _spy_union)

    meta = extract_facades(
        ply,
        tmp_path / "facades.obj",
        frames=frames,
        n_planes=16,
        zncc_accept=0.35,
        planarize=True,
        keep_previous_on_fail=False,
        fallback_heading=False,
        hybrid_heading=True,
        hybrid_a_full_search=True,
        union_strategy="a_priority",
    )
    assert tel_out.get("a_kept") == 4
    assert tel_out.get("ma_added") == 1
    assert tel_out.get("union_kept") == 5
    assert tel_out.get("a_pre_nms") == 4
    assert meta.get("source") == "mapanything_hybrid"
    assert int(meta.get("planes") or 0) == 5


def test_hybrid_legacy_seeds_opt_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--no-hybrid-a-full-search restores seed inject into score_planar_hyps."""
    import ps1_hood.reconstruct.photo_planes as pp
    import ps1_hood.reconstruct.planarize as pl

    xyz = _synthetic_street_cloud(n_wall=400, n_ground=200)
    ply = tmp_path / "cloud.ply"
    _write_xyz_ply(ply, xyz)
    frames = []
    rng = np.random.default_rng(17)
    for i in range(2):
        img = rng.integers(0, 255, (80, 100, 3), dtype=np.uint8)
        p = tmp_path / f"lg{i}.jpg"
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
                "pano_id": f"lg{i}",
            }
        )

    called = {"search": 0, "seed_hyps": None, "hypothesize": 0}

    def _spy_hyp(*a, **k):  # noqa: ANN001
        called["hypothesize"] += 1
        return [
            {
                "n": np.array([1.0, 0.0, 0.0]),
                "d": -8.0,
                "center": np.array([8.0, 10.0, 4.0]),
                "width_m": 8.0,
                "height_m": 9.0,
                "source": "heading_distance",
            }
        ]

    def _fake_search(*a, **k):  # noqa: ANN001
        called["search"] += 1
        return []

    def _score(hyps, frames, **kwargs):  # noqa: ANN001
        called["seed_hyps"] = kwargs.get("seed_hyps")
        return [
            {
                "n": np.array([1.0, 0.0, 0.0]),
                "d": -5.0,
                "center": np.array([5.0, 0.0, 3.0]),
                "width_m": 8.0,
                "height_m": 6.0,
                "zncc": 0.439,
                "ok": True,
                "source": "ma_segment",
                "count": 100,
            }
        ]

    monkeypatch.setattr(pp, "hypothesize_vertical_planes", _spy_hyp)
    monkeypatch.setattr(pp, "search_photo_consistent_planes", _fake_search)
    monkeypatch.setattr(pl, "score_planar_hyps", _score)

    extract_facades(
        ply,
        tmp_path / "facades.obj",
        frames=frames,
        n_planes=8,
        zncc_accept=0.35,
        planarize=True,
        voxel_m=0.15,
        plane_dist_m=0.10,
        keep_previous_on_fail=False,
        fallback_heading=False,
        hybrid_heading=True,
        hybrid_a_full_search=False,
    )
    assert called["search"] == 0  # no dual-arm A search
    assert called["hypothesize"] >= 1
    assert called["seed_hyps"] is not None
    assert len(called["seed_hyps"]) >= 1


def test_resolve_a_and_ma_ply_are_distinct(tmp_path: Path) -> None:
    """Dual-source: A prefers cloud_flow.ply; MA prefers mapanything/cloud.ply."""
    ma_ply = tmp_path / "mapanything" / "cloud.ply"
    flow_ply = tmp_path / "recon" / "cloud_flow.ply"
    recon_ply = tmp_path / "recon" / "cloud.ply"
    _write_xyz_ply(ma_ply, np.zeros((8, 3)))
    _write_xyz_ply(flow_ply, np.ones((8, 3)))
    _write_xyz_ply(recon_ply, np.full((8, 3), 2.0))
    a = resolve_a_ply(tmp_path, "flow")
    ma = resolve_ma_ply(tmp_path, "mapanything")
    assert a == flow_ply
    assert ma == ma_ply
    assert a != ma
    assert resolve_a_ply(tmp_path, "product") is None


def test_resolve_a_ply_falls_back_to_recon_cloud(tmp_path: Path) -> None:
    recon_ply = tmp_path / "recon" / "cloud.ply"
    _write_xyz_ply(recon_ply, np.zeros((4, 3)))
    assert resolve_a_ply(tmp_path, "flow") == recon_ply


def test_dual_source_a_uses_recon_cloud_not_ma(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A search sees flow xyz; MA peel sees mapanything path — not one resolve."""
    import ps1_hood.reconstruct.photo_planes as pp
    import ps1_hood.reconstruct.planarize as pl

    ma_dir = tmp_path / "mapanything"
    recon = tmp_path / "recon"
    ma_xyz = np.column_stack(
        [np.full(120, 100.0), np.linspace(0, 10, 120), np.linspace(0, 4, 120)]
    )
    flow_xyz = np.column_stack(
        [np.full(80, 1.0), np.linspace(0, 10, 80), np.linspace(0, 3, 80)]
    )
    ma_ply = ma_dir / "cloud.ply"
    flow_ply = recon / "cloud_flow.ply"
    _write_xyz_ply(ma_ply, ma_xyz)
    _write_xyz_ply(flow_ply, flow_xyz)

    frames = []
    rng = np.random.default_rng(21)
    for i in range(2):
        img = rng.integers(0, 255, (80, 100, 3), dtype=np.uint8)
        pth = tmp_path / f"ds{i}.jpg"
        cv2.imwrite(str(pth), img)
        frames.append(
            {
                "path": str(pth),
                "e": 0.0,
                "n": float(i),
                "u": 2.0,
                "heading": 90.0,
                "pitch": 0.0,
                "fov": 90.0,
                "pano_id": f"ds{i}",
            }
        )

    captured: dict = {}

    def _fake_planes(ply_path, cam_c, **kwargs):  # noqa: ANN001
        captured["ma_ply"] = Path(ply_path)
        captured["peel_xyz_max_x"] = float(np.asarray(kwargs.get("xyz")).max()) if kwargs.get("xyz") is not None else None
        return ([], {"z": 0.0}, 0)

    def _fake_search(frames_in, xyz=None, **kwargs):  # noqa: ANN001
        captured["a_xyz"] = None if xyz is None else np.asarray(xyz)
        captured["a_ground_z"] = kwargs.get("ground_z")
        return [
            _a_plane((8.0, 0.0), zncc=0.42),
        ]

    def _fake_score(hyps, frames_in, **kwargs):  # noqa: ANN001
        captured["score_seed"] = kwargs.get("seed_hyps")
        return []

    monkeypatch.setattr(pl, "planes_from_mapanything_ply", _fake_planes)
    monkeypatch.setattr(pp, "search_photo_consistent_planes", _fake_search)
    monkeypatch.setattr(pl, "score_planar_hyps", _fake_score)

    meta = extract_facades(
        ma_ply,
        recon / "facades.obj",
        frames=frames,
        n_planes=16,
        zncc_accept=0.35,
        planarize=True,
        keep_previous_on_fail=False,
        fallback_heading=False,
        hybrid_heading=True,
        hybrid_a_full_search=True,
        a_ply_path=flow_ply,
        a_source="flow",
    )
    assert captured["ma_ply"] == ma_ply
    assert captured["score_seed"] is None
    assert captured["a_xyz"] is not None
    assert float(captured["a_xyz"][:, 0].max()) < 20.0  # flow x≈1, not MA x=100
    assert meta.get("a_ply") == str(flow_ply)
    assert meta.get("ma_ply") == str(ma_ply)
    assert meta.get("a_kept") == 1


def test_control_does_not_overwrite_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A-only / --no-planarize writes *.control and leaves *.candidate intact."""
    import json

    import ps1_hood.reconstruct.photo_planes as pp

    dest = tmp_path / "facades.obj"
    tex_dir = tmp_path / "textures"
    tex_dir.mkdir()
    dest.write_text(
        "\n".join(
            ["# product", "mtllib facades.mtl"]
            + [f"v {i} 0 0" for i in range(16)]
            + ["usemtl ground", "f 1 2 3 4"]
            + ["usemtl facade_00", "f 5 6 7 8"]
            + ["# pad"] * 40
        )
        + "\n",
        encoding="ascii",
    )
    (tmp_path / "facades.mtl").write_text("newmtl facade_00\n", encoding="ascii")
    planes = [
        {
            "id": f"facade_{i:02d}",
            "n": [1.0, 0.0, 0.0],
            "d": -5.0,
            "zncc": 0.42,
            "texture": f"textures/facade_{i:02d}.jpg" if i < 5 else None,
            "width_m": 8.0,
            "height_m": 6.0,
            "source": "photo_consistency",
        }
        for i in range(7)
    ]
    (tmp_path / "planes.json").write_text(
        json.dumps(
            {
                "frame": "ENU",
                "source": "photo_consistency",
                "ground_z": 0.0,
                "planes": planes,
                "residual_points": 0,
                "gates": {"mean_zncc": 0.42, "plane_count": 7},
            }
        ),
        encoding="utf-8",
    )
    for i in range(5):
        cv2.imwrite(
            str(tex_dir / f"facade_{i:02d}.jpg"),
            np.full((16, 16, 3), 30 + i, dtype=np.uint8),
        )

    cand_obj = tmp_path / "facades.candidate.obj"
    cand_json = tmp_path / "planes.candidate.json"
    cand_body = "# hybrid candidate - must survive control rerun\n"
    cand_obj.write_text(cand_body, encoding="ascii")
    cand_json.write_text('{"source":"mapanything_hybrid","planes":[1,2,3,4]}\n', encoding="utf-8")
    (tex_dir / "candidate").mkdir()
    cv2.imwrite(str(tex_dir / "candidate" / "facade_00.jpg"), np.full((8, 8, 3), 9, dtype=np.uint8))
    prior_cand_obj = cand_obj.read_bytes()
    prior_cand_json = cand_json.read_text(encoding="utf-8")
    prior_product = dest.read_bytes()
    prior_planes = (tmp_path / "planes.json").read_text(encoding="utf-8")

    xyz = _synthetic_street_cloud(n_wall=80, n_ground=40)
    ply = tmp_path / "cloud.ply"
    _write_xyz_ply(ply, xyz)
    frames = []
    rng = np.random.default_rng(23)
    for i in range(2):
        img = rng.integers(0, 255, (60, 80, 3), dtype=np.uint8)
        pth = tmp_path / f"ctl{i}.jpg"
        cv2.imwrite(str(pth), img)
        frames.append(
            {
                "path": str(pth),
                "e": float(i),
                "n": 0.0,
                "u": 2.0,
                "heading": 90.0,
                "pitch": 0.0,
                "fov": 90.0,
                "pano_id": f"ctl{i}",
            }
        )

    def _ctrl_search(*a, **k):  # noqa: ANN001
        return [_a_plane((8.0, 0.0), zncc=0.37), _a_plane((8.0, 20.0), zncc=0.36)]

    monkeypatch.setattr(pp, "search_photo_consistent_planes", _ctrl_search)

    meta = extract_facades(
        ply,
        dest,
        frames=frames,
        n_planes=8,
        zncc_accept=0.35,
        planarize=False,
        keep_previous_on_fail=True,
        fallback_heading=False,
        a_source="flow",
        control_out=False,
    )
    assert meta.get("output_kind") == "control"
    assert (tmp_path / "facades.control.obj").is_file()
    assert (tmp_path / "planes.control.json").is_file()
    assert cand_obj.read_bytes() == prior_cand_obj
    assert cand_json.read_text(encoding="utf-8") == prior_cand_json
    assert dest.read_bytes() == prior_product
    assert (tmp_path / "planes.json").read_text(encoding="utf-8") == prior_planes
    assert (tex_dir / "candidate" / "facade_00.jpg").is_file()


def test_candidate_survives_no_planarize_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explicit --control-out never writes facades.candidate.*."""
    import ps1_hood.reconstruct.photo_planes as pp

    dest = tmp_path / "facades.obj"
    dest.write_text("# empty start\n", encoding="ascii")
    xyz = _synthetic_street_cloud(n_wall=40, n_ground=20)
    ply = tmp_path / "cloud.ply"
    _write_xyz_ply(ply, xyz)
    frames = []
    rng = np.random.default_rng(29)
    for i in range(2):
        img = rng.integers(0, 255, (40, 50, 3), dtype=np.uint8)
        pth = tmp_path / f"np{i}.jpg"
        cv2.imwrite(str(pth), img)
        frames.append(
            {
                "path": str(pth),
                "e": float(i),
                "n": 0.0,
                "u": 2.0,
                "heading": 90.0,
                "pitch": 0.0,
                "fov": 90.0,
                "pano_id": f"np{i}",
            }
        )

    monkeypatch.setattr(
        pp,
        "search_photo_consistent_planes",
        lambda *a, **k: [_a_plane((8.0, 0.0), zncc=0.40)],
    )
    meta = extract_facades(
        ply,
        dest,
        frames=frames,
        planarize=False,
        keep_previous_on_fail=False,
        control_out=True,
        a_source="flow",
    )
    assert meta.get("output_kind") == "control"
    assert (tmp_path / "facades.control.obj").is_file()
    assert not (tmp_path / "facades.candidate.obj").exists()
    assert (tmp_path / "planes.control.json").is_file()
    assert not (tmp_path / "planes.candidate.json").exists()


def test_planes_from_product_json_is_a_family(tmp_path: Path) -> None:
    import json

    payload = {
        "planes": [
            {
                "n": [1.0, 0.0, 0.0],
                "d": -8.0,
                "quad": [[8, 0, 0], [8, 8, 0], [8, 8, 9], [8, 0, 9]],
                "width_m": 8.0,
                "height_m": 9.0,
                "zncc": 0.42,
            }
        ]
    }
    path = tmp_path / "planes.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    locked = planes_from_product_json(path)
    assert len(locked) == 1
    assert locked[0]["source"] == "product_lock"
    assert is_a_source(locked[0]["source"])
