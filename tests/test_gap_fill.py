"""PR-C façade gap fill: manhattan/corner seeds, road reject, quality-keep."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from ps1_hood.reconstruct.facades import _is_strictly_better, extract_facades
from ps1_hood.reconstruct.gap_fill import (
    GAP_ADD_MAX_ADDS,
    GAP_ADD_MIN_VIEWS,
    GAP_ADD_SAT_EDGE_M,
    GAP_FILL_MAX_KEEP,
    GAP_FILL_PEEL_CAP,
    WORST_CAM_DISTANCES_M,
    WORST_CAM_YAWS_DEG,
    build_gap_fill_seeds,
    corner_seeds_from_roof_aabbs,
    count_untextured_product,
    estimate_travel_heading_deg,
    filter_gap_adds,
    filter_hyps_visible_in_cams,
    filter_ma_peels_for_gaps,
    find_frame_for_cam_id,
    gap_add_multiview_ok,
    load_worst_cam_ids_from_compare,
    manhattan_side_seeds,
    parse_cam_id,
    prefer_side_wall_order,
    reject_road_center_hyps,
    sat_aabb_edge_ok,
    worst_cam_seeds,
)
from ps1_hood.reconstruct.planarize import is_a_source, is_ma_source


def _frame(e: float, n: float, heading: float, *, travel: float | None = ..., i: int = 0) -> dict:
    fr = {
        "e": e,
        "n": n,
        "u": 2.0,
        "heading": heading,
        "pitch": 0.0,
        "fov": 90.0,
        "pano_id": f"p{i}",
        "path": None,
    }
    if travel is not ...:
        fr["travel_heading"] = travel
    return fr


def test_manhattan_side_seeds_perp_to_travel() -> None:
    frames = [
        _frame(0.0, 0.0, 90.0, travel=0.0, i=0),  # looking E, travel N
        _frame(0.0, 10.0, 90.0, travel=0.0, i=1),
    ]
    hyps = manhattan_side_seeds(frames, ground_z=0.0, distances_m=(10.0,))
    assert hyps
    assert all(h["source"] == "manhattan" for h in hyps)
    # Normals should be roughly E/W (±90 from travel N) → |n_y| small or |n_x| large
    assert any(abs(float(h["n"][0])) > 0.8 for h in hyps)


def test_corner_seeds_two_headings_per_corner() -> None:
    regions = [
        {
            "kind": "roof",
            "id": "roof_0",
            "aabb_enu": [(10.0, 10.0), (20.0, 10.0), (20.0, 18.0), (10.0, 18.0)],
        }
    ]
    hyps = corner_seeds_from_roof_aabbs(regions, ground_z=0.0, max_corners=4)
    assert len(hyps) == 8  # 4 corners × 2 headings
    assert all(h["source"] == "corner_sat" for h in hyps)
    assert is_a_source("corner_sat")
    assert not is_ma_source("corner_sat")


def test_reject_road_center_cam_corridor() -> None:
    cam_xy = np.array([[0.0, 0.0], [5.0, 0.0]], dtype=np.float64)
    hyps = [
        {
            "n": np.array([1.0, 0.0, 0.0]),
            "d": -10.0,
            "center": np.array([1.0, 0.0, 4.0]),  # on road
            "width_m": 8.0,
            "height_m": 9.0,
            "source": "ma_segment",
        },
        {
            "n": np.array([1.0, 0.0, 0.0]),
            "d": -10.0,
            "center": np.array([10.0, 8.0, 4.0]),  # off road
            "width_m": 8.0,
            "height_m": 9.0,
            "source": "ma_segment",
        },
    ]
    kept = reject_road_center_hyps(hyps, cam_xy=cam_xy, margin_m=3.0)
    assert len(kept) == 1
    assert float(kept[0]["center"][0]) == 10.0


def test_prefer_side_wall_order() -> None:
    # travel +N (0°); side wall normal +E should rank above travel-aligned
    hyps = [
        {
            "n": np.array([0.0, 1.0, 0.0]),
            "d": 0.0,
            "center": np.zeros(3),
            "width_m": 8.0,
            "height_m": 9.0,
            "source": "ma_segment",
        },
        {
            "n": np.array([1.0, 0.0, 0.0]),
            "d": 0.0,
            "center": np.zeros(3),
            "width_m": 8.0,
            "height_m": 9.0,
            "source": "ma_segment",
        },
    ]
    ordered = prefer_side_wall_order(hyps, 0.0)
    assert abs(float(ordered[0]["n"][0])) > 0.8


def test_filter_ma_peels_cap() -> None:
    frames = [_frame(0.0, float(i), 90.0, travel=0.0, i=i) for i in range(3)]
    hyps = []
    for i in range(40):
        hyps.append(
            {
                "n": np.array([1.0, 0.0, 0.0]),
                "d": -20.0,
                "center": np.array([20.0, float(i), 4.0]),
                "width_m": 8.0,
                "height_m": 9.0,
                "source": "ma_segment",
            }
        )
    out = filter_ma_peels_for_gaps(hyps, frames, peel_cap=GAP_FILL_PEEL_CAP)
    assert len(out) <= GAP_FILL_PEEL_CAP


def test_count_untextured_product(tmp_path: Path) -> None:
    dest = tmp_path / "facades.obj"
    dest.write_text("# empty\n", encoding="utf-8")
    planes = {
        "planes": [
            {"id": "facade_00", "texture": "textures/facade_00.jpg", "zncc": 0.4},
            {"id": "facade_01", "texture": None, "zncc": 0.3},
            {"id": "facade_02", "zncc": 0.35},
        ]
    }
    (tmp_path / "planes.json").write_text(json.dumps(planes), encoding="utf-8")
    tex = tmp_path / "textures"
    tex.mkdir()
    cv2.imwrite(str(tex / "facade_00.jpg"), np.full((8, 8, 3), 40, dtype=np.uint8))
    n_planes, n_untex = count_untextured_product(dest)
    assert n_planes == 3
    assert n_untex == 2


def test_quality_keep_gap_fill_bar() -> None:
    """Promote only if textured≥8 and planes≥10 under PR #24 (or beat mean)."""
    prev = {"textured": 8, "plane_count": 10, "mean_zncc": 0.418}
    # Weaker: fewer planes — must NOT promote
    ok, why = _is_strictly_better(
        {"textured": 8, "plane_count": 9, "mean_zncc": 0.45}, prev
    )
    assert not ok
    # More textured, planes≥, mean within eps — promote
    ok2, why2 = _is_strictly_better(
        {"textured": 9, "plane_count": 11, "mean_zncc": 0.41}, prev
    )
    assert ok2
    assert "clause1" in why2
    # Same textured, more planes, mean ok — promote
    ok3, _ = _is_strictly_better(
        {"textured": 8, "plane_count": 12, "mean_zncc": 0.41}, prev
    )
    assert ok3


def test_estimate_travel_heading() -> None:
    frames = [
        _frame(0.0, 0.0, 90.0, travel=None, i=0),
        _frame(0.0, 20.0, 90.0, travel=None, i=1),
    ]
    h = estimate_travel_heading_deg(frames)
    assert h is not None
    # Mostly +N
    assert abs(h) < 15.0 or abs(h - 360.0) < 15.0


def test_gap_fill_extract_keeps_product_on_weak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gap-fill that yields weaker union must write *.candidate and keep 10/8."""
    # Seed 10-plane / 8-textured product
    dest = tmp_path / "facades.obj"
    dest.write_text("mtllib facades.mtl\n", encoding="utf-8")
    (tmp_path / "facades.mtl").write_text("newmtl ground\n", encoding="utf-8")
    tex = tmp_path / "textures"
    tex.mkdir()
    planes = []
    for i in range(10):
        planes.append(
            {
                "id": f"facade_{i:02d}",
                "n": [1.0, 0.0, 0.0],
                "d": -5.0 - 0.1 * i,
                "quad": [
                    [5.0, float(i), 0.0],
                    [5.0, float(i) + 4.0, 0.0],
                    [5.0, float(i) + 4.0, 8.0],
                    [5.0, float(i), 8.0],
                ],
                "width_m": 4.0,
                "height_m": 8.0,
                "zncc": 0.42,
                "texture": f"textures/facade_{i:02d}.jpg" if i < 8 else None,
                "source": "product_lock",
            }
        )
        if i < 8:
            cv2.imwrite(
                str(tex / f"facade_{i:02d}.jpg"),
                np.full((16, 16, 3), 50 + i, dtype=np.uint8),
            )
    (tmp_path / "planes.json").write_text(
        json.dumps(
            {
                "source": "mapanything_hybrid",
                "ground_z": 0.0,
                "planes": planes,
                "gates": {"mean_zncc": 0.418, "plane_count": 10},
            }
        ),
        encoding="utf-8",
    )

    # Minimal frames (no images → scoring yields nothing new)
    frames = [
        _frame(0.0, 0.0, 90.0, travel=0.0, i=0),
        _frame(0.0, 8.0, 90.0, travel=0.0, i=1),
    ]

    # Dense-enough fake MA ply so planarize path engages
    xyz = np.random.default_rng(0).normal(size=(60_000, 3))
    xyz[:, 2] = np.abs(xyz[:, 2]) * 3
    ply = tmp_path / "cloud.ply"
    with ply.open("w", encoding="ascii") as fh:
        fh.write("ply\nformat ascii 1.0\n")
        fh.write(f"element vertex {len(xyz)}\n")
        fh.write("property float x\nproperty float y\nproperty float z\n")
        fh.write("end_header\n")
        for p in xyz:
            fh.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")

    # Stub peels + scoring so gap path runs but returns only a weak 2-plane set
    weak = [
        {
            "n": np.array([1.0, 0.0, 0.0]),
            "d": -5.0,
            "center": np.array([5.0, 2.0, 4.0]),
            "width_m": 4.0,
            "height_m": 8.0,
            "zncc": 0.30,
            "ok": True,
            "source": "ma_segment",
            "corners": np.array(
                [[5, 0, 0], [5, 4, 0], [5, 4, 8], [5, 0, 8]], dtype=np.float64
            ),
        }
    ]

    with (
        patch(
            "ps1_hood.reconstruct.planarize.planes_from_mapanything_ply",
            return_value=(weak, {"z": 0.0}, 0),
        ),
        patch(
            "ps1_hood.reconstruct.planarize.score_planar_hyps",
            return_value=weak,
        ),
        patch(
            "ps1_hood.reconstruct.gap_fill.build_gap_fill_seeds",
            return_value=[],
        ),
        patch(
            "ps1_hood.reconstruct.gap_fill.filter_ma_peels_for_gaps",
            side_effect=lambda hyps, *a, **k: hyps,
        ),
    ):
        meta = extract_facades(
            ply,
            dest,
            n_planes=16,
            frames=frames,
            planarize=True,
            keep_previous_on_fail=True,
            fallback_heading=False,
            hybrid_heading=True,
            hybrid_a_full_search=True,
            gap_fill=True,
            project_root=tmp_path,
            ps1_rectify=False,
            ps1_tex_size=None,
        )

    # Product preserved (10 planes); candidate may exist if bake produced something
    product = json.loads((tmp_path / "planes.json").read_text(encoding="utf-8"))
    assert len(product["planes"]) == 10
    assert meta.get("preserved_previous") or meta.get("planes") == 10


def test_parse_cam_id_heading_suffix() -> None:
    pano, h = parse_cam_id("1-hH0xS8V_656nPgqQnl6A_h000")
    assert pano == "1-hH0xS8V_656nPgqQnl6A"
    assert h == 0.0
    pano2, h2 = parse_cam_id("Vd7vlCY1OAUkLlszrnLbDg_h120")
    assert pano2 == "Vd7vlCY1OAUkLlszrnLbDg"
    assert h2 == 120.0
    pano3, h3 = parse_cam_id("1g2FRL3bCGcwgPO48E70-Q_h180")
    assert h3 == 180.0
    with pytest.raises(ValueError):
        parse_cam_id("no_suffix")


def test_worst_cam_seeds_rings_and_yaws() -> None:
    frames = [
        _frame(0.0, 0.0, 0.0, travel=90.0, i=0),  # looking N
    ]
    frames[0]["pano_id"] = "1-hH0xS8V_656nPgqQnl6A"
    cam_id = "1-hH0xS8V_656nPgqQnl6A_h000"
    hyps = worst_cam_seeds(frames, [cam_id], ground_z=0.0)
    assert hyps
    assert all(h["source"] == "worst_cam" for h in hyps)
    assert all(h.get("pano_id") == "1-hH0xS8V_656nPgqQnl6A" for h in hyps)
    # 4 distances × 5 yaws = 20 (under cap)
    assert len(hyps) == len(WORST_CAM_DISTANCES_M) * len(WORST_CAM_YAWS_DEG)
    # Centers should sit ~8–20 m from cam along some yaw
    dists = [float(np.linalg.norm(h["center"][:2] - np.array([0.0, 0.0]))) for h in hyps]
    assert min(dists) >= 7.5
    assert max(dists) <= 20.5


def test_filter_hyps_visible_in_worst_cams() -> None:
    # Cam looking +E (heading 90); plane facing cam (normal +W toward cam from +E)
    cam = _frame(0.0, 0.0, 90.0, travel=0.0, i=0)
    visible = {
        "n": np.array([-1.0, 0.0, 0.0]),
        "d": 10.0,
        "center": np.array([10.0, 0.0, 4.0]),
        "width_m": 8.0,
        "height_m": 9.0,
        "source": "manhattan",
    }
    behind = {
        "n": np.array([1.0, 0.0, 0.0]),
        "d": -10.0,
        "center": np.array([-10.0, 0.0, 4.0]),
        "width_m": 8.0,
        "height_m": 9.0,
        "source": "manhattan",
    }
    kept = filter_hyps_visible_in_cams([visible, behind], [cam], min_frontal=0.25)
    assert len(kept) == 1
    assert float(kept[0]["center"][0]) == 10.0


def test_find_frame_for_cam_id() -> None:
    frames = [
        _frame(1.0, 2.0, 120.0, travel=0.0, i=0),
        _frame(3.0, 4.0, 0.0, travel=0.0, i=1),
    ]
    frames[0]["pano_id"] = "Vd7vlCY1OAUkLlszrnLbDg"
    frames[1]["pano_id"] = "other"
    fr = find_frame_for_cam_id(frames, "Vd7vlCY1OAUkLlszrnLbDg_h120")
    assert fr is not None
    assert float(fr["e"]) == 1.0


def test_load_worst_from_compare(tmp_path: Path) -> None:
    compare = tmp_path / "recon" / "compare"
    compare.mkdir(parents=True)
    (compare / "summary.json").write_text(
        json.dumps(
            {
                "worst": [
                    {"id": "a_h000", "zncc_mean": -0.1},
                    {"id": "b_h120", "zncc_mean": -0.05},
                    {"id": "c_h180", "zncc_mean": -0.01},
                ]
            }
        ),
        encoding="utf-8",
    )
    ids = load_worst_cam_ids_from_compare(tmp_path, n=2)
    assert ids == ["a_h000", "b_h120"]


def test_build_gap_fill_seeds_worst_cam_filter(tmp_path: Path) -> None:
    frames = [
        _frame(0.0, 0.0, 0.0, travel=90.0, i=0),
        _frame(5.0, 0.0, 0.0, travel=90.0, i=1),
    ]
    frames[0]["pano_id"] = "camA"
    frames[1]["pano_id"] = "camB"
    with patch(
        "ps1_hood.reconstruct.gap_fill.load_sat_roof_regions", return_value=[]
    ):
        seeds = build_gap_fill_seeds(
            frames,
            tmp_path,
            ground_z=0.0,
            worst_cam_ids=["camA_h000"],
        )
    assert any(h["source"] == "worst_cam" for h in seeds)
    # manhattan (if any remain) must be visible to camA
    man = [h for h in seeds if h["source"] == "manhattan"]
    if man:
        kept = filter_hyps_visible_in_cams(man, [frames[0]], min_frontal=0.25)
        assert len(kept) == len(man)


def test_gap_fill_max_keep_24() -> None:
    assert GAP_FILL_MAX_KEEP == 24


def test_worst_cam_not_a_family() -> None:
    assert not is_a_source("worst_cam")
    assert not is_a_source("ma_gap_worst_cam")
    assert not is_a_source("ma_gap_manhattan")
    assert is_a_source("product_lock")
    assert is_a_source("manhattan")
    assert is_ma_source("ma_gap_worst_cam")


def test_quality_keep_18_18_bar() -> None:
    """Promote if planes↑ or textured≥18 with mean not ↓>0.02."""
    prev = {"textured": 18, "plane_count": 18, "mean_zncc": 0.138}
    # More planes, mean within eps — promote (clause2)
    ok, why = _is_strictly_better(
        {"textured": 18, "plane_count": 20, "mean_zncc": 0.130}, prev
    )
    assert ok
    assert "clause2" in why
    # Mean regress >0.02 with more planes — reject
    ok2, _ = _is_strictly_better(
        {"textured": 18, "plane_count": 20, "mean_zncc": 0.100}, prev
    )
    assert not ok2
    # Fewer textured — never promote
    ok3, _ = _is_strictly_better(
        {"textured": 17, "plane_count": 20, "mean_zncc": 0.20}, prev
    )
    assert not ok3


def test_cli_worst_cams_options() -> None:
    from click.testing import CliRunner

    from ps1_hood.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["facades", "--help"])
    assert result.exit_code == 0
    assert "--worst-cams" in result.output
    assert "--worst-from-compare" in result.output
    assert "--gap-fill" in result.output
    assert "--max-gap-adds" in result.output
    assert "--sat-aabb-gate" in result.output
    assert "--ma-peel-cap" in result.output
    assert "--gap-min-views" in result.output
    assert "--gap-seeds" in result.output


def test_gap_fill_peel_cap_default_3() -> None:
    assert GAP_FILL_PEEL_CAP == 3
    assert GAP_ADD_MAX_ADDS == 3
    assert GAP_ADD_MIN_VIEWS == 2
    assert GAP_ADD_SAT_EDGE_M == 2.0


def test_sat_aabb_edge_gate_on_boundary() -> None:
    regions = [
        {
            "kind": "roof",
            "aabb_enu": [(10.0, 10.0), (20.0, 10.0), (20.0, 18.0), (10.0, 18.0)],
        }
    ]
    # Center on south edge (n=10), normal +N → aligns with south-edge outward (-N)? 
    # South edge tangent = +E; outward from interior = -N = (0,-1).
    # Plane facing south (normal 0,-1) → |n·n_edge|=1.
    center = np.array([15.0, 10.0, 4.0])
    n_out = np.array([0.0, -1.0, 0.0])
    ok, why = sat_aabb_edge_ok(center, n_out, regions, max_edge_m=2.0)
    assert ok, why
    # Far from any roof
    far = np.array([50.0, 50.0, 4.0])
    ok2, why2 = sat_aabb_edge_ok(far, n_out, regions, max_edge_m=2.0)
    assert not ok2
    assert "sat_aabb_edge" in why2
    # Soft-pass with no roofs
    ok3, why3 = sat_aabb_edge_ok(far, n_out, [], max_edge_m=2.0)
    assert ok3
    assert "soft_pass" in why3


def test_sat_aabb_rejects_misaligned_normal() -> None:
    regions = [
        {
            "kind": "roof",
            "aabb_enu": [(0.0, 0.0), (10.0, 0.0), (10.0, 8.0), (0.0, 8.0)],
        }
    ]
    center = np.array([5.0, 0.0, 4.0])  # on south edge
    n_parallel_to_edge = np.array([1.0, 0.0, 0.0])  # along +E = tangent, not outward
    ok, why = sat_aabb_edge_ok(center, n_parallel_to_edge, regions)
    assert not ok
    assert "align" in why


def test_gap_add_multiview_requires_two_cams() -> None:
    frames = [
        _frame(0.0, 0.0, 90.0, travel=0.0, i=0),
        _frame(0.0, 5.0, 90.0, travel=0.0, i=1),
        _frame(0.0, 10.0, 90.0, travel=0.0, i=2),
    ]
    # Plane facing -E (toward cams looking +E): n=(-1,0), cam_fwd=(1,0) → |n·fwd|=1
    good = {
        "n": np.array([-1.0, 0.0, 0.0]),
        "center": np.array([10.0, 5.0, 4.0]),
        "zncc": 0.42,
        "scores": [0.45, 0.38],
        "view_indices": [0, 1, 2],
        "source": "ma_segment",
    }
    ok, why = gap_add_multiview_ok(good, frames)
    assert ok, why
    # Only one weak source score + low mean → fail
    weak = {
        **good,
        "zncc": 0.20,
        "scores": [0.20],
    }
    ok2, why2 = gap_add_multiview_ok(weak, frames)
    assert not ok2
    # Grazing: plane normal nearly ⟂ cam forward (n=+N, cams look +E)
    grazing = {
        "n": np.array([0.0, 1.0, 0.0]),
        "center": np.array([10.0, 5.0, 4.0]),
        "zncc": 0.50,
        "scores": [0.50, 0.48],
        "view_indices": [0, 1],
        "source": "manhattan",
    }
    ok3, why3 = gap_add_multiview_ok(grazing, frames)
    assert not ok3
    assert "grazing" in why3
    # Low median
    low_med = {
        **good,
        "zncc": 0.36,
        "scores": [0.70, -0.40, -0.35],  # median < 0.10
    }
    ok4, why4 = gap_add_multiview_ok(low_med, frames)
    assert not ok4
    assert "median" in why4


def test_filter_gap_adds_cap_and_prefer_non_ma() -> None:
    frames = [
        _frame(0.0, 0.0, 90.0, travel=0.0, i=0),
        _frame(0.0, 5.0, 90.0, travel=0.0, i=1),
        _frame(0.0, 10.0, 90.0, travel=0.0, i=2),
    ]
    regions = [
        {
            "kind": "roof",
            "aabb_enu": [(8.0, 0.0), (12.0, 0.0), (12.0, 20.0), (8.0, 20.0)],
        }
    ]
    def _pl(src: str, z: float, n_y: float = 0.0) -> dict:
        return {
            "n": np.array([-1.0, n_y, 0.0]),
            "center": np.array([8.0, 5.0 + z, 4.0]),  # on west roof edge
            "zncc": z,
            "scores": [z, z - 0.02],
            "view_indices": [0, 1, 2],
            "source": src,
        }
    planes = [
        _pl("ma_segment", 0.50),
        _pl("ma_gap_manhattan", 0.45),
        _pl("ma_gap_worst_cam", 0.40),
        _pl("ma_segment", 0.55),
    ]
    kept = filter_gap_adds(
        planes,
        frames,
        roof_regions=regions,
        sat_aabb_gate_m=2.0,
        max_gap_adds=3,
    )
    assert len(kept) <= 3
    # Non-ma_segment preferred in ranking — first slots should not all be ma_segment
    # With cap 3 and 2 non-ma + 2 ma, expect both non-ma kept
    srcs = [p["source"] for p in kept]
    assert "ma_gap_manhattan" in srcs
    assert "ma_gap_worst_cam" in srcs
