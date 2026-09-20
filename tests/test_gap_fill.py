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
    SAT_EDGE_COVER_CENTER_M,
    SAT_EDGE_MAX_DRIFT_M,
    SAT_EDGE_MAX_EDGES,
    WORST_CAM_DISTANCES_M,
    WORST_CAM_YAWS_DEG,
    aabb_boundary_edges,
    build_gap_fill_seeds,
    closest_point_on_segment_xy,
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
    product_plane_covers_edge,
    reanchor_sat_edge_after_score,
    reject_road_center_hyps,
    sat_aabb_edge_ok,
    sat_edge_seeds,
    uncovered_roof_edges,
    worst_cam_seeds,
    write_gap_needs_json,
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
            gap_seeds_mode="legacy",
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
    assert not is_a_source("sat_edge")
    assert not is_a_source("ma_gap_worst_cam")
    assert not is_a_source("ma_gap_sat_edge")
    assert not is_a_source("ma_gap_manhattan")
    assert is_a_source("product_lock")
    assert is_a_source("manhattan")
    assert is_ma_source("ma_gap_worst_cam")
    assert is_ma_source("ma_gap_sat_edge")


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


def test_uncovered_roof_edges_vs_product() -> None:
    """West product plane covers west AABB edge; other three stay uncovered."""
    regions = [
        {
            "kind": "roof",
            "id": "roof_0",
            "aabb_enu": [(10.0, 10.0), (20.0, 10.0), (20.0, 18.0), (10.0, 18.0)],
        }
    ]
    # Plane on west face: n = -E, through e=10
    n = np.array([-1.0, 0.0, 0.0])
    center = np.array([10.0, 14.0, 3.5])
    d = float(-n @ center)
    product = [{"n": n, "d": d, "center": center, "source": "product_lock"}]
    edges = uncovered_roof_edges(regions, product, max_edges=8)
    assert len(edges) == 3
    # No west edge (n_out ≈ -E)
    for e in edges:
        assert float(e["n_out"][0]) > -0.5  # not strongly -E


def test_product_plane_covers_edge_align_and_dist() -> None:
    aabb = [(0.0, 0.0), (10.0, 0.0), (10.0, 8.0), (0.0, 8.0)]
    edges = aabb_boundary_edges(aabb, roof_id="r")
    # South edge: mid~(5,0), n_out~(0,-1)
    south = [e for e in edges if float(e["n_out"][1]) < -0.5][0]
    n = np.array([0.0, -1.0, 0.0])
    center = np.array([5.0, 0.0, 3.5])
    pl = {"n": n, "d": float(-n @ center), "center": center}
    assert product_plane_covers_edge(pl, south)
    # Far plane — not covering
    far = {
        "n": n,
        "d": float(-n @ np.array([5.0, -10.0, 3.5])),
        "center": np.array([5.0, -10.0, 3.5]),
    }
    assert not product_plane_covers_edge(far, south)


def test_sat_edge_seeds_facing_cam() -> None:
    """Cam east of building looking west → seed on east wall."""
    regions = [
        {
            "kind": "roof",
            "id": "garage",
            "aabb_enu": [(0.0, 0.0), (8.0, 0.0), (8.0, 6.0), (0.0, 6.0)],
        }
    ]
    # No product → all 4 edges uncovered (capped)
    # Cam east of east wall (e=8), looking west (heading 270)
    frames = [
        _frame(20.0, 3.0, 270.0, travel=0.0, i=0),  # 12 m east of mid~(8,3)
        _frame(22.0, 3.0, 270.0, travel=0.0, i=1),
    ]
    hyps, needs = sat_edge_seeds(frames, regions, product_planes=[], ground_z=0.0)
    assert hyps
    assert all(h["source"] == "sat_edge" for h in hyps)
    # At least one east-facing plane (n ≈ +E)
    assert any(float(h["n"][0]) > 0.7 for h in hyps)
    assert all("seed_cams" in h for h in hyps)


def test_sat_edge_needs_no_facing_cam(tmp_path: Path) -> None:
    """Zero facing cams → needs entry; write gap_needs.json; no MA invent."""
    regions = [
        {
            "kind": "roof",
            "id": "far_side",
            "aabb_enu": [(100.0, 100.0), (110.0, 100.0), (110.0, 108.0), (100.0, 108.0)],
        }
    ]
    # Cams far away / wrong heading
    frames = [_frame(0.0, 0.0, 90.0, travel=0.0, i=0)]
    hyps, needs = sat_edge_seeds(frames, regions, [], ground_z=0.0, max_edges=4)
    assert hyps == []
    assert needs
    assert all(n["reason"] == "no_facing_cam" for n in needs)
    out = write_gap_needs_json(
        tmp_path / "recon" / "gap_needs.json",
        needs,
        run="smoke-dense",
        product_planes=18,
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["product_planes"] == 18
    assert payload["run"] == "smoke-dense"
    assert payload["needs"]
    assert "fixed poses only" in payload["hint"]


def test_build_gap_fill_seeds_sat_edge_mode(tmp_path: Path) -> None:
    regions = [
        {
            "kind": "roof",
            "id": "r0",
            "aabb_enu": [(0.0, 0.0), (8.0, 0.0), (8.0, 6.0), (0.0, 6.0)],
        }
    ]
    frames = [
        _frame(20.0, 3.0, 270.0, travel=0.0, i=0),
        _frame(22.0, 3.0, 270.0, travel=0.0, i=1),
    ]
    with patch(
        "ps1_hood.reconstruct.gap_fill.load_sat_roof_regions",
        return_value=regions,
    ):
        seeds = build_gap_fill_seeds(
            frames,
            tmp_path,
            ground_z=0.0,
            gap_seeds_mode="sat-edge",
            product_planes=[],
        )
    assert seeds
    assert all(h["source"] == "sat_edge" for h in seeds)
    # No manhattan in sat-edge mode
    assert not any(h["source"] == "manhattan" for h in seeds)


def test_filter_gap_adds_prefer_sat_edge() -> None:
    frames = [
        _frame(0.0, 0.0, 90.0, travel=0.0, i=0),
        _frame(0.0, 5.0, 90.0, travel=0.0, i=1),
    ]
    regions = [
        {
            "kind": "roof",
            "aabb_enu": [(8.0, 0.0), (12.0, 0.0), (12.0, 20.0), (8.0, 20.0)],
        }
    ]

    def _pl(src: str, z: float) -> dict:
        return {
            "n": np.array([-1.0, 0.0, 0.0]),
            "center": np.array([8.0, 5.0, 4.0]),
            "zncc": z,
            "scores": [z, z - 0.02],
            "view_indices": [0, 1],
            "source": src,
        }

    planes = [
        _pl("ma_segment", 0.55),
        _pl("ma_gap_manhattan", 0.50),
        _pl("ma_gap_sat_edge", 0.40),
    ]
    kept = filter_gap_adds(
        planes, frames, roof_regions=regions, sat_aabb_gate_m=2.0, max_gap_adds=2
    )
    assert len(kept) == 2
    assert kept[0]["source"] == "ma_gap_sat_edge"
    assert kept[1]["source"] == "ma_gap_manhattan"


def test_cli_gap_seeds_sat_edge_default() -> None:
    from click.testing import CliRunner

    from ps1_hood.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["facades", "--help"])
    assert result.exit_code == 0
    assert "--gap-seeds" in result.output
    assert "sat-edge" in result.output
    assert "both" in result.output
    assert "--sat-edge-reanchor" in result.output
    assert "--sat-edge-max-drift" in result.output
    assert "--sat-aabb-gate-sat-edge" in result.output
    assert "--gap-nms-xy" in result.output
    assert "--gap-nms-d-tol" in result.output
    assert "--gap-nms-no-opposite" in result.output


def test_sat_edge_cover_center_tightened() -> None:
    assert SAT_EDGE_COVER_CENTER_M == 2.0
    assert SAT_EDGE_MAX_EDGES == 16
    assert SAT_EDGE_MAX_DRIFT_M == 3.0


def test_uncovered_prefers_return_walls() -> None:
    """Street-long sides fill quota last; returns (|heading−travel|≈90) first."""
    # Travel +N (0°). Building with long E/W street sides and short N/S returns.
    # With max_edges=2 and no product, prefer N/S (return) over E/W (street).
    regions = [
        {
            "kind": "roof",
            "id": "garage",
            "aabb_enu": [(0.0, 0.0), (20.0, 0.0), (20.0, 6.0), (0.0, 6.0)],
        }
    ]
    edges = uncovered_roof_edges(
        regions, [], max_edges=2, travel_heading_deg=0.0
    )
    assert len(edges) == 2
    # Return walls: outward ±E (heading 90/270) when travel is N
    for e in edges:
        assert abs(float(e["n_out"][0])) > 0.7  # ±E normals


def test_cover_center_2m_does_not_cover_distant_return() -> None:
    """Front-wall plane 3.5 m from return segment mid must not cover it."""
    aabb = [(0.0, 0.0), (10.0, 0.0), (10.0, 8.0), (0.0, 8.0)]
    edges = aabb_boundary_edges(aabb, roof_id="r")
    # East return: mid~(10,4), n_out~(+1,0)
    east = [e for e in edges if float(e["n_out"][0]) > 0.5][0]
    # Product on south front, center near SW — within old 4 m of east mid? 
    # east segment from (10,0)-(10,8); point (10,0.5) is on segment.
    # Use a south-facing plane whose center is at (5,0) — far from east.
    n = np.array([0.0, -1.0, 0.0])
    center = np.array([5.0, 0.0, 3.5])
    pl = {"n": n, "d": float(-n @ center), "center": center, "id": "front"}
    assert not product_plane_covers_edge(pl, east, center_m=2.0)
    # Same-orientation wrong: plane facing east but center 3 m west of east wall
    # along the wall length offset — center (7,4) is 3 m from east segment
    n_e = np.array([1.0, 0.0, 0.0])
    c_off = np.array([7.0, 4.0, 3.5])
    pl_e = {"n": n_e, "d": float(-n_e @ np.array([10.0, 4.0, 3.5])), "center": c_off}
    # |d+n·M|=0 so plane-dist ok; center-to-segment = 3 m > 2 → not covered
    assert not product_plane_covers_edge(pl_e, east, center_m=2.0)
    assert product_plane_covers_edge(pl_e, east, center_m=4.0)


def test_reanchor_sat_edge_snaps_and_rejects_drift() -> None:
    regions = [
        {
            "kind": "roof",
            "id": "r0",
            "aabb_enu": [(0.0, 0.0), (10.0, 0.0), (10.0, 8.0), (0.0, 8.0)],
        }
    ]
    # Seed on south edge mid~(5,0), n=(0,-1)
    p0 = np.array([0.0, 0.0])
    p1 = np.array([10.0, 0.0])
    n = np.array([0.0, -1.0, 0.0])
    # Refined center walked ~5 m south (classic ZNCC drift)
    center_ref = np.array([5.0, -5.0, 3.5])
    d_ref = float(-n @ center_ref)  # = -5
    plane = {
        "n": n,
        "d": d_ref,
        "center": center_ref,
        "source": "ma_gap_sat_edge",
        "edge_id": "r0_e0",
        "edge_p0": p0,
        "edge_p1": p1,
        "edge_n_out": np.array([0.0, -1.0]),
        "zncc": 0.5,
    }
    out, why = reanchor_sat_edge_after_score(plane, regions, max_drift_m=3.0)
    assert out is None
    assert "refine_drift" in why
    # Mild drift 2 m — accept and snap back onto edge
    center_mild = np.array([5.0, -2.0, 3.5])
    d_mild = float(-n @ center_mild)
    plane2 = {**plane, "center": center_mild, "d": d_mild}
    out2, why2 = reanchor_sat_edge_after_score(plane2, regions, max_drift_m=3.0)
    assert out2 is not None, why2
    assert abs(float(out2["center"][1])) < 1e-6  # on y=0 edge
    assert abs(float(out2["center"][0]) - 5.0) < 1e-6
    assert abs(float(out2["refine_drift_m"]) - 2.0) < 1e-6


def test_sat_aabb_uses_seed_edge_not_global_nearest() -> None:
    """After drift toward another building, gate to seed edge_id — not nearest."""
    regions = [
        {
            "kind": "roof",
            "id": "seed_bldg",
            "aabb_enu": [(0.0, 0.0), (10.0, 0.0), (10.0, 8.0), (0.0, 8.0)],
        },
        {
            "kind": "roof",
            "id": "other",
            "aabb_enu": [(0.0, -6.0), (10.0, -6.0), (10.0, -4.0), (0.0, -4.0)],
        },
    ]
    # Center near other building south of seed; seed edge is south of seed_bldg
    center = np.array([5.0, -4.5, 3.5])
    n = np.array([0.0, -1.0, 0.0])
    # Global nearest is other roof (~0.5 m); seed edge at y=0 is ~4.5 m away
    ok_global, why_g = sat_aabb_edge_ok(center, n, regions, max_edge_m=2.0)
    assert ok_global, why_g  # would falsely pass via other building
    ok_seed, why_s = sat_aabb_edge_ok(
        center,
        n,
        regions,
        max_edge_m=2.0,
        edge_id="seed_bldg_e0",
        edge_p0=np.array([0.0, 0.0]),
        edge_p1=np.array([10.0, 0.0]),
    )
    assert not ok_seed
    assert "dist_seed_edge" in why_s
    assert "dist_nearest_any" in why_s


def test_filter_gap_adds_reanchor_and_sat_edge_soft_only() -> None:
    frames = [
        _frame(0.0, -12.0, 0.0, travel=90.0, i=0),  # looking N at south wall
        _frame(2.0, -12.0, 0.0, travel=90.0, i=1),
    ]
    regions = [
        {
            "kind": "roof",
            "id": "r0",
            "aabb_enu": [(0.0, 0.0), (10.0, 0.0), (10.0, 8.0), (0.0, 8.0)],
        }
    ]
    n = np.array([0.0, -1.0, 0.0])
    # Mild-drift sat_edge: 1.5 m off edge → reanchor to 0, pass 2 m gate
    c_sat = np.array([5.0, -1.5, 3.5])
    sat = {
        "n": n,
        "d": float(-n @ c_sat),
        "center": c_sat,
        "zncc": 0.45,
        "scores": [0.45, 0.40],
        "view_indices": [0, 1],
        "source": "ma_gap_sat_edge",
        "edge_id": "r0_e0",
        "edge_p0": np.array([0.0, 0.0]),
        "edge_p1": np.array([10.0, 0.0]),
        "edge_n_out": np.array([0.0, -1.0]),
    }
    # MA peel 4 m off any edge — soft sat_edge gate must NOT rescue it
    c_ma = np.array([5.0, -4.0, 3.5])
    ma = {
        "n": n,
        "d": float(-n @ c_ma),
        "center": c_ma,
        "zncc": 0.55,
        "scores": [0.55, 0.50],
        "view_indices": [0, 1],
        "source": "ma_segment",
    }
    kept = filter_gap_adds(
        [sat, ma],
        frames,
        roof_regions=regions,
        sat_aabb_gate_m=2.0,
        sat_aabb_gate_sat_edge_m=5.0,  # soft for sat_edge only
        max_gap_adds=3,
        sat_edge_reanchor=True,
        sat_edge_max_drift_m=3.0,
    )
    srcs = [p["source"] for p in kept]
    assert "ma_gap_sat_edge" in srcs
    assert "ma_segment" not in srcs
    sat_kept = [p for p in kept if "sat_edge" in p["source"]][0]
    assert abs(float(sat_kept["center"][1])) < 1e-6  # reanchored


def test_filter_rejects_large_refine_drift() -> None:
    frames = [
        _frame(0.0, -12.0, 0.0, travel=90.0, i=0),
        _frame(2.0, -12.0, 0.0, travel=90.0, i=1),
    ]
    regions = [
        {
            "kind": "roof",
            "id": "r0",
            "aabb_enu": [(0.0, 0.0), (10.0, 0.0), (10.0, 8.0), (0.0, 8.0)],
        }
    ]
    n = np.array([0.0, -1.0, 0.0])
    c = np.array([5.0, -5.0, 3.5])  # 5 m drift
    plane = {
        "n": n,
        "d": float(-n @ c),
        "center": c,
        "zncc": 0.50,
        "scores": [0.50, 0.48],
        "view_indices": [0, 1],
        "source": "ma_gap_sat_edge",
        "edge_id": "r0_e0",
        "edge_p0": np.array([0.0, 0.0]),
        "edge_p1": np.array([10.0, 0.0]),
    }
    kept = filter_gap_adds(
        [plane],
        frames,
        roof_regions=regions,
        sat_aabb_gate_m=2.0,
        sat_aabb_gate_sat_edge_m=5.0,
        max_gap_adds=3,
        sat_edge_reanchor=True,
        sat_edge_max_drift_m=3.0,
    )
    assert kept == []


def test_closest_point_on_segment() -> None:
    p = closest_point_on_segment_xy(
        np.array([5.0, -3.0]), np.array([0.0, 0.0]), np.array([10.0, 0.0])
    )
    assert abs(float(p[0]) - 5.0) < 1e-9
    assert abs(float(p[1])) < 1e-9
