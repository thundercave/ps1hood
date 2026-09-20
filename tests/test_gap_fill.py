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
    GAP_FILL_PEEL_CAP,
    build_gap_fill_seeds,
    corner_seeds_from_roof_aabbs,
    count_untextured_product,
    estimate_travel_heading_deg,
    filter_ma_peels_for_gaps,
    manhattan_side_seeds,
    prefer_side_wall_order,
    reject_road_center_hyps,
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
