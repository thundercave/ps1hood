"""Align prior sat (default): never BAG-snap; bag prior still can; georef in scene."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ps1_hood.align.georef import (
    apply_se2_xy,
    georef_payload,
    summarize_se2,
)
from ps1_hood.config import ProjectSpec
from ps1_hood.geo import BBox, LocalFrame
from ps1_hood.project import Project, create_project
from ps1_hood.reconstruct.export import scene_payload


def _mini_project(tmp_path: Path) -> Project:
    spec = ProjectSpec(
        name="alignprior",
        bbox=BBox(52.0895, 5.1192, 52.09, 5.12),
        source="google_static",
        align_prior="sat",
    )
    project = create_project(spec, runs_root=tmp_path)
    # Minimal shots / osm / sat / bag so stage_align can start
    shot_img = project.cropped_dir / "dummy.jpg"
    # tiny jpeg via opencv if available, else empty placeholder path referenced after mock
    try:
        import cv2

        cv2.imwrite(str(shot_img), np.zeros((32, 32, 3), dtype=np.uint8))
    except Exception:
        shot_img.write_bytes(b"")
    project.write_json(
        project.cropped_dir / "shots.json",
        [
            {
                "pano_id": "p0",
                "lat": 52.08975,
                "lon": 5.1196,
                "heading": 90.0,
                "pitch": 0.0,
                "fov": 90.0,
                "path": str(shot_img),
            },
            {
                "pano_id": "p1",
                "lat": 52.0898,
                "lon": 5.1197,
                "heading": 90.0,
                "pitch": 0.0,
                "fov": 90.0,
                "path": str(shot_img),
            },
        ],
    )
    project.write_json(project.osm_dir / "roads.json", {"polylines_enu": []})
    # ortho meta + dummy image
    ortho_path = project.satellite_dir / "ortho.jpg"
    try:
        import cv2

        cv2.imwrite(str(ortho_path), np.zeros((64, 64, 3), dtype=np.uint8))
    except Exception:
        ortho_path.write_bytes(b"")
    project.write_json(
        project.satellite_dir / "ortho.json",
        {
            "path": str(ortho_path),
            "width": 64,
            "height": 64,
            "bbox": spec.bbox.as_dict(),
            "provider": "esri_world_imagery",
            "crs": "EPSG:4326",
        },
    )
    # Non-empty BAG so legacy path would have snapped
    project.write_json(
        project.bag_dir / "buildings.json",
        [
            {
                "id": "b1",
                "bbox": [-5.0, -5.0, 0.0, 5.0, 5.0, 10.0],
                "vertices": [
                    [-5.0, -5.0, 0.0],
                    [5.0, -5.0, 0.0],
                    [5.0, 5.0, 0.0],
                    [-5.0, 5.0, 0.0],
                    [-5.0, -5.0, 10.0],
                    [5.0, -5.0, 10.0],
                    [5.0, 5.0, 10.0],
                    [-5.0, 5.0, 10.0],
                ],
                "faces": [],
            }
        ],
    )
    return project


def _fake_initial(shots, spec, frame, osm):
    out = []
    for i, s in enumerate(shots):
        if i and shots[i - 1]["pano_id"] == s["pano_id"]:
            continue
        # one pose per pano
        e, n, _ = frame.to_enu(s["lat"], s["lon"], 0.0)
        out.append(
            {
                "pano_id": s["pano_id"],
                "e": e,
                "n": n,
                "u": spec.camera_height_m,
                "heading": float(s["heading"]),
                "pitch": float(s.get("pitch") or 0.0),
                "fov": float(s.get("fov") or spec.fov_deg),
                "shot_path": s["path"],
                "lat": s["lat"],
                "lon": s["lon"],
                "travel_heading": float(s["heading"]),
                "e_gps": e,
                "n_gps": n,
            }
        )
    return out


def _fake_refine(poses, spec, frame, ortho, *, use_satellite=True, use_features=True, **_kw):
    refined = [dict(p) for p in poses]
    for p in refined:
        if use_satellite:
            p["e"] = float(p["e"]) + 1.5
            p["n"] = float(p["n"]) + 0.5
            p["sat_score"] = 0.42
        p["heading"] = float(p["heading"]) + (2.0 if use_satellite else 0.0)
    return refined


def test_summarize_se2_centroid_and_yaw() -> None:
    before = [
        {"e": 0.0, "n": 0.0, "heading": 0.0},
        {"e": 10.0, "n": 0.0, "heading": 90.0},
    ]
    after = [
        {"e": 2.0, "n": 1.0, "heading": 10.0},
        {"e": 12.0, "n": 1.0, "heading": 100.0},
    ]
    T = summarize_se2(before, after)
    assert abs(T["tx_m"] - 2.0) < 1e-9
    assert abs(T["ty_m"] - 1.0) < 1e-9
    assert abs(T["yaw_deg"] - 10.0) < 1e-6
    assert abs(T["pivot_e"] - 5.0) < 1e-9
    # Pure translation of XY (yaw recorded for headings; zero yaw for point seat)
    T_xy = {**T, "yaw_deg": 0.0}
    e2, n2 = apply_se2_xy(0.0, 0.0, T_xy)
    assert abs(e2 - 2.0) < 1e-6
    assert abs(n2 - 1.0) < 1e-6


def test_scene_payload_includes_georef() -> None:
    bbox = BBox(52.0895, 5.1192, 52.09, 5.12)
    frame = LocalFrame.from_bbox(bbox)
    georef = georef_payload(
        prior="sat",
        T_sat={"tx_m": 1.0, "ty_m": -0.5, "yaw_deg": 2.0, "s": 1.0},
        sat_score_mean=0.3,
        bag_snapped=0,
        n_poses=2,
    )
    scene = scene_payload(
        spec_name="smoke",
        bbox=bbox,
        frame=frame,
        poses=[
            {
                "pano_id": "a",
                "e": 0.0,
                "n": 0.0,
                "u": 2.5,
                "heading": 0.0,
                "pitch": 0.0,
                "fov": 90.0,
            }
        ],
        frames=[],
        satellite=None,
        cloud=None,
        georef=georef,
    )
    assert scene["georef"]["prior"] == "sat"
    assert scene["georef"]["T_sat"]["tx_m"] == 1.0
    assert scene["georef"]["overlay_registered"] is True


def test_sat_prior_never_calls_bag_snap(tmp_path: Path) -> None:
    project = _mini_project(tmp_path)
    with (
        patch("ps1_hood.pipeline.initial_poses", side_effect=_fake_initial) as init,
        patch("ps1_hood.pipeline.refine_poses", side_effect=_fake_refine) as refine,
        patch("ps1_hood.pipeline.snap_camera_to_bag") as snap,
        patch("ps1_hood.pipeline.write_debug_overlay"),
        patch("ps1_hood.pipeline.Ortho.load", return_value=MagicMock()),
        patch("ps1_hood.pipeline.explode_orbit_cameras", return_value=[]),
        patch("ps1_hood.pipeline._write_live"),
        patch("ps1_hood.pipeline.push_out_of_footprints", side_effect=lambda e, n, *_a, **_k: (e, n)) as push,
    ):
        from ps1_hood.pipeline import stage_align

        poses = stage_align(project, align_prior="sat")
    snap.assert_not_called()
    # refine must have been asked for sat + features even though BAG buildings exist
    assert refine.call_args.kwargs.get("use_satellite") is True
    assert refine.call_args.kwargs.get("use_features") is True
    assert all(not p.get("bag_snapped") for p in poses)
    georef = project.read_json(project.align_dir / "georef.json")
    assert georef["prior"] == "sat"
    assert georef["bag_snapped"] == 0
    assert "T_sat" in georef
    # footprint push skipped under sat prior
    assert push.call_count == 0
    init.assert_called()


def test_bag_prior_can_call_bag_snap(tmp_path: Path) -> None:
    project = _mini_project(tmp_path)
    spec = project.load_spec()
    spec.align_prior = "bag"
    project.save_spec(spec)

    def fake_snap(photo, buildings, **kwargs):
        return {
            "e": kwargs["e"] + 0.1,
            "n": kwargs["n"] - 0.1,
            "u": kwargs["u"],
            "residual_m": 0.2,
            "score": 0.9,
            "snapped": True,
        }

    with (
        patch("ps1_hood.pipeline.initial_poses", side_effect=_fake_initial),
        patch("ps1_hood.pipeline.refine_poses", side_effect=_fake_refine) as refine,
        patch("ps1_hood.pipeline.snap_camera_to_bag", side_effect=fake_snap) as snap,
        patch("ps1_hood.pipeline.write_debug_overlay"),
        patch("ps1_hood.pipeline.Ortho.load", return_value=MagicMock()),
        patch("ps1_hood.pipeline.explode_orbit_cameras", return_value=[]),
        patch("ps1_hood.pipeline._write_live"),
        patch(
            "ps1_hood.pipeline.push_out_of_footprints",
            side_effect=lambda e, n, *_a, **_k: (e, n),
        ),
        patch("ps1_hood.pipeline.cv2", create=True),
    ):
        # force photo load path: cv2.imread returns a tiny array via real cv2 if present
        import ps1_hood.pipeline as pipe

        real_imread = None
        try:
            import cv2

            real_imread = cv2.imread
        except Exception:
            pass

        poses = pipe.stage_align(project, align_prior="bag")

    # With buildings present, bag prior disables sat refine (legacy)
    assert refine.call_args.kwargs.get("use_satellite") is False
    assert refine.call_args.kwargs.get("use_features") is False
    assert snap.call_count >= 1
    assert any(p.get("bag_snapped") for p in poses)
    georef = project.read_json(project.align_dir / "georef.json")
    assert georef["prior"] == "bag"


def test_project_spec_align_prior_default(tmp_path: Path) -> None:
    spec = ProjectSpec(name="x", bbox=BBox(52.0, 5.0, 52.01, 5.01))
    assert spec.align_prior == "sat"
    project = create_project(spec, runs_root=tmp_path)
    loaded = project.load_spec()
    assert loaded.align_prior == "sat"
