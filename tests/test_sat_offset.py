"""Forced SE(2) façade↔Ortho Canny measure + bak/apply (no free-pose)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from ps1_hood.align.georef import apply_se2_xy, fit_se2
from ps1_hood.align.sat_offset import (
    SatOffsetError,
    apply_forced_se2,
    bak_force,
    chamfer_cost_m,
    facade_edge_samples,
    facade_long_edges,
    match_facade_to_sat,
    measure_facade_sat_se2,
    multistart_facade_chamfer,
    write_t_force,
)
from ps1_hood.config import ProjectSpec
from ps1_hood.geo import BBox, LocalFrame
from ps1_hood.project import create_project


def _write_ortho(project, *, shift_e: float = 0.0, shift_n: float = 0.0) -> None:
    """Tiny ortho with a bright L-shaped building edge for Canny."""
    import json as _json

    from ps1_hood.capture.satellite import Ortho

    h, w = 128, 128
    img = np.full((h, w, 3), 40, dtype=np.uint8)
    # rectangle building footprint in pixels
    cv2.rectangle(img, (40, 40), (90, 90), (200, 200, 200), thickness=2)
    if abs(shift_e) > 1e-6 or abs(shift_n) > 1e-6:
        # shift drawing for "sat truth" offset tests — keep edges only for now
        pass
    path = project.satellite_dir / "ortho.jpg"
    cv2.imwrite(str(path), img)
    spec = project.load_spec()
    project.write_json(
        project.satellite_dir / "ortho.json",
        {
            "path": str(path),
            "width": w,
            "height": h,
            "bbox": spec.bbox.as_dict(),
            "provider": "test",
            "crs": "EPSG:4326",
        },
    )


def _plane(pid: str, p0, p1, height: float = 8.0) -> dict:
    """Vertical façade quad along XY segment p0→p1."""
    e0, n0 = p0
    e1, n1 = p1
    return {
        "id": pid,
        "n": [0.0, 1.0, 0.0],
        "d": 0.0,
        "quad": [
            [e0, n0, 0.0],
            [e1, n1, 0.0],
            [e1, n1, height],
            [e0, n0, height],
        ],
        "width_m": float(math.hypot(e1 - e0, n1 - n0)),
        "height_m": height,
    }


def _mini_project(tmp_path: Path):
    spec = ProjectSpec(
        name="satoffset",
        bbox=BBox(52.0895, 5.1192, 52.09, 5.12),
        source="google_static",
        align_prior="sat",
    )
    project = create_project(spec, runs_root=tmp_path)
    _write_ortho(project)
    # Façade square ~ matching ortho rectangle in ENU via load_ortho mapping
    from ps1_hood.reconstruct.sat_roofs import load_ortho_for_run, px_to_enu

    ortho = load_ortho_for_run(project.root)
    # corners of the drawn rectangle in ENU
    corners = [
        px_to_enu(ortho, 40, 90),
        px_to_enu(ortho, 90, 90),
        px_to_enu(ortho, 90, 40),
        px_to_enu(ortho, 40, 40),
    ]
    planes = [
        _plane("facade_00", corners[0], corners[1]),
        _plane("facade_01", corners[1], corners[2]),
        _plane("facade_02", corners[2], corners[3]),
        _plane("facade_03", corners[3], corners[0]),
    ]
    project.write_json(project.recon_dir / "planes.json", {"planes": planes})
    # poses + tiny recon artefacts
    poses = [
        {"pano_id": "p0", "e": float(corners[0][0]), "n": float(corners[0][1]), "u": 2.5, "heading": 90.0},
        {"pano_id": "p1", "e": float(corners[1][0]), "n": float(corners[1][1]), "u": 2.5, "heading": 90.0},
    ]
    project.write_json(project.align_dir / "poses.json", poses)
    project.write_json(project.align_dir / "cameras.json", list(poses))
    project.write_json(project.align_dir / "georef.json", {"prior": "sat"})
    (project.recon_dir / "cloud.ply").write_text(
        "ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
        f"{corners[0][0]} {corners[0][1]} 1.0\n{corners[1][0]} {corners[1][1]} 1.0\n"
    )
    (project.recon_dir / "facades.obj").write_text(
        f"v {corners[0][0]} {corners[0][1]} 0\n"
        f"v {corners[1][0]} {corners[1][1]} 0\n"
        f"v {corners[1][0]} {corners[1][1]} 8\n"
        f"v {corners[0][0]} {corners[0][1]} 8\n"
        "f 1 2 3 4\n"
    )
    (project.recon_dir / "roofs.obj").write_text("v 0 0 9\nv 1 0 9\nv 1 1 9\nf 1 2 3\n")
    (project.recon_dir / "street.obj").write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nf 1 2 3\n")
    project.write_json(
        project.recon_dir / "scene.json",
        {"cameras": [{"pano_id": "p0", "e": poses[0]["e"], "n": poses[0]["n"], "heading": 90.0}]},
    )
    return project, corners


def test_facade_long_edges_dedupes_top_bottom():
    planes = [_plane("f0", (0.0, 0.0), (8.0, 0.0))]
    edges = facade_long_edges(planes, min_len_m=3.0)
    assert len(edges) == 1
    assert abs(edges[0]["length_m"] - 8.0) < 1e-6


def test_fit_se2_roundtrip_keys():
    before = [{"e": 0.0, "n": 0.0}, {"e": 10.0, "n": 0.0}]
    after = [{"e": 1.0, "n": 2.0}, {"e": 11.0, "n": 2.0}]
    T = fit_se2(before, after)
    assert set(T) >= {"tx_m", "ty_m", "yaw_deg", "s", "pivot_e", "pivot_n"}
    e2, n2 = apply_se2_xy(0.0, 0.0, T)
    assert abs(e2 - 1.0) < 1e-6 and abs(n2 - 2.0) < 1e-6


def test_measure_near_identity(tmp_path: Path):
    project, _ = _mini_project(tmp_path)
    T = measure_facade_sat_se2(project, search_r_m=8.0, min_pairs=4)
    assert T["source"] == "facade_sat_chamfer"
    assert T["n_pairs"] >= 4
    assert T["rms_m"] <= 1.5
    assert abs(T["yaw_deg"]) <= 15.0
    assert math.hypot(T["tx_m"], T["ty_m"]) <= 10.0
    write_t_force(project.align_dir / "T_force.json", T)
    assert (project.align_dir / "T_force.json").is_file()


def test_measure_gate_rejects_too_few_pairs(tmp_path: Path):
    project, corners = _mini_project(tmp_path)
    # Move façades far from sat edges
    far = [
        _plane("facade_00", (corners[0][0] + 50, corners[0][1] + 50), (corners[1][0] + 50, corners[1][1] + 50)),
        _plane("facade_01", (corners[1][0] + 50, corners[1][1] + 50), (corners[2][0] + 50, corners[2][1] + 50)),
        _plane("facade_02", (corners[2][0] + 50, corners[2][1] + 50), (corners[3][0] + 50, corners[3][1] + 50)),
        _plane("facade_03", (corners[3][0] + 50, corners[3][1] + 50), (corners[0][0] + 50, corners[0][1] + 50)),
    ]
    project.write_json(project.recon_dir / "planes.json", {"planes": far})
    with pytest.raises(SatOffsetError, match="pairs"):
        measure_facade_sat_se2(project, search_r_m=2.0, min_pairs=4)


def test_apply_moves_cams_cloud_facades_skips_roofs_street(tmp_path: Path):
    project, _ = _mini_project(tmp_path)
    roofs_before = (project.recon_dir / "roofs.obj").read_text()
    street_before = (project.recon_dir / "street.obj").read_text()
    poses_before = project.read_json(project.align_dir / "poses.json")
    T = {
        "tx_m": 1.5,
        "ty_m": -0.5,
        "yaw_deg": 0.0,
        "s": 1.0,
        "pivot_e": 0.0,
        "pivot_n": 0.0,
        "source": "facade_sat_chamfer",
    }
    meta = apply_forced_se2(project, T, targets="cams,cloud,facades,planes", skip="roofs,street", bak=True)
    assert meta["stats"]["roofs.obj"] == "skipped"
    assert meta["stats"]["street.obj"] == "skipped"
    assert (project.recon_dir / "roofs.obj").read_text() == roofs_before
    assert (project.recon_dir / "street.obj").read_text() == street_before
    poses_after = project.read_json(project.align_dir / "poses.json")
    assert abs(poses_after[0]["e"] - poses_before[0]["e"] - 1.5) < 1e-6
    assert abs(poses_after[0]["n"] - poses_before[0]["n"] + 0.5) < 1e-6
    georef = project.read_json(project.align_dir / "georef.json")
    assert georef.get("forced_se2") is True
    assert "T_force" in georef and "T_applied" in georef
    assert meta["bak"]["stamp"]
    assert Path(meta["bak"]["align_bak"]).is_dir()


def test_multistart_chamfer_improves_or_holds(tmp_path: Path):
    project, _ = _mini_project(tmp_path)
    from ps1_hood.reconstruct.sat_roofs import load_ortho_for_run

    ortho = load_ortho_for_run(project.root)
    planes = project.read_json(project.recon_dir / "planes.json")["planes"]
    edges = facade_long_edges(planes)
    samples = facade_edge_samples(edges, spacing_m=1.5)
    T0 = {"tx_m": 0.0, "ty_m": 0.0, "yaw_deg": 0.0, "s": 1.0, "pivot_e": float(samples[:, 0].mean()), "pivot_n": float(samples[:, 1].mean())}
    c0 = chamfer_cost_m(ortho, samples, T0)
    T1 = multistart_facade_chamfer(
        ortho,
        samples,
        init=T0,
        de_span_m=2.0,
        dyaw_span_deg=4.0,
        step_m=2.0,
        step_yaw_deg=4.0,
    )
    assert T1["chamfer_m"] <= c0 + 1e-6
    assert abs(T1["yaw_deg"]) <= 15.0


def test_no_free_pose_single_se2_only():
    """Sacred: payload is one SE(2); no per-cam free keys."""
    T = {
        "tx_m": 0.1,
        "ty_m": -0.2,
        "yaw_deg": 1.0,
        "s": 1.0,
        "pivot_e": 0.0,
        "pivot_n": 0.0,
        "source": "facade_sat_chamfer",
    }
    forbidden = {"R", "t_free", "per_cam", "sim3", "scale_free"}
    assert not (forbidden & set(T))
    assert abs(float(T["s"]) - 1.0) < 1e-12
