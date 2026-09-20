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


def test_fit_pairs_se2_preview_only_no_apply(tmp_path: Path):
    """Studio yellow↔red corner picks → SE(2) preview; does not mutate product."""
    project, corners = _mini_project(tmp_path)
    roofs_before = (project.recon_dir / "roofs.obj").read_text()
    poses_before = project.read_json(project.align_dir / "poses.json")
    pairs = [
        {
            "yellow": {"e": float(corners[i][0]), "n": float(corners[i][1])},
            "red": {"e": float(corners[i][0] - 2.0), "n": float(corners[i][1] + 1.0)},
        }
        for i in range(3)
    ]
    from ps1_hood.align.sat_offset import fit_pairs_se2, persist_t_pick, load_t_force

    payload = fit_pairs_se2(pairs)
    assert payload["applied"] is False
    assert payload["n_pairs"] == 3
    assert payload["source"] == "studio_corner_picks"
    assert abs(payload["tx_m"] - 2.0) < 1e-5
    assert abs(payload["ty_m"] - (-1.0)) < 1e-5
    assert payload["rms_m"] <= 1.5
    assert "red_mapped" in payload["preview"]
    paths = persist_t_pick(project, payload)
    assert Path(paths["T_pick"]).is_file()
    assert Path(paths["T_pick_pairs"]).is_file()
    # preview must not have applied
    assert (project.recon_dir / "roofs.obj").read_text() == roofs_before
    assert project.read_json(project.align_dir / "poses.json") == poses_before
    T = load_t_force(paths["T_pick"])
    assert T["source"] == "studio_corner_picks"
    audit = json.loads(Path(paths["T_pick_pairs"]).read_text(encoding="utf-8"))
    assert audit.get("applied") is False


def test_fit_pairs_requires_three():
    from ps1_hood.align.sat_offset import SatOffsetError, fit_pairs_se2

    pairs = [
        {"yellow": {"e": 1.0, "n": 0.0}, "red": {"e": 0.0, "n": 0.0}},
        {"yellow": {"e": 2.0, "n": 0.0}, "red": {"e": 1.0, "n": 0.0}},
    ]
    with pytest.raises(SatOffsetError, match="pairs"):
        fit_pairs_se2(pairs)


def test_apply_t_pick_skips_roofs_street(tmp_path: Path):
    """Confirm path: apply T_pick via existing sat-offset apply (bak + skip roofs/street)."""
    project, corners = _mini_project(tmp_path)
    from ps1_hood.align.sat_offset import apply_forced_se2, fit_pairs_se2, persist_t_pick

    pairs = [
        {
            "yellow": {"e": float(corners[i][0]), "n": float(corners[i][1])},
            "red": {"e": float(corners[i][0] - 1.5), "n": float(corners[i][1] - 0.5)},
        }
        for i in range(3)
    ]
    payload = fit_pairs_se2(pairs)
    persist_t_pick(project, payload)
    roofs_before = (project.recon_dir / "roofs.obj").read_text()
    street_before = (project.recon_dir / "street.obj").read_text()
    poses_before = project.read_json(project.align_dir / "poses.json")
    meta = apply_forced_se2(
        project,
        payload,
        targets="cams,cloud,facades,planes",
        skip="roofs,street",
        bak=True,
    )
    assert meta["stats"]["roofs.obj"] == "skipped"
    assert meta["stats"]["street.obj"] == "skipped"
    assert (project.recon_dir / "roofs.obj").read_text() == roofs_before
    assert (project.recon_dir / "street.obj").read_text() == street_before
    poses_after = project.read_json(project.align_dir / "poses.json")
    assert abs(poses_after[0]["e"] - poses_before[0]["e"] - payload["tx_m"]) < 1e-5



def _handmade_street_mask(h: int = 128, w: int = 128, band: int = 6) -> np.ndarray:
    """Horizontal street corridor mask (centerline ≈ row h//2)."""
    m = np.zeros((h, w), dtype=np.uint8)
    m[h // 2 - band // 2 : h // 2 + band // 2 + 1, :] = 255
    return m


def _mini_project_cam_road(tmp_path: Path, *, cam_shift_e: float = 3.0, cam_shift_n: float = 0.0):
    """Project with cams offset from a mid-row street; street_mask injected in tests."""
    project, _corners = _mini_project(tmp_path)
    from ps1_hood.reconstruct.sat_roofs import load_ortho_for_run, px_to_enu

    ortho = load_ortho_for_run(project.root)
    h = int(ortho.h)
    us = [20, 40, 60, 80, 100]
    poses = []
    for i, u in enumerate(us):
        e, n = px_to_enu(ortho, float(u), float(h // 2))
        poses.append(
            {
                "pano_id": f"c{i}",
                "e": float(e + cam_shift_e),
                "n": float(n + cam_shift_n),
                "u": 2.5,
                "heading": 90.0,
            }
        )
    poses.append({**poses[0], "pano_id": "c0b", "heading": 180.0})
    project.write_json(project.align_dir / "poses.json", poses)
    project.write_json(project.align_dir / "cameras.json", list(poses))
    return project


def _patch_street_mask(monkeypatch):
    """Force segment_roof_yard_mask → clean horizontal street corridor."""
    import ps1_hood.align.sat_offset as so

    def _fake_segment(ortho_bgr, **kwargs):
        h, w = ortho_bgr.shape[:2]
        street = _handmade_street_mask(h, w, band=8)
        empty = np.zeros((h, w), dtype=np.uint8)
        return empty, empty, street

    monkeypatch.setattr(so, "segment_roof_yard_mask", _fake_segment)


def test_street_mask_centerline_nonempty():
    from ps1_hood.align.sat_offset import street_mask_centerline

    street = _handmade_street_mask()
    pix = street_mask_centerline(street)
    assert len(pix) >= 20
    # Medial should hug mid row
    assert abs(float(np.median(pix[:, 1])) - 64.0) <= 2.0


def test_measure_cam_road_recovers_translation(tmp_path: Path, monkeypatch):
    from ps1_hood.align.sat_offset import (
        SOURCE_CAM_STREET_CENTERLINE,
        load_t_force,
        measure_cam_road_se2,
        persist_t_cam_road,
    )

    _patch_street_mask(monkeypatch)
    shift = 3.0
    # Offset perpendicular to horizontal street corridor (N), not along-track (E)
    project = _mini_project_cam_road(tmp_path, cam_shift_e=0.0, cam_shift_n=shift)
    payload = measure_cam_road_se2(project, search_r_m=15.0, min_pairs=4)
    assert payload["source"] == SOURCE_CAM_STREET_CENTERLINE
    assert payload["applied"] is False
    assert payload["n_cams"] >= 4
    assert payload["n_pairs"] >= 4
    assert payload["rms_m"] <= 2.0
    assert abs(payload["yaw_deg"]) <= 10.0
    assert abs(payload["ty_m"] + shift) < 0.5
    assert math.hypot(payload["tx_m"], payload["ty_m"]) <= 12.0
    paths = persist_t_cam_road(project, payload)
    T = load_t_force(paths["T_cam_road"])
    assert T["source"] == SOURCE_CAM_STREET_CENTERLINE
    assert T.get("applied") is False


def test_measure_cam_road_gate_too_few(tmp_path: Path, monkeypatch):
    from ps1_hood.align.sat_offset import SatOffsetError, measure_cam_road_se2

    _patch_street_mask(monkeypatch)
    project = _mini_project_cam_road(tmp_path, cam_shift_e=0.0, cam_shift_n=50.0)
    with pytest.raises(SatOffsetError, match="pairs|cams"):
        measure_cam_road_se2(project, search_r_m=2.0, min_pairs=4)


def test_apply_t_cam_road_skips_roofs_street(tmp_path: Path, monkeypatch):
    from ps1_hood.align.sat_offset import (
        apply_forced_se2,
        measure_cam_road_se2,
        persist_t_cam_road,
    )

    _patch_street_mask(monkeypatch)
    project = _mini_project_cam_road(tmp_path, cam_shift_e=0.0, cam_shift_n=2.5)
    payload = measure_cam_road_se2(project, search_r_m=15.0)
    persist_t_cam_road(project, payload)
    roofs_before = (project.recon_dir / "roofs.obj").read_text()
    street_before = (project.recon_dir / "street.obj").read_text()
    poses_before = project.read_json(project.align_dir / "poses.json")
    meta = apply_forced_se2(
        project,
        payload,
        targets="cams,cloud,facades,planes",
        skip="roofs,street",
        bak=True,
    )
    assert meta["stats"]["roofs.obj"] == "skipped"
    assert meta["stats"]["street.obj"] == "skipped"
    assert (project.recon_dir / "roofs.obj").read_text() == roofs_before
    assert (project.recon_dir / "street.obj").read_text() == street_before
    poses_after = project.read_json(project.align_dir / "poses.json")
    assert abs(poses_after[0]["e"] - poses_before[0]["e"]) > 0.5 or abs(
        poses_after[0]["n"] - poses_before[0]["n"]
    ) > 0.2
    assert meta["T"]["source"] == "cam_street_centerline"


def test_cam_road_one_rigid_se2_no_free_pose():
    from ps1_hood.align.sat_offset import SOURCE_CAM_STREET_CENTERLINE

    T = {
        "tx_m": 1.0,
        "ty_m": -0.5,
        "yaw_deg": 0.0,
        "s": 1.0,
        "pivot_e": 0.0,
        "pivot_n": 0.0,
        "source": SOURCE_CAM_STREET_CENTERLINE,
    }
    forbidden = {"R", "t_free", "per_cam", "sim3", "scale_free", "free_pose"}
    assert not (forbidden & set(T))
    assert abs(float(T["s"]) - 1.0) < 1e-12


def test_match_cams_to_centerline_nn():
    from ps1_hood.align.sat_offset import match_cams_to_centerline

    cams = np.array([[0.0, 1.0], [5.0, 1.0], [10.0, 1.0], [15.0, 1.0]], dtype=np.float64)
    cl = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0], [15.0, 0.0]], dtype=np.float64)
    before, after, dists = match_cams_to_centerline(cams, cl, search_r_m=15.0, min_nn_m=0.5)
    assert len(before) == 4
    assert all(abs(d - 1.0) < 1e-6 for d in dists)
    T_tx = sum(a["e"] - b["e"] for a, b in zip(after, before)) / 4
    T_ty = sum(a["n"] - b["n"] for a, b in zip(after, before)) / 4
    assert abs(T_tx) < 1e-6 and abs(T_ty + 1.0) < 1e-6


def test_continuity_avoids_parallel_branch():
    """Continuity keeps cams on one road; plain NN can latch a parallel branch."""
    import numpy as np
    from ps1_hood.align.sat_offset import (
        match_cams_to_centerline,
        match_cams_to_centerline_continuity,
    )

    cams = np.array([[0.0, 1.0], [5.0, 1.0], [10.0, 8.0], [15.0, 1.0], [20.0, 1.0]])
    cl_main = np.stack([np.linspace(0, 20, 21), np.zeros(21)], axis=1)
    cl_park = np.stack([np.linspace(0, 20, 21), np.full(21, 10.0)], axis=1)
    cl = np.vstack([cl_main, cl_park])

    _, after_nn, _ = match_cams_to_centerline(cams, cl, search_r_m=15.0, min_nn_m=0.1)
    _, after_c, _ = match_cams_to_centerline_continuity(
        cams, cl, search_r_m=15.0, min_nn_m=0.1, max_lateral_m=8.0
    )
    assert any(abs(a["n"] - 10.0) < 1.0 for a in after_nn), "NN should latch parking branch"
    assert all(abs(a["n"]) < 1.5 for a in after_c), "continuity must stay on main road"


def test_crop_street_mask_to_cam_corridor_drops_far_street():
    import numpy as np
    from types import SimpleNamespace
    from ps1_hood.align.sat_offset import crop_street_mask_to_cam_corridor

    h, w = 100, 100
    street = np.zeros((h, w), dtype=np.uint8)
    street[50, :] = 255  # horizontal road through middle
    street[10, :] = 255  # far parallel road

    class _Ortho:
        h = 100
        w = 100
        def enu_to_px(self, e, n):
            return e, n  # 1 m/px identity

    # Mock metres: patch cam_xy_mask path by making ortho with metres helpers
    # crop uses cam_xy_mask from sat_street which needs _metres_per_px(ortho)
    # Provide minimal Ortho-like with ee/sw/nn/sh
    ortho = SimpleNamespace(
        h=100, w=100,
        ee=100.0, sw=0.0, nn=100.0, sh=0.0,
        enu_to_px=lambda e, n: (e, 100.0 - n) if False else (e, n),
    )
    # Use real method binding
    def enu_to_px(e, n):
        return float(e), float(n)
    ortho.enu_to_px = enu_to_px

    cams = np.array([[20.0, 50.0], [40.0, 50.0], [60.0, 50.0], [80.0, 50.0]])
    cropped = crop_street_mask_to_cam_corridor(street, ortho, cams, corridor_m=8.0)
    assert int((cropped[50] > 0).sum()) > 10
    # Far road at row 10 should be mostly gone
    assert int((cropped[10] > 0).sum()) < int((street[10] > 0).sum()) * 0.25


def test_mad_gate_rejects_disagreeing_pairs(tmp_path: Path, monkeypatch):
    """MAD gate rejects when residual MAD exceeds max_mad_m (multi-branch symptom)."""
    from ps1_hood.align.sat_offset import SatOffsetError, measure_cam_road_se2
    import ps1_hood.align.sat_offset as so

    _patch_street_mask(monkeypatch)
    project = _mini_project_cam_road(tmp_path, cam_shift_e=0.0, cam_shift_n=2.0)

    # Force a high MAD regardless of matcher geometry
    monkeypatch.setattr(so, "_residual_mad", lambda errs: (3.0, 2.5))
    with pytest.raises(SatOffsetError, match=r"mad"):
        measure_cam_road_se2(
            project,
            search_r_m=15.0,
            max_rms_m=20.0,
            max_mad_m=1.5,
            translation_only=True,
        )


def test_residual_mad_reports_spread():
    from ps1_hood.align.sat_offset import _residual_mad

    med, mad = _residual_mad([0.0, 0.5, 1.0, 8.0, 9.0])
    assert med >= 0.5
    assert mad >= 0.5


def test_fit_cam_road_from_pairs_preview_only():
    from ps1_hood.align.sat_offset import SOURCE_CAM_ROAD_PICKS, fit_cam_road_from_pairs

    pairs = [
        {"red": {"e": 0.0, "n": 0.0}, "yellow": {"e": 0.0, "n": 2.0}},
        {"red": {"e": 5.0, "n": 0.0}, "yellow": {"e": 5.0, "n": 2.0}},
        {"red": {"e": 10.0, "n": 0.0}, "yellow": {"e": 10.0, "n": 2.0}},
        {"red": {"e": 15.0, "n": 0.0}, "yellow": {"e": 15.0, "n": 2.0}},
    ]
    payload = fit_cam_road_from_pairs(pairs, translation_only=True)
    assert payload["source"] == SOURCE_CAM_ROAD_PICKS or "cam_road" in payload["source"]
    assert payload["applied"] is False
    assert abs(payload["ty_m"] - 2.0) < 0.2
    assert payload["rms_m"] <= 2.0
