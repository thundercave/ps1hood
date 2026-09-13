"""MapAnything export / pose-lock — mocks, no GPU required."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner
from PIL import Image

from ps1_hood.cli import main
from ps1_hood.config import ProjectSpec
from ps1_hood.geo import BBox, camera_rotation_cv
from ps1_hood.project import create_project
from ps1_hood.reconstruct import mapanything as ma


def _fake_frame(
    tmp: Path,
    *,
    name: str,
    e: float,
    n: float,
    u: float = 1.7,
    heading: float = 90.0,
    pitch: float = 0.0,
    fov: float = 90.0,
    w: int = 64,
    h: int = 48,
) -> dict:
    img = tmp / name
    Image.new("RGB", (w, h), color=(120, 80, 40)).save(img, format="JPEG")
    return {
        "path": str(img),
        "e": e,
        "n": n,
        "u": u,
        "heading": heading,
        "pitch": pitch,
        "fov": fov,
        "width": w,
        "height": h,
        "pano_id": name.split(".")[0],
        "interpolated": False,
    }


def test_K_from_fov_center() -> None:
    K = ma.K_from_fov(640, 480, 90.0)
    assert K.shape == (3, 3)
    assert abs(K[0, 2] - 320.0) < 1e-6
    assert abs(K[1, 2] - 240.0) < 1e-6
    assert K[0, 0] == pytest.approx(K[1, 1])
    assert K[0, 0] == pytest.approx(320.0, rel=1e-6)


def test_cam2world_matches_rotation_cv() -> None:
    frame = {
        "e": 1.0,
        "n": 2.0,
        "u": 3.0,
        "heading": 45.0,
        "pitch": -10.0,
        "fov": 90.0,
        "path": "x.jpg",
    }
    T = ma.cam2world_from_frame(frame)
    R = np.asarray(camera_rotation_cv(45.0, -10.0), dtype=np.float64)
    assert T.shape == (4, 4)
    assert np.allclose(T[:3, :3], R)
    assert np.allclose(T[:3, 3], [1.0, 2.0, 3.0])
    assert np.allclose(T[3], [0, 0, 0, 1])


def test_export_bundle_shape_and_pose_lock(tmp_path: Path) -> None:
    frames = [
        _fake_frame(tmp_path, name="a.jpg", e=0.0, n=0.0, heading=0.0),
        _fake_frame(tmp_path, name="b.jpg", e=5.0, n=1.0, heading=10.0),
        _fake_frame(tmp_path, name="c.jpg", e=10.0, n=2.0, heading=20.0),
    ]
    out = tmp_path / "bundle"
    meta = ma.export_mapanything_bundle(frames, out, stride=1, run_name="t")
    assert meta["n_views"] == 3
    assert meta["pose_lock"] is True
    assert meta["format"] == ma.BUNDLE_FORMAT
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["ignore_pose_inputs"] is False
    assert manifest["infer_flags"]["ignore_pose_inputs"] is False
    assert manifest["hf_model_preferred"] == ma.DEFAULT_HF_MODEL
    ma.assert_bundle_pose_lock(manifest)
    npz = np.load(out / "poses_cam2world.npz")
    assert npz["poses"].shape == (3, 4, 4)
    assert npz["intrinsics"].shape == (3, 3, 3)
    for v in manifest["views"]:
        assert (out / v["image"]).is_file()
        assert v["is_metric_scale"] is True


def test_export_stride_and_pose_lock_assert(tmp_path: Path) -> None:
    frames = [
        _fake_frame(tmp_path, name=f"f{i}.jpg", e=float(i) * 4.0, n=0.0)
        for i in range(6)
    ]
    meta = ma.export_mapanything_bundle(frames, tmp_path / "b2", stride=2)
    assert meta["n_views"] == 3  # 0,2,4


def test_assert_bundle_rejects_drift(tmp_path: Path) -> None:
    frames = [
        _fake_frame(tmp_path, name="a.jpg", e=0.0, n=0.0),
        _fake_frame(tmp_path, name="b.jpg", e=4.0, n=0.0),
    ]
    ma.export_mapanything_bundle(frames, tmp_path / "b3")
    manifest = json.loads(
        (tmp_path / "b3" / "manifest.json").read_text(encoding="utf-8")
    )
    manifest["views"][0]["camera_poses"][0][3] += 1.5  # drift translation
    with pytest.raises(RuntimeError, match="drifted"):
        ma.assert_bundle_pose_lock(manifest)


def test_colmap_demo_argv_never_ignores_poses(tmp_path: Path) -> None:
    demo = tmp_path / "demo_inference_on_colmap_outputs.py"
    demo.write_text("#", encoding="utf-8")
    argv = ma.colmap_demo_argv(
        demo,
        colmap_path=tmp_path / "sparse",
        output_directory=tmp_path / "out",
        stride=2,
        apache=True,
    )
    assert "--apache" in argv
    assert "--stride" in argv
    assert argv[argv.index("--stride") + 1] == "2"
    assert "--save_glb" in argv
    assert "--save_colmap" in argv
    assert "--ignore_pose_inputs" not in argv


def test_require_cuda_no_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    fake = types.ModuleType("torch")

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return False

    fake.cuda = _Cuda  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake)
    with pytest.raises(RuntimeError, match="requires CUDA"):
        ma.require_cuda_for_densify()


def test_which_mapanything_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
        if name == "mapanything" or (isinstance(name, str) and name.startswith("mapanything.")):
            raise ImportError("missing")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert ma.which_mapanything() is None


def test_cam_points_to_enu_matches_cam2world() -> None:
    """WORLD = R @ X_cam + C with camera_rotation_cv columns as R_c2w."""
    frame = {"e": 10.0, "n": -2.0, "u": 1.7, "heading": 90.0, "pitch": 0.0}
    T = ma.cam2world_from_frame(frame)
    R, C = T[:3, :3], T[:3, 3]
    # Point 5 m along camera forward (+Z_cam) should land near C + 5*forward.
    X_cam = np.array([[0.0, 0.0, 5.0]])
    Pw = ma.cam_points_to_enu(X_cam, R, C)
    forward = R[:, 2]
    expected = C + 5.0 * forward
    assert np.allclose(Pw[0], expected, atol=1e-6)
    # Right (+X_cam) → ENU eastish for heading=90 (looking east).
    X_right = np.array([[2.0, 0.0, 0.0]])
    Pr = ma.cam_points_to_enu(X_right, R, C)
    assert np.allclose(Pr[0], C + 2.0 * R[:, 0], atol=1e-6)


def test_cam2world_is_not_world2cam() -> None:
    """Translation must be camera centre C, not OpenCV t=-Rcw@C."""
    frame = {"e": 3.0, "n": 4.0, "u": 1.5, "heading": 45.0, "pitch": -5.0}
    T = ma.cam2world_from_frame(frame)
    R = T[:3, :3]
    C = T[:3, 3]
    assert np.allclose(C, [3.0, 4.0, 1.5])
    Rcw = R.T
    t_w2c = -Rcw @ C
    assert not np.allclose(C, t_w2c), "cam2world translation must not equal w2c t"
    # Round-trip: world point = C maps to camera origin.
    X_cam0 = Rcw @ (C - C)
    assert np.allclose(X_cam0, 0.0)


def test_fuse_predictions_uses_locked_poses_not_predicted(tmp_path: Path) -> None:
    """Product PLY from pts3d_cam + our R,C — predicted pose only logged."""
    # Camera at ENU origin, identity R (heading such that R≈I is awkward;
    # use explicit locked T = eye → C=0, R=I so ENU == cam for this unit test).
    locked = [np.eye(4, dtype=np.float64)]
    locked[0][:3, 3] = [0.0, 0.0, 1.7]

    pts_cam = np.zeros((2, 2, 3), dtype=np.float64)
    pts_cam[0, 0] = [1.0, 0.0, 4.0]   # street-ish depth
    pts_cam[0, 1] = [0.0, 0.0, 5.0]
    pts_cam[1, 0] = [-1.0, 0.5, 3.5]
    pts_cam[1, 1] = [0.5, -0.2, 6.0]
    mask = np.array([[True, False], [True, True]], dtype=bool)
    pred_pose = np.eye(4)
    pred_pose[:3, 3] = [100.0, 0.0, 50.0]  # bogus predicted C (Studio ~50m float)

    meta = ma.fuse_predictions_to_ply(
        [{"pts3d_cam": pts_cam, "mask": mask, "camera_poses": pred_pose}],
        tmp_path / "out.ply",
        locked_poses=locked,
    )
    assert meta["points"] == 3
    assert meta["predicted_poses_discarded"] is True
    assert meta["export_source"] == "pts3d_cam"
    assert meta["export_frame"] == "ENU"
    assert meta["predicted_pose_median_delta_m"] == pytest.approx(
        float(np.linalg.norm(np.array([100.0, 0.0, 50.0]) - np.array([0.0, 0.0, 1.7]))),
        rel=1e-6,
    )
    assert Path(meta["path"]).is_file()
    # Points must sit near C + X_cam (R=I), not near predicted C.
    assert meta["enu_check"]["cloud_z_p50"] == pytest.approx(1.7 + 4.0, abs=2.5)


def test_fuse_refuses_raw_pts3d_as_enu(tmp_path: Path) -> None:
    """Dumping model-world pts3d without pts3d_cam must fail loud."""
    locked = [np.eye(4)]
    pts = np.ones((2, 2, 3), dtype=np.float64) * 50.0
    with pytest.raises(RuntimeError, match="pts3d_cam"):
        ma.fuse_predictions_to_ply(
            [{"pts3d": pts, "mask": np.ones((2, 2), dtype=bool)}],
            tmp_path / "bad.ply",
            locked_poses=locked,
        )


def test_assert_cloud_near_cameras_catches_float_sheet() -> None:
    """Studio symptom: z p50~60 vs cameras ~3 → fail loud."""
    cams = np.array([[-20.0, 5.0, 3.4], [-15.0, 5.0, 3.5], [-10.0, 4.0, 3.3]])
    # Floating sheet like the broken MA export.
    bad = np.random.default_rng(0).normal(loc=(20.5, -0.5, 60.0), scale=2.0, size=(1000, 3))
    with pytest.raises(RuntimeError, match="ENU frame check failed"):
        ma.assert_cloud_near_cameras(bad, cams)

    # Street-level cloud near cameras should pass.
    good = np.random.default_rng(1).normal(loc=(-15.0, 5.0, 4.0), scale=3.0, size=(1000, 3))
    stats = ma.assert_cloud_near_cameras(good, cams)
    assert stats["centroid_delta_m"] < 25.0
    assert stats["z_p50_delta_m"] < 15.0


def test_fuse_predictions_discards_poses(tmp_path: Path) -> None:
    """Back-compat name: predicted poses logged, locked ENU used for fuse."""
    pts_cam = np.zeros((2, 2, 3), dtype=np.float64)
    pts_cam[0, 0] = [1, 0, 3]
    pts_cam[0, 1] = [4, 0, 5]
    pts_cam[1, 0] = [0, 0, 4]
    pts_cam[1, 1] = [1, 0, 6]
    mask = np.array([[True, False], [True, True]], dtype=bool)
    pred_pose = np.eye(4)
    pred_pose[:3, 3] = [100, 0, 0]
    locked = [np.eye(4)]
    locked[0][:3, 3] = [0.0, 0.0, 1.7]
    meta = ma.fuse_predictions_to_ply(
        [{"pts3d_cam": pts_cam, "mask": mask, "camera_poses": pred_pose}],
        tmp_path / "out.ply",
        locked_poses=locked,
    )
    assert meta["points"] == 3
    assert meta["predicted_poses_discarded"] is True
    assert Path(meta["path"]).is_file()
    assert meta["predicted_pose_median_delta_m"] == pytest.approx(
        float(np.linalg.norm([100.0, 0.0, -1.7])), rel=1e-5
    )


def test_cli_export_mapanything_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    create_project(
        ProjectSpec(name="matest", bbox=BBox(52.0, 5.0, 52.001, 5.001)),
        runs_root=tmp_path,
    )
    run = tmp_path / "matest"
    align = run / "align"
    align.mkdir(parents=True, exist_ok=True)
    # cameras.json → load_keyframes
    shots = run / "cropped"
    shots.mkdir(exist_ok=True)
    cams = []
    for i, (e, n) in enumerate(((0.0, 0.0), (6.0, 0.0), (12.0, 1.0))):
        p = shots / f"s{i}.jpg"
        Image.new("RGB", (32, 24), color=(10 * i, 20, 30)).save(p)
        cams.append(
            {
                "shot_path": str(p),
                "e": e,
                "n": n,
                "u": 1.7,
                "heading": 90.0,
                "pitch": 0.0,
                "fov": 90.0,
                "width": 32,
                "height": 24,
                "pano_id": f"p{i}",
            }
        )
    (align / "cameras.json").write_text(json.dumps(cams), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    # open_project looks under default_runs_root — patch via env-like chdir + runs
    # create_project already wrote under tmp_path/matest; point default root.
    monkeypatch.setenv("PS1HOOD_RUNS", str(tmp_path))
    from ps1_hood import project as project_mod

    monkeypatch.setattr(project_mod, "default_runs_root", lambda: tmp_path)

    runner = CliRunner()
    result = runner.invoke(main, ["export", "mapanything-bundle", "matest", "--stride", "1"])
    assert result.exit_code == 0, result.output
    assert "pose_lock=True" in result.output
    bundle = run / "mapanything" / "bundle" / "manifest.json"
    assert bundle.is_file()
    manifest = json.loads(bundle.read_text(encoding="utf-8"))
    ma.assert_bundle_pose_lock(manifest)


def test_cli_densify_mapanything_export_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    create_project(
        ProjectSpec(name="madense", bbox=BBox(52.0, 5.0, 52.001, 5.001)),
        runs_root=tmp_path,
    )
    run = tmp_path / "madense"
    align = run / "align"
    align.mkdir(parents=True, exist_ok=True)
    shots = run / "cropped"
    shots.mkdir(exist_ok=True)
    cams = []
    for i in range(2):
        p = shots / f"d{i}.jpg"
        Image.new("RGB", (40, 30), color=(50, 50, 50)).save(p)
        cams.append(
            {
                "shot_path": str(p),
                "e": float(i) * 5,
                "n": 0.0,
                "u": 1.5,
                "heading": 0.0,
                "pitch": 0.0,
                "fov": 80.0,
                "width": 40,
                "height": 30,
                "pano_id": f"q{i}",
            }
        )
    (align / "cameras.json").write_text(json.dumps(cams), encoding="utf-8")
    from ps1_hood import project as project_mod

    monkeypatch.setattr(project_mod, "default_runs_root", lambda: tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["densify", "madense", "--backend", "mapanything", "--export-only", "--stride", "1"],
    )
    assert result.exit_code == 0, result.output
    assert "export-only" in result.output
    assert (run / "mapanything" / "bundle" / "manifest.json").is_file()


def test_cli_densify_mapanything_no_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    create_project(
        ProjectSpec(name="manocuda", bbox=BBox(52.0, 5.0, 52.001, 5.001)),
        runs_root=tmp_path,
    )
    run = tmp_path / "manocuda"
    align = run / "align"
    align.mkdir(parents=True, exist_ok=True)
    shots = run / "cropped"
    shots.mkdir(exist_ok=True)
    cams = []
    for i in range(2):
        p = shots / f"n{i}.jpg"
        Image.new("RGB", (40, 30), color=(1, 2, 3)).save(p)
        cams.append(
            {
                "shot_path": str(p),
                "e": float(i) * 5,
                "n": 0.0,
                "u": 1.5,
                "heading": 0.0,
                "pitch": 0.0,
                "fov": 80.0,
                "width": 40,
                "height": 30,
                "pano_id": f"n{i}",
            }
        )
    (align / "cameras.json").write_text(json.dumps(cams), encoding="utf-8")
    from ps1_hood import project as project_mod

    monkeypatch.setattr(project_mod, "default_runs_root", lambda: tmp_path)

    def boom(*_a, **_k):  # noqa: ANN001
        raise RuntimeError("MapAnything densify requires CUDA (test)")

    monkeypatch.setattr(ma, "run_mapanything_densify", boom)
    runner = CliRunner()
    result = runner.invoke(
        main, ["densify", "manocuda", "--backend", "mapanything", "--stride", "1"]
    )
    assert result.exit_code == 1
    assert "CUDA" in result.output or "densify failed" in result.output

def test_default_amp_dtype_hip_fp16(monkeypatch: pytest.MonkeyPatch) -> None:
    """When torch.version.hip is set (ROCm), default AMP is fp16 not bf16."""
    import sys
    import types

    fake = types.ModuleType("torch")
    fake.version = types.SimpleNamespace(hip="6.3.42134")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert ma.default_amp_dtype() == "fp16"


def test_default_amp_dtype_cuda_bf16(monkeypatch: pytest.MonkeyPatch) -> None:
    """NVIDIA CUDA builds (no HIP) keep bf16 as the MapAnything default."""
    import sys
    import types

    fake = types.ModuleType("torch")
    fake.version = types.SimpleNamespace(hip=None)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert ma.default_amp_dtype() == "bf16"

