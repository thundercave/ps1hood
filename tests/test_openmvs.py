"""Optional OpenMVS densify — mocks when binary absent."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from ps1_hood.cli import main
from ps1_hood.reconstruct import openmvs as om


def _fake_sparse(tmp: Path) -> Path:
    sparse = tmp / "sparse_posed"
    sparse.mkdir(parents=True)
    (sparse / "cameras.txt").write_text("#\n", encoding="ascii")
    (sparse / "images.txt").write_text("#\n", encoding="ascii")
    (sparse / "points3D.bin").write_bytes(b"\x01\x00\x00\x00\x00\x00\x00\x00" + b"\x00" * 64)
    return sparse


def _fake_images(tmp: Path, n: int = 2) -> Path:
    images = tmp / "images"
    images.mkdir(parents=True)
    for i in range(n):
        (images / f"img{i}.jpg").write_bytes(b"\xff\xd8\xff")  # tiny jpeg-ish
    return images


def test_which_openmvs_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(om.shutil, "which", lambda _name: None)
    assert om.which_openmvs() is None
    with pytest.raises(RuntimeError, match="OpenMVS not on PATH"):
        om.require_openmvs()
    assert "AGPL" in om.INSTALL_HINT
    assert "not redistributed" in om.INSTALL_HINT.lower() or "NOT redistributed" in om.INSTALL_HINT


def test_which_openmvs_present(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_which(name: str) -> str | None:
        if name in {"InterfaceCOLMAP", "DensifyPointCloud"}:
            return f"/usr/bin/{name}"
        return None

    monkeypatch.setattr(om.shutil, "which", fake_which)
    bins = om.require_openmvs()
    assert bins["InterfaceCOLMAP"].endswith("InterfaceCOLMAP")
    assert bins["DensifyPointCloud"].endswith("DensifyPointCloud")


def test_densify_argv_smoke_flags() -> None:
    argv = om.densify_argv("/bin/DensifyPointCloud", Path("/tmp/scene.mvs"))
    assert argv[0] == "/bin/DensifyPointCloud"
    assert argv[1] == "/tmp/scene.mvs"
    assert "--resolution-level" in argv
    assert argv[argv.index("--resolution-level") + 1] == "2"
    assert argv[argv.index("--number-views") + 1] == "4"


def test_resolve_posed_inputs_defaults(tmp_path: Path) -> None:
    run = tmp_path / "run"
    colmap = run / "recon" / "colmap"
    images = _fake_images(colmap)
    sparse = _fake_sparse(colmap)
    got_images, got_sparse = om.resolve_posed_inputs(run)
    assert got_images == images
    assert got_sparse == sparse


def test_resolve_posed_inputs_missing_sparse(tmp_path: Path) -> None:
    run = tmp_path / "run"
    colmap = run / "recon" / "colmap"
    _fake_images(colmap)
    (colmap / "sparse_posed").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="posed COLMAP"):
        om.resolve_posed_inputs(run)


def test_resolve_rejects_missing_images(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    with pytest.raises(RuntimeError, match="no COLMAP images"):
        om.resolve_posed_inputs(run)


def test_run_openmvs_densify_mocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = tmp_path / "run"
    colmap = run / "recon" / "colmap"
    _fake_images(colmap)
    sparse = _fake_sparse(colmap)

    def fake_which(name: str) -> str | None:
        mapping = {
            "InterfaceCOLMAP": "/mock/InterfaceCOLMAP",
            "DensifyPointCloud": "/mock/DensifyPointCloud",
            "colmap": "/mock/colmap",
        }
        return mapping.get(name)

    calls: list[list[str]] = []

    def fake_run(cmd: list[str], *, cwd: str | None = None) -> None:  # noqa: ARG001
        calls.append(list(cmd))
        # Simulate side effects of each stage.
        if "image_undistorter" in cmd:
            dense = Path(cmd[cmd.index("--output_path") + 1])
            (dense / "images").mkdir(parents=True)
            (dense / "images" / "a.jpg").write_bytes(b"x")
            (dense / "sparse").mkdir(parents=True)
        elif cmd[0].endswith("InterfaceCOLMAP"):
            out = Path(cmd[cmd.index("-o") + 1])
            out.write_text("mvs", encoding="ascii")
        elif cmd[0].endswith("DensifyPointCloud"):
            # scene.mvs path is argv[1]; write sibling dense ply
            scene = Path(cmd[1])
            ply = scene.with_name("scene_dense.ply")
            ply.write_text(
                "ply\nformat ascii 1.0\nelement vertex 3\n"
                "property float x\nproperty float y\nproperty float z\n"
                "end_header\n0 0 0\n1 0 0\n0 1 0\n",
                encoding="ascii",
            )

    monkeypatch.setattr(om.shutil, "which", fake_which)
    monkeypatch.setattr(om, "_run", fake_run)

    meta = om.run_openmvs_densify(run, resolution_level=2, number_views=4)
    assert meta["backend"] == "openmvs"
    assert meta["points"] == 3
    assert meta["agpl"] is True
    assert Path(meta["path"]).is_file()
    assert Path(meta["seed"]) == sparse
    assert any("image_undistorter" in c for c in calls)
    assert any(c[0].endswith("InterfaceCOLMAP") for c in calls)
    densify_calls = [c for c in calls if c[0].endswith("DensifyPointCloud")]
    assert densify_calls
    assert densify_calls[0][densify_calls[0].index("--resolution-level") + 1] == "2"


def test_run_fails_gracefully_without_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(om.shutil, "which", lambda _n: None)
    run = tmp_path / "run"
    colmap = run / "recon" / "colmap"
    _fake_images(colmap)
    _fake_sparse(colmap)
    with pytest.raises(RuntimeError, match="OpenMVS not on PATH"):
        om.run_openmvs_densify(run)


def test_cli_densify_missing_binary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(om, "which_openmvs", lambda: None)
    # open_project needs a real project.yaml
    from ps1_hood.config import ProjectSpec
    from ps1_hood.geo import BBox
    from ps1_hood.project import create_project

    create_project(
        ProjectSpec(name="dtest", bbox=BBox(52.0, 5.0, 52.001, 5.001)),
        runs_root=tmp_path,
    )
    monkeypatch.chdir(tmp_path)
    # Patch which_openmvs where CLI imports it
    import ps1_hood.reconstruct.openmvs as om_mod

    monkeypatch.setattr(om_mod, "which_openmvs", lambda: None)
    runner = CliRunner()
    # CLI open_project looks under cwd/runs
    result = runner.invoke(main, ["densify", "dtest"])
    assert result.exit_code == 1
    assert "AGPL" in result.output or "OpenMVS" in result.output


def test_ply_vertex_count(tmp_path: Path) -> None:
    ply = tmp_path / "t.ply"
    ply.write_text(
        "ply\nformat ascii 1.0\nelement vertex 42\nend_header\n",
        encoding="ascii",
    )
    assert om._ply_vertex_count(ply) == 42


def test_densify_argv_view_neighbors() -> None:
    argv = om.densify_argv(
        "/bin/DensifyPointCloud",
        Path("/tmp/scene.mvs"),
        view_neighbors_file=Path("/tmp/nbrs.txt"),
    )
    assert "--view-neighbors-file" in argv
    assert argv[argv.index("--view-neighbors-file") + 1] == "/tmp/nbrs.txt"


def test_write_view_neighbors_from_match_list(tmp_path: Path) -> None:
    images_txt = tmp_path / "images.txt"
    images_txt.write_text(
        "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        "1 1 0 0 0 0 0 0 1 a.jpg\n"
        "\n"
        "2 1 0 0 0 1 0 0 1 b.jpg\n"
        "\n"
        "3 1 0 0 0 2 0 0 1 c.jpg\n"
        "\n",
        encoding="ascii",
    )
    pairs = tmp_path / "pairs.txt"
    pairs.write_text("a.jpg b.jpg\nb.jpg c.jpg\n", encoding="ascii")
    out = tmp_path / "nbrs.txt"
    n = om.write_view_neighbors_from_match_list(images_txt, pairs, out)
    assert n == 3
    lines = {ln.split()[0]: ln.split()[1:] for ln in out.read_text().splitlines() if ln}
    assert lines["1"] == ["2"]
    assert set(lines["2"]) == {"1", "3"}
    assert lines["3"] == ["2"]
