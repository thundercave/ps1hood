"""Studio ENU façade sculpt — translate along n, resize, bak/undo."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from ps1_hood.reconstruct.sculpt import (
    MAX_DELTA_D_M,
    apply_sculpt,
    corners_from_center,
    mutate_plane,
    undo_sculpt,
)


def _plane(*, n=(0.0, 1.0, 0.0), width=8.0, height=6.0, i=0, center=None):
    c = np.array(center if center is not None else [float(i) * 3.0, 5.0, 4.0])
    quad = corners_from_center(c, np.array(n), width, height)
    center_v = np.mean(np.asarray(quad, dtype=np.float64), axis=0)
    nn = np.array([n[0], n[1], 0.0], dtype=np.float64)
    nn = nn / (np.linalg.norm(nn) + 1e-12)
    d = float(-nn @ center_v)
    return {
        "id": f"facade_{i:02d}",
        "n": [float(nn[0]), float(nn[1]), 0.0],
        "d": d,
        "quad": [list(map(float, q)) for q in quad],
        "width_m": float(width),
        "height_m": float(height),
        "zncc": 0.5,
        "texture": f"textures/facade_{i:02d}.jpg",
        "inliers": 10,
        "source": "test",
    }


def _write_mini_product(recon: Path, n_planes: int = 3) -> None:
    recon.mkdir(parents=True, exist_ok=True)
    planes = [
        _plane(i=i, n=(0.0, 1.0, 0.0) if i % 2 == 0 else (1.0, 0.0, 0.0))
        for i in range(n_planes)
    ]
    payload = {
        "frame": "ENU",
        "source": "test",
        "ground_z": 0.0,
        "planes": planes,
        "residual_points": 0,
        "gates": {"mean_zncc": 0.5, "plane_count": n_planes},
    }
    (recon / "planes.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = ["# test", "mtllib facades.mtl"]
    for v in [(-10, -10, 0), (10, -10, 0), (10, 10, 0), (-10, 10, 0)]:
        lines.append(f"v {v[0]} {v[1]} {v[2]}")
    for pl in planes:
        for c in pl["quad"]:
            lines.append(f"v {c[0]:.3f} {c[1]:.3f} {c[2]:.3f}")
    for _ in range(1 + n_planes):
        lines.append("vt 0 0")
        lines.append("vt 1 0")
        lines.append("vt 1 1")
        lines.append("vt 0 1")
    lines.append("usemtl ground")
    lines.append("f 1/1 2/2 3/3 4/4")
    for i in range(n_planes):
        base = 5 + i * 4
        lines.append(f"usemtl facade_{i:02d}")
        lines.append(
            f"f {base}/{base} {base+1}/{base+1} {base+2}/{base+2} {base+3}/{base+3}"
        )
    (recon / "facades.obj").write_text("\n".join(lines) + "\n", encoding="ascii")
    mtl = ["# test mtl", "newmtl ground", "Kd 0.3 0.3 0.3", ""]
    for i in range(n_planes):
        mtl += [
            f"newmtl facade_{i:02d}",
            "Kd 0.5 0.5 0.5",
            f"map_Kd textures/facade_{i:02d}.jpg",
            "",
        ]
    (recon / "facades.mtl").write_text("\n".join(mtl), encoding="ascii")
    tex = recon / "textures"
    tex.mkdir(exist_ok=True)
    for i in range(n_planes):
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        img[:] = (40 + i * 20, 80, 120)
        cv2.imwrite(str(tex / f"facade_{i:02d}.jpg"), img)


def test_mutate_translate_preserves_winding_and_clamps():
    pl = _plane()
    q0 = [tuple(c) for c in pl["quad"]]
    out = mutate_plane(pl, delta_d=1.0)
    assert abs(out["center"][1] - 6.0) < 1e-6
    assert abs(out["quad"][0][0] - q0[0][0]) < 1e-6
    assert out["sculpt_edit"] is True
    big = mutate_plane(pl, delta_d=99.0)
    assert abs(big["center"][1] - (5.0 + MAX_DELTA_D_M)) < 1e-6


def test_mutate_resize_about_center():
    pl = _plane()
    out = mutate_plane(pl, width_m=10.0, height_m=7.0)
    assert abs(out["width_m"] - 10.0) < 1e-6
    assert abs(out["height_m"] - 7.0) < 1e-6
    assert abs(out["center"][0]) < 1e-6
    assert abs(out["center"][1] - 5.0) < 1e-6


def test_apply_sculpt_baks_and_undo_restores(tmp_path: Path):
    recon = tmp_path / "recon"
    _write_mini_product(recon, n_planes=3)
    before = json.loads((recon / "planes.json").read_text(encoding="utf-8"))
    assert len(before["planes"]) == 3

    meta = apply_sculpt(
        recon,
        "facade_01",
        delta_d=0.4,
        bake=False,
        frames=[],
    )
    assert meta["ok"] is True
    assert meta["plane_count"] == 3
    assert meta["bak"]["stamp"]
    after = json.loads((recon / "planes.json").read_text(encoding="utf-8"))
    assert len(after["planes"]) == 3
    assert after["planes"][1].get("sculpt_edit") is True
    assert after["planes"][0]["id"] == "facade_00"
    assert after["planes"][2]["id"] == "facade_02"
    # facade_01 has n=+X; center should move +0.4 along east
    before_cx = float(np.mean([c[0] for c in before["planes"][1]["quad"]]))
    assert after["planes"][1]["center"][0] == pytest.approx(before_cx + 0.4, abs=1e-3)

    undo = undo_sculpt(recon)
    assert undo["ok"] is True
    restored = json.loads((recon / "planes.json").read_text(encoding="utf-8"))
    assert len(restored["planes"]) == 3
    assert restored["planes"][1].get("sculpt_edit") is not True
    assert restored["planes"][1]["d"] == pytest.approx(before["planes"][1]["d"])


def test_apply_sculpt_refuses_missing_plane(tmp_path: Path):
    recon = tmp_path / "recon"
    _write_mini_product(recon, n_planes=2)
    with pytest.raises(KeyError):
        apply_sculpt(recon, "facade_99", delta_d=0.1, bake=False, frames=[])


def test_viewer_has_sculpt_controls():
    viewer = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "ps1_hood"
        / "studio"
        / "static"
        / "viewer.html"
    )
    body = viewer.read_text(encoding="utf-8")
    assert "TransformControls" in body
    assert 'id="togSculpt"' in body
    assert "SCULPT_MAX_D" in body
    assert "/sculpt" in body
    assert "planes.json" in body
    assert "setMode('rotate')" not in body
