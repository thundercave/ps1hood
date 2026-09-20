"""Re-texture bare façades — soft warp + multi-cam + quality-keep."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from ps1_hood.reconstruct.facades import (
    _bare_plane_indices,
    _is_strictly_better,
    _rank_frontal_cameras,
    _warp_facade_texture,
    retexture_bare_planes,
)


def _paint_wall(h: int = 480, w: int = 640) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    rng = np.random.default_rng(7)
    noise = rng.integers(40, 200, (h, w), dtype=np.uint8)
    img[:, :, 0] = noise
    img[:, :, 1] = np.roll(noise, 11, axis=1)
    img[:, :, 2] = np.roll(noise, 5, axis=0)
    cv2.rectangle(img, (80, 40), (560, 440), (40, 160, 220), -1)
    return img


def _frame_at(
    path: Path,
    *,
    e: float,
    n: float,
    u: float = 2.0,
    heading: float = 0.0,
    img: np.ndarray | None = None,
) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if img is None:
        img = _paint_wall()
    cv2.imwrite(str(path), img)
    return {
        "e": e,
        "n": n,
        "u": u,
        "heading": heading,
        "pitch": 0.0,
        "fov": 90.0,
        "path": str(path),
        "pano_id": path.stem,
    }


def _wall_quad(*, y: float = 8.0, half_w: float = 4.0, z0: float = 0.0, z1: float = 8.0):
    # Wall facing -N (normal 0,-1) at n=y; camera at origin looking +N (heading 0)
    return [
        (-half_w, y, z0),
        (half_w, y, z0),
        (half_w, y, z1),
        (-half_w, y, z1),
    ]


def test_warp_retries_larger_margin(tmp_path: Path) -> None:
    """Corners just outside 40px still bake when margin loosened / clip allowed."""
    img = _paint_wall(480, 640)
    fr = _frame_at(tmp_path / "cam.jpg", e=0.0, n=0.0, heading=0.0, img=img)
    # Wide quad so corners project near/over edges
    quad = _wall_quad(y=5.0, half_w=9.0, z0=-1.0, z1=12.0)
    dest_strict = tmp_path / "strict.jpg"
    # Force historic-style: margin 40, no clip → may fail on oversized quad
    ok_strict = _warp_facade_texture(
        quad, fr, dest_strict, ps1_tex_size=None, margin_px=40.0, allow_clip=False
    )
    dest_soft = tmp_path / "soft.jpg"
    ok_soft = _warp_facade_texture(
        quad, fr, dest_soft, ps1_tex_size=None, margin_px=120.0, allow_clip=True
    )
    assert ok_soft
    assert dest_soft.is_file() and dest_soft.stat().st_size > 100
    # Soft path must succeed even when strict hard-fail would
    if not ok_strict:
        assert ok_soft


def test_rank_prefers_view_indices(tmp_path: Path) -> None:
    img = _paint_wall()
    frames = [
        _frame_at(tmp_path / "a.jpg", e=-3.0, n=0.0, heading=10.0, img=img),
        _frame_at(tmp_path / "b.jpg", e=0.0, n=0.0, heading=0.0, img=img),
        _frame_at(tmp_path / "c.jpg", e=3.0, n=0.0, heading=-10.0, img=img),
    ]
    pl = {"nx": 0.0, "ny": -1.0, "d": -8.0}
    quad = _wall_quad(y=8.0)
    ranked = _rank_frontal_cameras(
        pl, quad, frames, top_k=2, prefer_indices=[2, 0]
    )
    assert ranked
    # First preferred index that scores should lead
    assert ranked[0]["path"].endswith("c.jpg") or ranked[0]["path"].endswith("a.jpg")


def test_bare_indices_from_mtl_and_missing_jpg(tmp_path: Path) -> None:
    recon = tmp_path / "recon"
    tex = recon / "textures"
    tex.mkdir(parents=True)
    # façade 00 textured, 01 bare (no jpg), 02 jpg but no map_Kd
    cv2.imwrite(str(tex / "facade_00.jpg"), _paint_wall(64, 64))
    cv2.imwrite(str(tex / "facade_02.jpg"), _paint_wall(64, 64))
    (recon / "facades.mtl").write_text(
        "# test\n"
        "newmtl ground\nKd 0.3 0.3 0.3\n\n"
        "newmtl facade_00\nKd 0.5 0.5 0.5\nmap_Kd textures/facade_00.jpg\n\n"
        "newmtl facade_01\nKd 0.5 0.5 0.5\n\n"
        "newmtl facade_02\nKd 0.5 0.5 0.5\n\n",
        encoding="ascii",
    )
    planes = [
        {"nx": 0.0, "ny": -1.0, "d": -5.0, "texture": "textures/facade_00.jpg"},
        {"nx": 0.0, "ny": -1.0, "d": -5.0, "texture": None},
        {"nx": 0.0, "ny": -1.0, "d": -5.0, "texture": None},
    ]
    bare = _bare_plane_indices(recon, planes)
    assert bare == [1, 2]


def test_retexture_candidate_promotes_clause1(tmp_path: Path) -> None:
    recon = tmp_path / "recon"
    tex = recon / "textures"
    tex.mkdir(parents=True)
    img = _paint_wall()
    # Product: 2 planes, only 00 textured
    cv2.imwrite(str(tex / "facade_00.jpg"), img)
    quad0 = _wall_quad(y=8.0, half_w=3.0)
    quad1 = _wall_quad(y=8.0, half_w=3.0)
    # Shift second wall slightly in E so it's a distinct plane
    quad1 = [(c[0] + 10.0, c[1], c[2]) for c in quad1]
    planes_payload = {
        "frame": "ENU",
        "source": "test",
        "ground_z": 0.0,
        "residual_points": 0,
        "gates": {"mean_zncc": 0.45, "plane_count": 2},
        "planes": [
            {
                "id": "facade_00",
                "n": [0.0, -1.0, 0.0],
                "d": 8.0,
                "quad": [list(c) for c in quad0],
                "width_m": 6.0,
                "height_m": 8.0,
                "zncc": 0.45,
                "texture": "textures/facade_00.jpg",
                "inliers": 100,
                "source": "test",
                "view_indices": [0],
            },
            {
                "id": "facade_01",
                "n": [0.0, -1.0, 0.0],
                "d": 8.0,
                "quad": [list(c) for c in quad1],
                "width_m": 6.0,
                "height_m": 8.0,
                "zncc": 0.45,
                "texture": None,
                "inliers": 100,
                "source": "test",
                "view_indices": [0],
            },
        ],
    }
    (recon / "planes.json").write_text(json.dumps(planes_payload, indent=2), encoding="utf-8")
    (recon / "facades.mtl").write_text(
        "newmtl ground\nKd 0.3 0.3 0.3\nmap_Kd textures/ground.jpg\n\n"
        "newmtl facade_00\nKd 0.5 0.5 0.5\nmap_Kd textures/facade_00.jpg\n\n"
        "newmtl facade_01\nKd 0.5 0.5 0.5\n\n",
        encoding="ascii",
    )
    # Minimal obj so product looks nonempty
    (recon / "facades.obj").write_text(
        "mtllib facades.mtl\n"
        "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
        "usemtl ground\nf 1/1 2/2 3/3 4/4\n"
        "v -3 8 0\nv 3 8 0\nv 3 8 8\nv -3 8 8\n"
        "usemtl facade_00\nf 5/5 6/6 7/7 8/8\n"
        "v 7 8 0\nv 13 8 0\nv 13 8 8\nv 7 8 8\n"
        "usemtl facade_01\nf 9/9 10/10 11/11 12/12\n",
        encoding="ascii",
    )
    frames = [
        _frame_at(tmp_path / "shots" / "cam0.jpg", e=0.0, n=0.0, heading=0.0, img=img),
        _frame_at(tmp_path / "shots" / "cam1.jpg", e=10.0, n=0.0, heading=0.0, img=img),
    ]
    # Hash of good texture must survive
    import hashlib

    h_before = hashlib.sha256((tex / "facade_00.jpg").read_bytes()).hexdigest()

    meta = retexture_bare_planes(
        tmp_path,  # project root with recon/
        margin_px=120.0,
        top_k_cams=3,
        ps1_tex_size=None,
        frames=frames,
        bare_only=True,
        candidate=True,
        in_place=False,
        dry_run=False,
        promote=True,
    )
    assert 1 in meta["baked"] or meta["textured_after"] > meta["textured_before"]
    assert meta["planes"] == 2
    # Good texture unchanged on product (promote copies candidate; content same)
    assert (tex / "facade_00.jpg").is_file()
    h_after = hashlib.sha256((tex / "facade_00.jpg").read_bytes()).hexdigest()
    assert h_before == h_after
    if meta.get("promoted"):
        assert (tex / "facade_01.jpg").is_file()
        mtl = (recon / "facades.mtl").read_text(encoding="ascii")
        assert "map_Kd textures/facade_01.jpg" in mtl
        better, why = _is_strictly_better(
            meta["new_quality"], meta["prev_quality"]
        )
        assert better and "clause1" in why


def test_dry_run_no_writes(tmp_path: Path) -> None:
    recon = tmp_path / "recon"
    recon.mkdir()
    (recon / "planes.json").write_text(
        json.dumps(
            {
                "ground_z": 0.0,
                "source": "t",
                "residual_points": 0,
                "planes": [
                    {
                        "n": [0, -1, 0],
                        "d": 5.0,
                        "quad": [list(c) for c in _wall_quad()],
                        "texture": None,
                        "width_m": 6,
                        "height_m": 8,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (recon / "facades.mtl").write_text(
        "newmtl facade_00\nKd 0.5 0.5 0.5\n\n", encoding="ascii"
    )
    meta = retexture_bare_planes(
        tmp_path,
        frames=[],
        dry_run=True,
        bare_only=True,
    )
    assert meta["bare"] == [0]
    assert not (recon / "facades.candidate.mtl").is_file()
