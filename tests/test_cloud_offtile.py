"""Soft support offtile gate (dilated roof∪yard∪street; not Ortho±2 m)."""

from __future__ import annotations

import json
import struct
from pathlib import Path

from ps1_hood.align.georef import (
    dist_xy_to_support,
    load_support_shell_aabbs,
    offtile_ply_against_support,
    offtile_recon_clouds,
    resolve_local_ground_z,
)


def _ascii_ply(path: Path, verts: list[tuple[float, float, float]]) -> None:
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(verts)}",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
    ]
    for x, y, z in verts:
        lines.append(f"{x} {y} {z}")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _bin_ply(path: Path, verts: list[tuple[float, float, float]]) -> None:
    header = (
        b"ply\nformat binary_little_endian 1.0\nelement vertex %d\n"
        b"property float x\nproperty float y\nproperty float z\nend_header\n"
    ) % len(verts)
    body = b"".join(struct.pack("<fff", *v) for v in verts)
    path.write_bytes(header + body)


def _roofs_json(path: Path) -> None:
    payload = {
        "frame": "ENU",
        "cam_u_median": 2.5,
        "shells": [
            {
                "id": "roof_test",
                "kind": "roof",
                "z": 10.0,
                "aabb_enu": [
                    [0.0, 0.0],
                    [10.0, 0.0],
                    [10.0, 10.0],
                    [0.0, 10.0],
                ],
            },
            {
                "id": "yard_test",
                "kind": "yard",
                "z": 0.2,
                "aabb_enu": [
                    [12.0, 0.0],
                    [18.0, 0.0],
                    [18.0, 10.0],
                    [12.0, 10.0],
                ],
            },
            {
                "id": "street_test",
                "kind": "street",
                "z": 0.0,
                "aabb_enu": [
                    [-2.0, -4.0],
                    [20.0, -4.0],
                    [20.0, -1.0],
                    [-2.0, -1.0],
                ],
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_support_includes_yard_street_and_dilate(tmp_path: Path) -> None:
    p = tmp_path / "roofs.json"
    _roofs_json(p)
    base = load_support_shell_aabbs(p, dilate_m=0.0)
    assert len(base) == 3
    kinds = {a["kind"] for a in base}
    assert kinds == {"roof", "yard", "street"}
    dil = load_support_shell_aabbs(p, dilate_m=10.0)
    roof = next(a for a in dil if a["kind"] == "roof")
    assert abs(roof["e_lo"] - (-10.0)) < 1e-9
    assert abs(roof["e_hi"] - 20.0) < 1e-9


def test_resolve_local_ground_uses_cam_u(tmp_path: Path) -> None:
    p = tmp_path / "roofs.json"
    _roofs_json(p)
    assert abs(resolve_local_ground_z(p) - 0.0) < 1e-9  # 2.5 - 2.5


def test_dist_xy_to_support() -> None:
    aabbs = [{"e_lo": 0.0, "e_hi": 10.0, "n_lo": 0.0, "n_hi": 10.0}]
    assert dist_xy_to_support(5.0, 5.0, aabbs) == 0.0
    assert abs(dist_xy_to_support(15.0, 5.0, aabbs) - 5.0) < 1e-9


def test_offtile_keeps_yard_drops_far_sky(tmp_path: Path) -> None:
    roofs = tmp_path / "roofs.json"
    _roofs_json(roofs)
    # dilate 0 so geometry is clear; yard at x=12..18 stays in support
    support = load_support_shell_aabbs(roofs, dilate_m=0.0)
    gnd = resolve_local_ground_z(roofs)  # 0.0
    ply = tmp_path / "cloud.ply"
    _ascii_ply(
        ply,
        [
            (5.0, 5.0, 9.0),  # roof: keep
            (15.0, 5.0, 0.5),  # yard: keep
            (5.0, -2.5, 0.1),  # street: keep
            (5.0, 5.0, 50.0),  # in support, sky ok (in support always keep)
            (100.0, 100.0, 50.0),  # far + sky → drop_z
            (100.0, 100.0, 1.0),  # far ground → drop_far
            (25.0, 5.0, 1.0),  # outside, within far_m=15 fringe → keep
            (25.0, 5.0, 20.0),  # outside, high z → drop_z
        ],
    )
    dest = tmp_path / "cloud_offtile.ply"
    stats = offtile_ply_against_support(
        ply, support, far_m=15.0, z_out_m=8.0, local_ground_z=gnd, dest=dest
    )
    assert stats["dropped_far"] == 1
    assert stats["dropped_z"] == 2
    assert stats["dropped"] == 3
    assert stats["kept"] == 5
    body = dest.read_text(encoding="ascii").split("end_header", 1)[1]
    zs = [float(ln.split()[2]) for ln in body.strip().splitlines() if ln.strip()]
    assert 50.0 in zs  # in-support sky kept
    assert 20.0 not in zs
    xs = [float(ln.split()[0]) for ln in body.strip().splitlines() if ln.strip()]
    assert 15.0 in xs  # yard
    assert 100.0 not in xs


def test_offtile_binary_ply(tmp_path: Path) -> None:
    roofs = tmp_path / "roofs.json"
    _roofs_json(roofs)
    support = load_support_shell_aabbs(roofs, dilate_m=0.0)
    ply = tmp_path / "c.bin.ply"
    _bin_ply(ply, [(5.0, 5.0, 1.0), (100.0, 100.0, 50.0)])
    dest = tmp_path / "out.ply"
    stats = offtile_ply_against_support(
        ply, support, far_m=15.0, z_out_m=8.0, local_ground_z=0.0, dest=dest
    )
    assert stats["kept"] == 1
    assert stats["dropped"] == 1


def test_offtile_recon_sidecar_not_product(tmp_path: Path) -> None:
    recon = tmp_path / "recon"
    recon.mkdir()
    align = tmp_path / "align"
    align.mkdir()
    _roofs_json(recon / "roofs.json")
    src = recon / "cloud.ply"
    _ascii_ply(
        src,
        [
            (5.0, 5.0, 1.0),
            (100.0, 100.0, 50.0),
            (100.0, 100.0, 0.5),
        ],
    )
    raw = src.read_bytes()
    stats = offtile_recon_clouds(
        recon,
        dilate_m=0.0,
        far_m=15.0,
        z_out_m=8.0,
        cam_corridor_m=0.0,
        poses_path=None,
        replace_product=False,
    )
    assert stats["dropped"] == 2
    assert stats["kept"] == 1
    assert (recon / "cloud_offtile.ply").is_file()
    assert src.read_bytes() == raw


def test_offtile_cli_registered_default_opt_in() -> None:
    from click.core import Command
    from ps1_hood import cli as cli_mod

    assert "cloud-offtile" in cli_mod.main.commands
    zc: Command = cli_mod.main.commands["cloud-zclean"]
    opt = next(p for p in zc.params if p.name == "offtile")
    assert opt.default is False


def test_not_hungry_ortho_clip_defaults() -> None:
    """Sacred: offtile must not be Ortho±2 m XY — far/z_out soft gate only."""
    from ps1_hood.align import georef as g

    assert g.DEFAULT_OFFTILE_DILATE_M == 10.0
    assert g.DEFAULT_OFFTILE_FAR_M == 15.0
    assert g.DEFAULT_OFFTILE_Z_OUT_M == 8.0
