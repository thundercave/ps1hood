"""Soft Z gate vs sat roof AABB shells (opt-in; product cloud stays unclipped)."""

from __future__ import annotations

import json
import struct
from pathlib import Path

from ps1_hood.align.georef import (
    load_roof_shell_aabbs,
    zclean_ply_against_roof_aabbs,
    zclean_recon_clouds,
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


def _roofs_json(path: Path, *, shell_z: float = 10.0) -> None:
    payload = {
        "frame": "ENU",
        "shells": [
            {
                "id": "roof_test",
                "kind": "roof",
                "z": shell_z,
                "ground_z": 0.0,
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
                    [20.0, 20.0],
                    [30.0, 20.0],
                    [30.0, 30.0],
                    [20.0, 30.0],
                ],
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_roof_shell_aabbs_inset_and_kind(tmp_path: Path) -> None:
    p = tmp_path / "roofs.json"
    _roofs_json(p)
    roofs = load_roof_shell_aabbs(p, aabb_inset_m=0.5)
    assert len(roofs) == 1
    r = roofs[0]
    assert r["shell_z"] == 10.0
    # 0..10 inset 0.5 → 0.5..9.5
    assert abs(r["e_lo"] - 0.5) < 1e-9
    assert abs(r["e_hi"] - 9.5) < 1e-9


def test_zclean_drops_sky_floater_keeps_yard(tmp_path: Path) -> None:
    """Inside roof AABB: drop z > shell_z+1.5; outside (yard/street): keep."""
    roofs_path = tmp_path / "roofs.json"
    _roofs_json(roofs_path, shell_z=10.0)
    roofs = load_roof_shell_aabbs(roofs_path, aabb_inset_m=0.0)

    ply = tmp_path / "cloud.ply"
    # (5,5) inside roof: ok at 10.5, floater at 12.0 (>10+1.5)
    # (25,25) outside roofs (yard AABB ignored): keep even at z=50
    # (50,50) street: keep
    _ascii_ply(
        ply,
        [
            (5.0, 5.0, 10.5),  # keep
            (5.0, 5.0, 12.0),  # drop
            (25.0, 25.0, 50.0),  # keep (outside roof kind)
            (50.0, 50.0, 3.0),  # keep
        ],
    )
    dest = tmp_path / "cloud_zclean.ply"
    stats = zclean_ply_against_roof_aabbs(ply, roofs, margin_m=1.5, dest=dest)
    assert stats["dropped"] == 1
    assert stats["kept"] == 3
    body = dest.read_text(encoding="ascii")
    assert "12.0" not in body.split("end_header")[-1] or "12.0" not in [
        ln.split()[2] for ln in body.split("end_header")[-1].strip().splitlines() if ln.strip()
    ]
    zs = [
        float(ln.split()[2])
        for ln in body.split("end_header", 1)[1].strip().splitlines()
        if ln.strip()
    ]
    assert 12.0 not in zs
    assert 50.0 in zs
    assert 10.5 in zs


def test_zclean_binary_ply(tmp_path: Path) -> None:
    roofs_path = tmp_path / "roofs.json"
    _roofs_json(roofs_path, shell_z=5.0)
    roofs = load_roof_shell_aabbs(roofs_path, aabb_inset_m=0.0)
    ply = tmp_path / "cloud.bin.ply"
    _bin_ply(ply, [(5.0, 5.0, 4.0), (5.0, 5.0, 9.0)])
    dest = tmp_path / "out.ply"
    stats = zclean_ply_against_roof_aabbs(ply, roofs, margin_m=1.5, dest=dest)
    assert stats["kept"] == 1
    assert stats["dropped"] == 1


def test_zclean_recon_writes_sidecar_not_product(tmp_path: Path) -> None:
    recon = tmp_path / "recon"
    recon.mkdir()
    _roofs_json(recon / "roofs.json", shell_z=8.0)
    src = recon / "cloud.ply"
    _ascii_ply(src, [(5.0, 5.0, 7.0), (5.0, 5.0, 20.0), (100.0, 100.0, 20.0)])
    raw = src.read_bytes()
    stats = zclean_recon_clouds(recon, margin_m=1.5, aabb_inset_m=0.0, replace_product=False)
    assert stats["dropped"] == 1
    assert stats["kept"] == 2
    assert (recon / "cloud_zclean.ply").is_file()
    # Product untouched
    assert src.read_bytes() == raw


def test_zclean_cli_default_off() -> None:
    from click.core import Command
    from ps1_hood import cli as cli_mod

    assert "cloud-zclean" in cli_mod.main.commands
    align: Command = cli_mod.main.commands["align"]
    opt = next(p for p in align.params if p.name == "cloud_zclean")
    assert opt.default is False


def test_sink_drop_optional_off(tmp_path: Path) -> None:
    roofs_path = tmp_path / "roofs.json"
    _roofs_json(roofs_path, shell_z=10.0)
    roofs = load_roof_shell_aabbs(roofs_path, aabb_inset_m=0.0)
    ply = tmp_path / "cloud.ply"
    # deep sink inside roof AABB
    _ascii_ply(ply, [(5.0, 5.0, -5.0)])
    dest = tmp_path / "a.ply"
    stats_off = zclean_ply_against_roof_aabbs(
        ply, roofs, margin_m=1.5, drop_sinks=False, ground_z=0.0, dest=dest
    )
    assert stats_off["kept"] == 1
    dest2 = tmp_path / "b.ply"
    stats_on = zclean_ply_against_roof_aabbs(
        ply, roofs, margin_m=1.5, drop_sinks=True, ground_z=0.0, dest=dest2
    )
    assert stats_on["dropped"] == 1
