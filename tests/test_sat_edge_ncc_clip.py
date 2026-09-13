"""Sat edge+NCC fuse + Ortho ENU floater clip (opt-in; prefer unclipped product)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from ps1_hood.align.georef import clip_ply_to_ortho_enu
from ps1_hood.align.sat_edges import edge_agree, ortho_canny
from ps1_hood.align.satellite_align import fuse_sat_score, ncc, render_satellite_into_camera_fast
from ps1_hood.capture.satellite import Ortho
from ps1_hood.geo import BBox, LocalFrame


def _stripe_ortho(frame: LocalFrame, *, shift_e: float = 0.0) -> Ortho:
    """Synthetic Ortho: vertical bright stripe (road) + horizontal (roof)."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[:, :] = (40, 40, 40)
    # road stripe ~ centre
    c = 100 + int(round(shift_e * 2))  # ~0.5 m/px rough
    img[:, max(0, c - 4) : min(200, c + 4)] = (180, 180, 180)
    # roof ridge horizontal
    img[80:84, 40:160] = (200, 160, 120)
    bbox = BBox(52.0895, 5.1192, 52.09, 5.12)
    return Ortho(img, bbox, frame)


def test_fuse_sat_score_weights() -> None:
    from ps1_hood.align.satellite_align import fuse_sat_score

    assert abs(fuse_sat_score(0.4, 0.8, w_ncc=0.35, w_edge=0.65) - (0.35 * 0.4 + 0.65 * 0.8)) < 1e-9
    # missing edge → pure ncc
    assert abs(fuse_sat_score(0.5, -1.0, w_ncc=0.35, w_edge=0.65) - 0.5) < 1e-9
    assert fuse_sat_score(-1.0, -1.0) == -1.0


def test_edge_agree_prefers_aligned_shift() -> None:
    """Fused edge term peaks when photo edges line up with remapped Ortho stripe."""
    from ps1_hood.align.sat_edges import ortho_edge_bgr
    from ps1_hood.align.satellite_align import fuse_sat_score

    bbox = BBox(52.0895, 5.1192, 52.09, 5.12)
    frame = LocalFrame.from_bbox(bbox)
    ortho = _stripe_ortho(frame, shift_e=0.0)
    # Camera looking north; ground band sees the stripe
    e0 = 0.5 * (ortho.sw + ortho.ee)
    n0 = 0.5 * (ortho.sh + ortho.nn) - 8.0
    heading = 0.0
    pitch = -5.0
    h_m = 2.5
    fov = 90.0
    w, h = 160, 120

    # Build a "photo" as the synth at truth pose (perfect match)
    truth = render_satellite_into_camera_fast(
        ortho, e=e0, n=n0, heading=heading, pitch=pitch, height_m=h_m, width=w, height=h, fov_deg=fov
    )
    # Add Canny-friendly contrast already present
    edge_src = ortho_edge_bgr(ortho)

    def score_at(de: float) -> tuple[float, float, float]:
        synth = render_satellite_into_camera_fast(
            ortho,
            e=e0 + de,
            n=n0,
            heading=heading,
            pitch=pitch,
            height_m=h_m,
            width=w,
            height=h,
            fov_deg=fov,
        )
        synth_e = render_satellite_into_camera_fast(
            ortho,
            e=e0 + de,
            n=n0,
            heading=heading,
            pitch=pitch,
            height_m=h_m,
            width=w,
            height=h,
            fov_deg=fov,
            image=edge_src,
        )
        ncc_v = ncc(truth, synth)
        edge_v = edge_agree(truth, synth_e)
        fused = fuse_sat_score(ncc_v, edge_v, w_ncc=0.35, w_edge=0.65)
        return fused, ncc_v, edge_v

    s0, n0s, e0s = score_at(0.0)
    s_bad, n_bad, e_bad = score_at(6.0)
    # Aligned pose should beat a 6 m lateral miss on fused and preferably edge
    assert s0 > s_bad
    if e0s > -0.5 and e_bad > -0.5:
        assert e0s >= e_bad - 1e-6


def test_ortho_canny_detects_stripe() -> None:
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, 30:34] = 200
    edges = ortho_canny(img, low=40, high=120)
    assert int((edges > 0).sum()) > 10


def test_clip_ply_drops_outside_enu(tmp_path: Path) -> None:
    ply = tmp_path / "cloud.ply"
    # 3 inside, 2 outside
    verts = [
        (0.0, 0.0, 1.0),
        (1.0, 1.0, 2.0),
        (2.0, -1.0, 0.5),
        (50.0, 0.0, 1.0),  # out
        (0.0, -40.0, 1.0),  # out
    ]
    body = "\n".join(f"{x} {y} {z}" for x, y, z in verts) + "\n"
    header = (
        "ply\nformat ascii 1.0\nelement vertex 5\n"
        "property float x\nproperty float y\nproperty float z\nend_header\n"
    )
    ply.write_text(header + body)
    dest = tmp_path / "cloud_satclipped.ply"
    stats = clip_ply_to_ortho_enu(ply, 0.0, -5.0, 10.0, 5.0, margin_m=2.0, dest=dest)
    assert stats["kept"] == 3
    assert stats["dropped"] == 2
    assert dest.is_file()
    # reload vertex count
    text = dest.read_text()
    assert "element vertex 3" in text
    lines = [ln for ln in text.splitlines() if ln and not ln[0].isalpha() and not ln.startswith("end")]
    # after end_header
    after = text.split("end_header", 1)[1].strip().splitlines()
    assert len(after) == 3


def test_clip_ply_binary(tmp_path: Path) -> None:
    import struct

    ply = tmp_path / "cloud.bin.ply"
    verts = [(0.0, 0.0, 1.0), (100.0, 0.0, 1.0)]
    header = (
        b"ply\nformat binary_little_endian 1.0\nelement vertex 2\n"
        b"property float x\nproperty float y\nproperty float z\nend_header\n"
    )
    body = b"".join(struct.pack("<fff", *v) for v in verts)
    ply.write_bytes(header + body)
    stats = clip_ply_to_ortho_enu(ply, -1.0, -1.0, 1.0, 1.0, margin_m=0.0)
    assert stats["kept"] == 1
    assert stats["dropped"] == 1


def test_cloud_clip_sat_default_off() -> None:
    """Pipeline / CLI default: cloud clip is opt-in (False), not clip-on."""
    from click.core import Command
    from ps1_hood import cli as cli_mod

    align: Command = cli_mod.main.commands["align"]
    opt = next(p for p in align.params if p.name == "cloud_clip_sat")
    assert opt.default is False

    run: Command = cli_mod.main.commands["run"]
    opt_run = next(p for p in run.params if p.name == "cloud_clip_sat")
    assert opt_run.default is False
