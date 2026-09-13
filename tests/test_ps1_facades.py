"""PS1 façade simplify — Manhattan rect + RGB555."""

from __future__ import annotations

import numpy as np

from ps1_hood.reconstruct.ps1_facades import (
    manhattan_rectify_planes,
    nearest_resize,
    ps1_quantize,
)


def test_manhattan_rectify_opposite_edges_equal() -> None:
    # Slightly non-rectangular / skewed quad → force true rectangle
    pl = {
        "nx": 0.0,
        "ny": 1.0,
        "n": [0.0, 1.0, 0.0],
        "d": 5.0,  # n_xy · xy = 5 → y=5
        "center": [0.0, 5.0, 5.0],
        "width_m": 8.0,
        "height_m": 6.0,
        "quad": [
            (-4.2, 5.1, 2.0),
            (4.1, 4.9, 1.8),
            (3.9, 5.2, 8.2),
            (-3.8, 4.8, 7.9),
        ],
        "zncc": 0.5,
        "source": "test",
    }
    out = manhattan_rectify_planes([pl])
    assert len(out) == 1
    q = np.asarray(out[0]["quad"], dtype=np.float64)
    assert q.shape == (4, 3)
    # Opposite edges equal length (BL-BR == TL-TR, BL-TL == BR-TR)
    e_bottom = float(np.linalg.norm(q[1] - q[0]))
    e_top = float(np.linalg.norm(q[2] - q[3]))
    e_left = float(np.linalg.norm(q[3] - q[0]))
    e_right = float(np.linalg.norm(q[2] - q[1]))
    assert abs(e_bottom - e_top) < 1e-6
    assert abs(e_left - e_right) < 1e-6
    # Right edges horizontal in XY (Manhattan vertical wall): z differs only on vertical edges
    assert abs(q[0][2] - q[1][2]) < 1e-6
    assert abs(q[3][2] - q[2][2]) < 1e-6
    # Center XY preserved (photo lock)
    c = np.asarray(out[0]["center"], dtype=np.float64)
    assert abs(c[0] - 0.0) < 1e-5
    assert abs(c[1] - 5.0) < 1e-5
    # Normal kept horizontal unit
    n = np.asarray(out[0]["n"], dtype=np.float64)
    assert abs(n[2]) < 1e-9
    assert abs(np.linalg.norm(n[:2]) - 1.0) < 1e-6


def test_ps1_quantize_rgb555_bits() -> None:
    img = np.array(
        [[[7, 15, 31], [255, 128, 64], [1, 2, 3]]],
        dtype=np.uint8,
    )
    out = ps1_quantize(img)
    # Low 3 bits cleared
    assert np.all((out & 0x07) == 0)
    assert out[0, 0, 0] == 0
    assert out[0, 0, 1] == 8
    assert out[0, 0, 2] == 24
    assert out[0, 1, 0] == 248
    assert out[0, 1, 1] == 128
    assert out[0, 1, 2] == 64


def test_nearest_resize_square_or_256() -> None:
    sq = np.zeros((200, 200, 3), dtype=np.uint8)
    out = nearest_resize(sq, tex_size=128)
    assert out.shape[0] == 128 and out.shape[1] == 128

    wide = np.zeros((100, 300, 3), dtype=np.uint8)
    out_w = nearest_resize(wide, tex_size=128)
    assert out_w.shape == (128, 256, 3)

    tall = np.zeros((300, 100, 3), dtype=np.uint8)
    out_t = nearest_resize(tall, tex_size=128)
    assert out_t.shape == (256, 128, 3)
