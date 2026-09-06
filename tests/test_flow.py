from __future__ import annotations

import numpy as np

from ps1_hood.interpolate.flow import interpolate_pair


def test_interpolate_pair_count_and_shape() -> None:
    a = np.zeros((40, 60, 3), dtype=np.uint8)
    b = np.zeros((40, 60, 3), dtype=np.uint8)
    a[:, :30] = (20, 40, 200)
    b[:, 15:45] = (20, 40, 200)
    frames = interpolate_pair(a, b, steps=3)
    assert len(frames) == 5  # start + 3 mids + end
    assert frames[0].shape == a.shape
    assert frames[-1].shape == b.shape
