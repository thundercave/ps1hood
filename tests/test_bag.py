from __future__ import annotations

from ps1_hood.capture.bag import _triangulate_pts, zip_status


def test_triangulate_pts() -> None:
    pts = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    tris = _triangulate_pts(pts)
    assert tris == [(0, 1, 2), (0, 2, 3)]


def test_zip_status_reports_path() -> None:
    st = zip_status()
    assert "path" in st
    assert "percent" in st
    assert st["expected"] > 0
