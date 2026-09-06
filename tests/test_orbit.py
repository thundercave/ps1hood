from __future__ import annotations

from ps1_hood.capture.google_js import orbit_headings


def test_full_circle_45() -> None:
    assert orbit_headings(45) == [0, 45, 90, 135, 180, 225, 270, 315]


def test_includes_travel_heading() -> None:
    hs = orbit_headings(45, 12.0)
    assert 12 in hs
    assert 0 in hs
    assert len(hs) == 9


def test_does_not_duplicate_cardinal_travel() -> None:
    assert orbit_headings(45, 90.0) == [0, 45, 90, 135, 180, 225, 270, 315]
