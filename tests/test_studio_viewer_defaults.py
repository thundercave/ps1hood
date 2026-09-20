"""Studio product viewer layer defaults (hero = façades + sat roofs)."""

from __future__ import annotations

from pathlib import Path

VIEWER = Path(__file__).resolve().parents[1] / "src" / "ps1_hood" / "studio" / "static" / "viewer.html"


def test_viewer_hero_layer_defaults() -> None:
    body = VIEWER.read_text(encoding="utf-8")
    # Raw MA cloud: checkbox unchecked + Points hidden after load
    assert 'id="togCloud" checked' not in body
    assert 'id="togCloud">' in body
    assert "cloud (raw MA)" in body
    assert "cloudPoints.visible = false" in body
    # Façades + sat roofs ON
    assert 'id="togFacades" checked' in body
    assert 'id="togRoofs" checked' in body
    # BAG / zclean / offtile OFF
    assert 'id="togBag">' in body
    assert 'id="togBag" checked' not in body
    assert 'id="togZclean">' in body
    assert 'id="togZclean" checked' not in body
    assert 'id="togOfftile">' in body
    assert 'id="togOfftile" checked' not in body
    assert "bagGroup.visible = false" in body
    assert "zcleanPoints.visible = false" in body
    assert "offtilePoints.visible = false" in body
