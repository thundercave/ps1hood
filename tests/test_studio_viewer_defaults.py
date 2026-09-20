"""Studio product viewer layer defaults (hero = raw MA cloud)."""

from __future__ import annotations

from pathlib import Path

VIEWER = Path(__file__).resolve().parents[1] / "src" / "ps1_hood" / "studio" / "static" / "viewer.html"


def test_viewer_hero_layer_defaults() -> None:
    body = VIEWER.read_text(encoding="utf-8")
    # Raw MA cloud: checkbox checked + Points visible after load
    assert 'id="togCloud" checked' in body
    assert "cloud (raw MA)" in body
    assert "cloudPoints.visible = true" in body
    # Façades / sat roofs / street OFF (overlay toggles; files stay on disk)
    assert 'id="togFacades">' in body or 'id="togFacades" ' in body
    assert 'id="togFacades" checked' not in body
    assert 'id="togRoofs">' in body or 'id="togRoofs" ' in body
    assert 'id="togRoofs" checked' not in body
    assert 'id="togStreet">' in body or 'id="togStreet" ' in body
    assert 'id="togStreet" checked' not in body
    assert "facadeRoot.visible = false" in body
    assert "roofRoot.visible = false" in body
    assert "streetRoot.visible = false" in body
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
    # Sculpt auto-enables façades when entering sculpt mode
    assert "togFacades" in body
    assert "facadeRoot.visible = true" in body  # sculpt path
