from __future__ import annotations

from ps1_hood.capture.google_web import looks_like_streetview, parse_maps_url, streetview_url


def test_streetview_url_has_consumer_params() -> None:
    url = streetview_url(52.0907, 5.1214, heading=90, pitch=-20, fov=80)
    assert "map_action=pano" in url
    assert "viewpoint=52.0907000,5.1214000" in url
    assert "heading=90.00" in url
    assert "pano=" not in url


def test_streetview_url_includes_real_pano() -> None:
    url = streetview_url(52.09, 5.12, pano_id="tu510ie_z4ptBZYo2BGEJg")
    assert "pano=tu510ie_z4ptBZYo2BGEJg" in url
    assert "pano=seed-" not in streetview_url(52.09, 5.12, pano_id="seed-0001")


def test_parse_hash_url() -> None:
    url = (
        "https://www.google.com/maps/@52.0901234,5.1219876,3a,75y,180h,70t"
        "/data=!3m6!1e1!3m4!1s_R1mwpMkiqa2p0zp48EBJg!2e0!7i16384!8i8192"
    )
    meta = parse_maps_url(url)
    assert meta["pano_id"] == "_R1mwpMkiqa2p0zp48EBJg"
    assert abs(meta["lat"] - 52.0901234) < 1e-9
    assert abs(meta["lon"] - 5.1219876) < 1e-9
    assert abs(meta["heading"] - 180) < 1e-9
    assert abs(meta["pitch"] - (-20)) < 1e-9
    assert looks_like_streetview(url)


def test_parse_api1_url() -> None:
    url = streetview_url(48.85, 2.29, pano_id="tu510ie_z4ptBZYo2BGEJg")
    meta = parse_maps_url(url)
    assert meta["pano_id"] == "tu510ie_z4ptBZYo2BGEJg"
    # share URL is only a request — loaded SV rewrites to data=!3m…!1e1
    assert not looks_like_streetview(url)
