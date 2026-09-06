from __future__ import annotations

from ps1_hood.geo import haversine_m, rd_to_wgs84, wgs84_to_rd


def test_rd_roundtrip_utrecht() -> None:
    lat0, lon0 = 52.0908, 5.1214
    x, y = wgs84_to_rd(lat0, lon0)
    lat1, lon1 = rd_to_wgs84(x, y)
    assert haversine_m(lat0, lon0, lat1, lon1) < 0.05
    assert 135000 < x < 138000
    assert 454000 < y < 457000


def test_rd_amersfoort_origin() -> None:
    lat, lon = rd_to_wgs84(155000, 463000)
    # official RD origin; PROJ may differ from the leaflet by ~0.5 m (ETRS89 vs WGS84)
    assert abs(lat - 52.15517440) < 1e-5
    assert abs(lon - 5.38720621) < 1e-5


def test_rd_roundtrip_haarlem_and_groningen() -> None:
    # Kadaster polynomials were 70–400 m off here; PROJ must stay sub-metre.
    for lat0, lon0, xmin, xmax, ymin, ymax in (
        (52.3874, 4.6462, 103000, 106000, 488000, 490500),
        (53.246129, 6.603294, 235000, 237500, 584000, 586500),
    ):
        x, y = wgs84_to_rd(lat0, lon0)
        lat1, lon1 = rd_to_wgs84(x, y)
        assert xmin < x < xmax, (lat0, lon0, x, y)
        assert ymin < y < ymax, (lat0, lon0, x, y)
        assert haversine_m(lat0, lon0, lat1, lon1) < 0.05
