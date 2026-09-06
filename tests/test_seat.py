from __future__ import annotations

from shapely.geometry import Polygon

from ps1_hood.align.seat import (
    building_footprints,
    horizon_cardinals,
    is_inside_footprint,
    look_at_nearest_wall,
    pick_facade_shot,
    push_out_of_footprints,
    uncollapse_along_gps,
)
from ps1_hood.geo import heading_diff


def test_footprints_from_wall_vertices() -> None:
    # vertical wall corners of a 10×8 house, not filled faces
    verts = []
    faces = []
    for x, y in ((0, 0), (10, 0), (10, 8), (0, 8)):
        verts.append((x, y, 0.0))
        verts.append((x, y, 9.0))
    fps = building_footprints([{"vertices": verts, "faces": faces}])
    assert len(fps) == 1
    assert 70 < fps[0].area < 90
    assert is_inside_footprint(5.0, 4.0, fps, 0.0)
    assert not is_inside_footprint(15.0, 4.0, fps, 0.0)


def test_push_out_moves_interior_point_outside() -> None:
    square = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    e, n = push_out_of_footprints(5.0, 5.0, [square], margin_m=2.0)
    assert not is_inside_footprint(e, n, [square], margin_m=1.5)
    assert (e - 5.0) ** 2 + (n - 5.0) ** 2 > 4.0


def test_push_out_keeps_street_point() -> None:
    square = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    e, n = push_out_of_footprints(12.0, 5.0, [square], margin_m=1.0)
    assert abs(e - 12.0) < 1e-6
    assert abs(n - 5.0) < 1e-6


def test_look_at_wall_faces_building() -> None:
    square = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    heading, dist = look_at_nearest_wall(15.0, 5.0, [square])
    assert dist == 5.0
    # standing east of the block, looking west ≈ 270
    assert abs(heading_diff(heading, 270.0)) < 1.0


def test_pick_facade_shot_prefers_wall_heading() -> None:
    shots = [
        {"heading": 10.0, "pitch": 0.0, "path": "a.jpg"},
        {"heading": 265.0, "pitch": 0.0, "path": "b.jpg"},
        {"heading": 265.0, "pitch": 18.0, "path": "c.jpg"},
    ]
    hit = pick_facade_shot(shots, 270.0)
    assert hit is not None
    assert hit["path"] == "b.jpg"


def test_uncollapse_restores_gps_spacing() -> None:
    poses = [
        {"e_gps": 0.0, "n_gps": 0.0, "e": 0.0, "n": 0.0},
        {"e_gps": 10.0, "n_gps": 0.0, "e": 0.8, "n": 0.1},
    ]
    n = uncollapse_along_gps(poses)
    assert n >= 1
    dist = ((poses[0]["e"] - poses[1]["e"]) ** 2 + (poses[0]["n"] - poses[1]["n"]) ** 2) ** 0.5
    assert abs(dist - 10.0) < 0.05
    # still along the GPS axis, not thrown sideways
    assert abs(poses[0]["n"]) < 0.2
    assert abs(poses[1]["n"]) < 0.2


def test_horizon_cardinals_four_views() -> None:
    pose = {"pano_id": "x", "e": 0, "n": 0, "u": 2.5, "heading": 0, "travel_heading": 90, "fov": 90}
    shots = [
        {"heading": h, "pitch": p, "path": f"{h}_{p}.jpg", "fov": 90}
        for h in (0, 90, 180, 270)
        for p in (0, 18)
    ]
    cams = horizon_cardinals(pose, shots)
    assert len(cams) == 4
    headings = sorted(c["heading"] for c in cams)
    assert headings == [0, 90, 180, 270]
