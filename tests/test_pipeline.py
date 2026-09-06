from __future__ import annotations

from ps1_hood.geo import BBox
from ps1_hood.pipeline import _bag_query_bbox, _shots_in_bbox


def test_shots_in_bbox_drops_far_panos() -> None:
    box = BBox(53.24587, 6.60282, 53.24639, 6.60377)
    shots = [
        {"pano_id": "in", "lat": 53.2461, "lon": 6.6033},
        {"pano_id": "in", "lat": 53.2461, "lon": 6.6033},
        {"pano_id": "far", "lat": 53.24691, "lon": 6.60516},
    ]
    kept, dropped = _shots_in_bbox(shots, box)
    assert dropped == 1
    assert {s["pano_id"] for s in kept} == {"in"}
    assert len(kept) == 2


def test_bag_query_bbox_includes_inblock_cameras() -> None:
    box = BBox(53.24587, 6.60282, 53.24639, 6.60377)
    shots = [
        {"pano_id": "in", "lat": 53.2461, "lon": 6.6039},
        {"pano_id": "far", "lat": 53.25, "lon": 6.62},
    ]
    q = _bag_query_bbox(box, shots)
    assert q.contains(53.2461, 6.6039)
    assert not q.contains(53.25, 6.62)
