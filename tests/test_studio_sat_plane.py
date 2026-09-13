"""Studio sat ground plane must come from Ortho ENU, not camera hull."""

from __future__ import annotations

from ps1_hood.capture.satellite import Ortho
from ps1_hood.geo import BBox, LocalFrame
from ps1_hood.reconstruct.export import satellite_with_enu, scene_payload


def test_ortho_enu_corners_match_init() -> None:
    bbox = BBox(52.0895, 5.1192, 52.09, 5.12)
    frame = LocalFrame.from_bbox(bbox)
    sw, sh, ee, nn = Ortho.enu_corners(bbox, frame)
    # Synthetic 2x2 image — only geometry matters
    import numpy as np

    ortho = Ortho(np.zeros((2, 2, 3), dtype=np.uint8), bbox, frame)
    assert abs(ortho.sw - sw) < 1e-9
    assert abs(ortho.sh - sh) < 1e-9
    assert abs(ortho.ee - ee) < 1e-9
    assert abs(ortho.nn - nn) < 1e-9
    # Smoke bbox centre ≈ ENU origin
    assert abs((sw + ee) / 2) < 0.05
    assert abs((sh + nn) / 2) < 0.05


def test_scene_payload_satellite_enu_from_bbox() -> None:
    bbox = BBox(52.0895, 5.1192, 52.09, 5.12)
    frame = LocalFrame.from_bbox(bbox)
    sat_meta = {
        "path": "/tmp/ortho.jpg",
        "width": 1400,
        "height": 1400,
        "bbox": bbox.as_dict(),
        "provider": "esri_world_imagery",
        "crs": "EPSG:4326",
    }
    # Cameras deliberately offset from bbox centre (old hull bug)
    poses = [
        {
            "pano_id": "a",
            "e": -12.5,
            "n": 8.0,
            "u": 2.5,
            "heading": 90.0,
            "pitch": 0.0,
            "fov": 90.0,
        },
        {
            "pano_id": "b",
            "e": 10.0,
            "n": -5.0,
            "u": 2.5,
            "heading": 180.0,
            "pitch": 0.0,
            "fov": 90.0,
        },
    ]
    scene = scene_payload(
        spec_name="smoke",
        bbox=bbox,
        frame=frame,
        poses=poses,
        frames=[],
        satellite=sat_meta,
        cloud=None,
        buildings=[],
    )
    enu = scene["satellite"]["enu"]
    sw, sh, ee, nn = Ortho.enu_corners(bbox, frame)
    assert abs(enu["sw"] - sw) < 1e-9
    assert abs(enu["sh"] - sh) < 1e-9
    assert abs(enu["ee"] - ee) < 1e-9
    assert abs(enu["nn"] - nn) < 1e-9
    assert abs(enu["centre_e"] - (sw + ee) / 2) < 1e-9
    assert abs(enu["centre_n"] - (sh + nn) / 2) < 1e-9
    assert abs(enu["width_m"] - (ee - sw)) < 1e-9
    assert abs(enu["height_m"] - (nn - sh)) < 1e-9
    # Must NOT match camera-hull centre (the old bug ~12 m on smoke)
    hull_e = (-35.5 + 10.5) / 2
    hull_n = (-16.8 + 32.9) / 2
    assert abs(enu["centre_e"] - hull_e) > 8.0
    assert abs(enu["centre_n"] - hull_n) > 5.0
    assert abs(enu["centre_e"]) < 0.1
    assert abs(enu["centre_n"]) < 0.1
    # origin matches LocalFrame.from_bbox
    assert abs(scene["origin"]["lat"] - frame.lat0) < 1e-12
    assert abs(scene["origin"]["lon"] - frame.lon0) < 1e-12


def test_satellite_with_enu_none() -> None:
    frame = LocalFrame(52.08975, 5.1196)
    assert satellite_with_enu(None, frame) is None
