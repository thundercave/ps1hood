"""Sat-locked roof/yard shells (PR-A) — ENU corners + fail-loud."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ps1_hood.capture.satellite import Ortho
from ps1_hood.geo import BBox, LocalFrame
from ps1_hood.reconstruct.sat_roofs import (
    SatRoofError,
    assign_shell_z,
    build_sat_roofs,
    enu_corners_quad,
    extract_roof_yard_polygons,
    load_ortho_for_run,
    px_to_enu,
    segment_roof_yard_mask,
    write_roofs_obj,
)


def _synthetic_ortho(
    *,
    w: int = 200,
    h: int = 200,
    with_building: bool = True,
) -> Ortho:
    """Gray street + optional brown roof rectangle + green yard."""
    bbox = BBox(52.0895, 5.1192, 52.09, 5.12)
    frame = LocalFrame.from_bbox(bbox)
    img = np.full((h, w, 3), 90, dtype=np.uint8)  # asphalt-ish BGR
    if with_building:
        # Roof block (brown) in centre — leave border as street for flood
        img[40:100, 40:120] = (40, 70, 120)
        # Yard (green)
        img[110:150, 50:110] = (40, 140, 50)
        # Dark outline to help Canny seal
        img[38:102, 38:40] = 20
        img[38:102, 120:122] = 20
        img[38:40, 38:122] = 20
        img[100:102, 38:122] = 20
    return Ortho(img, bbox, frame)


def test_enu_corners_quad_matches_ortho() -> None:
    ortho = _synthetic_ortho()
    corners = enu_corners_quad(ortho)
    assert len(corners) == 4
    assert abs(corners[0][0] - ortho.sw) < 1e-9
    assert abs(corners[0][1] - ortho.sh) < 1e-9
    assert abs(corners[2][0] - ortho.ee) < 1e-9
    assert abs(corners[2][1] - ortho.nn) < 1e-9
    # Round-trip pixel centre ↔ ENU
    e, n = px_to_enu(ortho, (ortho.w - 1) / 2, (ortho.h - 1) / 2)
    u, v = ortho.enu_to_px(e, n)
    assert abs(u - (ortho.w - 1) / 2) < 1e-6
    assert abs(v - (ortho.h - 1) / 2) < 1e-6


def test_px_to_enu_corners() -> None:
    ortho = _synthetic_ortho()
    e0, n0 = px_to_enu(ortho, 0, ortho.h - 1)
    assert abs(e0 - ortho.sw) < 1e-6
    assert abs(n0 - ortho.sh) < 1e-6
    e1, n1 = px_to_enu(ortho, ortho.w - 1, 0)
    assert abs(e1 - ortho.ee) < 1e-6
    assert abs(n1 - ortho.nn) < 1e-6


def test_extract_finds_roof_and_yard() -> None:
    ortho = _synthetic_ortho()
    regions = extract_roof_yard_polygons(ortho, min_area_m2=2.0)
    kinds = {r["kind"] for r in regions}
    assert "roof" in kinds or "yard" in kinds
    assert all("aabb_enu" in r and len(r["aabb_enu"]) == 4 for r in regions)
    # AABB corners sit inside Ortho ENU bounds (±1 m tolerance)
    for r in regions:
        for e, n in r["aabb_enu"]:
            assert ortho.sw - 1.0 <= e <= ortho.ee + 1.0
            assert ortho.sh - 1.0 <= n <= ortho.nn + 1.0


def test_empty_sat_fail_loud(tmp_path: Path) -> None:
    sat = tmp_path / "satellite"
    sat.mkdir()
    # Missing ortho.json
    with pytest.raises(SatRoofError, match="ortho.json"):
        load_ortho_for_run(tmp_path)

    # ortho.json pointing at missing image
    (sat / "ortho.json").write_text(
        '{"path": "/no/such/ortho.jpg", "bbox": '
        '{"south": 52.0895, "west": 5.1192, "north": 52.09, "east": 5.12}, '
        '"width": 10, "height": 10}',
        encoding="utf-8",
    )
    with pytest.raises(SatRoofError, match="empty/missing sat"):
        load_ortho_for_run(tmp_path)


def test_zero_footprints_fail_loud() -> None:
    # Uniform gray — flood eats everything → no interiors
    ortho = _synthetic_ortho(with_building=False)
    with pytest.raises(SatRoofError, match="no roof/yard footprints"):
        extract_roof_yard_polygons(ortho, min_area_m2=8.0)


def test_assign_z_ma_median_and_facade_fallback(tmp_path: Path) -> None:
    ortho = _synthetic_ortho()
    regions = extract_roof_yard_polygons(ortho, min_area_m2=2.0)
    # Synthetic cloud at z=9 over roof AABB
    roof = next(r for r in regions if r["kind"] == "roof")
    es = [p[0] for p in roof["polygon_enu"]]
    ns = [p[1] for p in roof["polygon_enu"]]
    e0, e1 = min(es), max(es)
    n0, n1 = min(ns), max(ns)
    rng = np.random.default_rng(0)
    pts = np.column_stack(
        [
            rng.uniform(e0, e1, 40),
            rng.uniform(n0, n1, 40),
            np.full(40, 9.5),
        ]
    )
    out = assign_shell_z(regions, pts, planes_json=None)
    roof_out = next(r for r in out if r["id"] == roof["id"])
    assert roof_out["z_source"] == "ma_median"
    assert abs(roof_out["z"] - 9.5) < 0.2

    # No cloud → façade top / default
    planes = tmp_path / "planes.json"
    planes.write_text(
        '{"ground_z": 1.0, "planes": [{"quad": [[0,0,1],[1,0,1],[1,0,10],[0,0,10]], '
        '"height_m": 9.0}]}',
        encoding="utf-8",
    )
    out2 = assign_shell_z(regions, None, planes_json=planes)
    roof2 = next(r for r in out2 if r["kind"] == "roof")
    assert roof2["z_source"] == "facade_top"
    assert abs(roof2["z"] - 10.0) < 0.1


def test_write_roofs_obj_and_build(tmp_path: Path) -> None:
    ortho = _synthetic_ortho()
    regions = extract_roof_yard_polygons(ortho, min_area_m2=2.0)
    regions = assign_shell_z(regions, None, planes_json=None)
    dest = tmp_path / "recon" / "roofs.obj"
    meta = write_roofs_obj(dest, regions, ortho)
    assert dest.is_file()
    assert dest.with_suffix(".mtl").is_file()
    body = dest.read_text(encoding="ascii")
    assert "mtllib" in body
    assert body.count("\nv ") >= 4
    assert meta["shells"] == len(regions)

    # Full build_sat_roofs against a fake run layout
    run = tmp_path / "run"
    sat = run / "satellite"
    sat.mkdir(parents=True)
    recon = run / "recon"
    recon.mkdir()
    img_path = sat / "ortho.jpg"
    import cv2

    cv2.imwrite(str(img_path), ortho.image)
    (sat / "ortho.json").write_text(
        '{"path": "%s", "width": %d, "height": %d, "bbox": %s, '
        '"provider": "test", "crs": "EPSG:4326"}'
        % (
            img_path.as_posix(),
            ortho.w,
            ortho.h,
            '{"south": 52.0895, "west": 5.1192, "north": 52.09, "east": 5.12}',
        ),
        encoding="utf-8",
    )
    meta2 = build_sat_roofs(run, min_area_m2=2.0)
    assert meta2["shells"] >= 1
    assert (recon / "roofs.obj").is_file()
    assert (recon / "roofs.json").is_file()


def test_segment_masks_nonempty_on_building() -> None:
    ortho = _synthetic_ortho()
    roof_m, yard_m = segment_roof_yard_mask(ortho.image)
    assert roof_m.shape[:2] == ortho.image.shape[:2]
    # At least one of roof/yard should light up
    assert int((roof_m > 0).sum()) + int((yard_m > 0).sum()) > 50
