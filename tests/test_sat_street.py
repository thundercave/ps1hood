"""Sat-locked street/ground shells — support punch + ground_z + no clobber."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from ps1_hood.capture.satellite import Ortho
from ps1_hood.geo import BBox, LocalFrame
from ps1_hood.reconstruct.sat_roofs import segment_roof_yard_mask
from ps1_hood.reconstruct.sat_street import (
    SatStreetError,
    build_sat_street,
    build_street_support_mask,
    extract_street_regions,
    resolve_ground_z,
    write_street_obj,
)


def _synthetic_ortho(
    *,
    w: int = 200,
    h: int = 200,
    with_building: bool = True,
) -> Ortho:
    """Gray street + optional brown roof + green yard (same recipe as roofs)."""
    bbox = BBox(52.0895, 5.1192, 52.09, 5.12)
    frame = LocalFrame.from_bbox(bbox)
    img = np.full((h, w, 3), 90, dtype=np.uint8)
    if with_building:
        img[40:100, 40:120] = (40, 70, 120)
        img[110:150, 50:110] = (40, 140, 50)
        img[38:102, 38:40] = 20
        img[38:102, 120:122] = 20
        img[38:40, 38:122] = 20
        img[100:102, 38:122] = 20
    return Ortho(img, bbox, frame)


def _fake_run(tmp_path: Path, ortho: Ortho, *, with_sacred: bool = True) -> Path:
    run = tmp_path / "run"
    sat = run / "satellite"
    sat.mkdir(parents=True)
    recon = run / "recon"
    recon.mkdir()
    img_path = sat / "ortho.jpg"
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
    # Cameras along south border (street) at u=2.5 → ground_z=0
    cams = [
        {"e": float(ortho.sw + 5 + i * 3), "n": float(ortho.sh + 2), "u": 2.5}
        for i in range(5)
    ]
    (recon / "scene.json").write_text(
        '{"name": "run", "cameras": %s}'
        % (
            "["
            + ",".join(
                '{"e": %s, "n": %s, "u": %s}' % (c["e"], c["n"], c["u"]) for c in cams
            )
            + "]"
        ),
        encoding="utf-8",
    )
    if with_sacred:
        (recon / "facades.obj").write_text("# sacred facades\nv 0 0 0\n", encoding="ascii")
        (recon / "facades.mtl").write_text("# sacred\n", encoding="ascii")
        (recon / "roofs.obj").write_text("# sacred roofs\nv 0 0 8\n", encoding="ascii")
        (recon / "roofs.mtl").write_text("# sacred\n", encoding="ascii")
        (recon / "planes.json").write_text(
            '{"ground_z": 1.2, "planes": []}', encoding="utf-8"
        )
        # Roof AABB covering the building block for ∩ reject tests
        e0, n0 = ortho.sw, ortho.sh
        # Approximate centre building in ENU via pixel corners
        from ps1_hood.reconstruct.sat_roofs import px_to_enu

        corners = [
            px_to_enu(ortho, 40, 100),
            px_to_enu(ortho, 120, 100),
            px_to_enu(ortho, 120, 40),
            px_to_enu(ortho, 40, 40),
        ]
        import json

        (recon / "roofs.json").write_text(
            json.dumps(
                {
                    "shells": [
                        {
                            "id": "roof_000",
                            "kind": "roof",
                            "z": 9.0,
                            "aabb_enu": corners,
                            "area_m2": 50.0,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
    return run


def test_support_punches_building() -> None:
    ortho = _synthetic_ortho()
    support, info = build_street_support_mask(ortho, building_dilate_m=1.5)
    assert info["support_px"] > 50
    roof_m, yard_m, street_m = segment_roof_yard_mask(ortho.image)
    # Centre of roof block should not be support
    assert support[70, 80] == 0
    # Border street flood should remain
    assert int((support > 0).sum()) < int((street_m > 0).sum()) + 5000
    assert int((support > 0).sum()) > 100


def test_ground_z_prefers_planes_then_cam_u(tmp_path: Path) -> None:
    run = _fake_run(tmp_path, _synthetic_ortho(), with_sacred=True)
    gz, src = resolve_ground_z(run, cam_u=2.5, camera_height_m=2.5)
    assert abs(gz - 1.2) < 1e-9
    assert src == "planes_ground_z"
    # No planes / roofs → cam_u − height
    run2 = _fake_run(tmp_path / "b", _synthetic_ortho(), with_sacred=False)
    gz2, src2 = resolve_ground_z(run2, cam_u=2.5, camera_height_m=2.5)
    assert abs(gz2 - 0.0) < 1e-9
    assert src2 == "cam_u_minus_height"


def test_extract_street_tiles_and_write(tmp_path: Path) -> None:
    ortho = _synthetic_ortho()
    support, _ = build_street_support_mask(ortho, building_dilate_m=1.0)
    regions = extract_street_regions(ortho, support, min_area_m2=2.0)
    assert len(regions) >= 1
    assert all(r["kind"] == "street" for r in regions)
    for r in regions:
        r["z"] = 0.15
    dest = tmp_path / "recon" / "street.obj"
    meta = write_street_obj(dest, regions, ortho)
    assert dest.is_file()
    assert dest.with_suffix(".mtl").is_file()
    body = dest.read_text(encoding="ascii")
    assert "mtllib" in body
    assert "street_" in body
    assert meta["shells"] == len(regions)
    assert meta["textured"] >= 1


def test_build_does_not_clobber_sacred(tmp_path: Path) -> None:
    ortho = _synthetic_ortho()
    run = _fake_run(tmp_path, ortho, with_sacred=True)
    recon = run / "recon"
    facades_before = (recon / "facades.obj").read_text(encoding="ascii")
    roofs_before = (recon / "roofs.obj").read_text(encoding="ascii")
    planes_before = (recon / "planes.json").read_text(encoding="utf-8")
    meta = build_sat_street(run, min_area_m2=2.0, building_dilate_m=1.0)
    assert meta["n_street"] >= 1
    assert (recon / "street.obj").is_file()
    assert (recon / "street.json").is_file()
    assert (recon / "facades.obj").read_text(encoding="ascii") == facades_before
    assert (recon / "roofs.obj").read_text(encoding="ascii") == roofs_before
    assert (recon / "planes.json").read_text(encoding="utf-8") == planes_before
    # Z at product ground (planes), not roof / cam slab
    assert abs(float(meta["ground_z"]) - 1.2) < 0.05
    assert meta["z_source"] == "planes_ground_z"


def test_street_aabb_not_under_roof(tmp_path: Path) -> None:
    ortho = _synthetic_ortho()
    run = _fake_run(tmp_path, ortho, with_sacred=True)
    meta = build_sat_street(run, min_area_m2=2.0, building_dilate_m=1.5)
    import json

    shells = json.loads((run / "recon" / "street.json").read_text(encoding="utf-8"))[
        "shells"
    ]
    roof = json.loads((run / "recon" / "roofs.json").read_text(encoding="utf-8"))[
        "shells"
    ][0]["aabb_enu"]
    res = [p[0] for p in roof]
    rns = [p[1] for p in roof]
    re0, re1 = min(res), max(res)
    rn0, rn1 = min(rns), max(rns)
    for s in shells:
        es = [p[0] for p in s["aabb_enu"]]
        ns = [p[1] for p in s["aabb_enu"]]
        e0, e1 = min(es), max(es)
        n0, n1 = min(ns), max(ns)
        # No overlap with roof AABB
        assert not (e0 < re1 and e1 > re0 and n0 < rn1 and n1 > rn0)


def test_empty_support_fail_loud(tmp_path: Path) -> None:
    # Tiny solid building filling almost everything after dilate → little street;
    # force fail by requiring huge min area on a uniform-ish result.
    ortho = _synthetic_ortho(with_building=False)
    run = _fake_run(tmp_path, ortho, with_sacred=False)
    # Uniform gray: street flood = whole tile → rejected as near-full-tile
    with pytest.raises(SatStreetError):
        build_sat_street(run, min_area_m2=8.0, cam_corridor_m=0.0)
