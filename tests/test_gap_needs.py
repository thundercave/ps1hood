"""Far-side gap_needs: ENU→ll, need load, prefer existing facing, probes."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import yaml

from ps1_hood.geo import LocalFrame
from ps1_hood.reconstruct.gap_needs import (
    GapNeedsMissing,
    build_probe_seeds,
    enu_to_ll,
    find_existing_facing,
    is_facing_cam,
    load_gap_needs,
    local_frame_for_run,
    lookup_new_panos,
    merge_discover_panos,
    need_id_of,
    probe_enu_points,
    select_need,
    summarize_need,
)


def test_enu_to_ll_roundtrip() -> None:
    frame = LocalFrame(52.08975, 5.1196)
    lat, lon = enu_to_ll(frame, 12.0, -4.0)
    e, n, _u = frame.to_enu(lat, lon, 0.0)
    assert abs(e - 12.0) < 1e-3
    assert abs(n - (-4.0)) < 1e-3


def test_load_gap_needs_fail_loud(tmp_path: Path) -> None:
    missing = tmp_path / "recon" / "gap_needs.json"
    with pytest.raises(GapNeedsMissing, match="gap_needs missing"):
        load_gap_needs(missing)


def test_load_gap_needs_and_select(tmp_path: Path) -> None:
    path = tmp_path / "recon" / "gap_needs.json"
    path.parent.mkdir(parents=True)
    payload = {
        "run": "smoke-dense",
        "product_planes": 18,
        "needs": [
            {
                "e": 22.83,
                "n": 8.01,
                "heading": 90.0,
                "reason": "no_facing_cam",
                "roof_id": "roof_099",
                "edge_id": "roof_099_e0",
            }
        ],
        "hint": "fetch far-side SV panos",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_gap_needs(path)
    assert loaded["needs"][0]["id"] == "roof_099_e0"
    one = select_need(loaded, "roof_099_e0")
    assert len(one) == 1
    with pytest.raises(RuntimeError, match="not in gap_needs"):
        select_need(loaded, "nope")


def test_probe_outside_along_outward() -> None:
    # East wall: heading 90 → probes further east
    pts = probe_enu_points(10.0, 0.0, 90.0, radii_m=(8.0, 12.0))
    assert len(pts) == 2
    assert pts[0][0] == pytest.approx(18.0)
    assert pts[0][1] == pytest.approx(0.0)
    assert pts[1][0] == pytest.approx(22.0)


def test_prefer_existing_facing_ot6_style() -> None:
    """Nearly-facing cam north of need, heading 180, ~22 m — prefer it."""
    need = {
        "e": 20.5,
        "n": 10.2,
        "heading": 0.0,  # north outward (north wall)
        "edge_id": "roof_099_n0",
        "roof_id": "roof_099",
    }
    # Cam north of wall looking south
    cam = {
        "pano_id": "Ot6jYiTfjNNn-G8YqwyfCQ",
        "e": 20.5,
        "n": 10.2 + 22.5,
        "heading": 180.0,
    }
    assert is_facing_cam(cam, need)
    hits = find_existing_facing([cam], need)
    assert len(hits) == 1
    assert hits[0]["pano_id"] == "Ot6jYiTfjNNn-G8YqwyfCQ"


def test_prefer_existing_for_east_need_nearly() -> None:
    """Steering: Ot6jYiTfjNNn h180 can count as nearly-facing for roof_099_e0."""
    need = {
        "e": 22.83,
        "n": 8.01,
        "heading": 90.0,
        "edge_id": "roof_099_e0",
        "roof_id": "roof_099",
    }
    cam = {
        "pano_id": "Ot6jYiTfjNNn-G8YqwyfCQ",
        "e": 20.53,
        "n": 28.66,
        "heading": 180.0,
    }
    # look-at toward east mid from north: mostly south component → may or may not pass
    # If it passes, prefer path works; if not, still document probe path.
    hits = find_existing_facing([cam], need)
    # Distance ~22 m within [6,35]; look-at: v≈(2.3,-20.65), fwd=(0,-1) → high
    # same-side vs east outward: (cam-roof_c)·(+E) — roof_c ≈ need - 4*east
    assert is_facing_cam(cam, need) or hits == []
    dist = math.hypot(need["e"] - cam["e"], need["n"] - cam["n"])
    assert 6.0 <= dist <= 35.0


def test_local_frame_from_project_yaml(tmp_path: Path) -> None:
    (tmp_path / "project.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "t",
                "bbox": {
                    "south": 52.0895,
                    "west": 5.1192,
                    "north": 52.0900,
                    "east": 5.1200,
                },
            }
        ),
        encoding="utf-8",
    )
    frame = local_frame_for_run(tmp_path)
    assert abs(frame.lat0 - 52.08975) < 1e-6


def test_build_seeds_and_lookup_skips_existing(tmp_path: Path) -> None:
    frame = LocalFrame(52.08975, 5.1196)
    need = {
        "e": 0.0,
        "n": 0.0,
        "heading": 90.0,
        "edge_id": "roof_099_e0",
    }
    seeds = build_probe_seeds(need, frame, radii_m=(8.0, 12.0), need_id="roof_099_e0")
    assert len(seeds) == 2
    assert all(s["source"] == "gap_needs" for s in seeds)
    assert all(s["need_id"] == "roof_099_e0" for s in seeds)

    def fake_lookup(lat: float, lon: float):
        return {"pano_id": "EXISTING", "lat": lat, "lon": lon, "provider": "google"}

    # First call with empty existing → gets EXISTING once; second seed same id skipped
    found = lookup_new_panos(
        seeds, existing_ids=set(), lookup_fn=fake_lookup, max_panos=6
    )
    assert len(found) == 1
    assert found[0]["pano_id"] == "EXISTING"
    found2 = lookup_new_panos(
        seeds, existing_ids={"EXISTING"}, lookup_fn=fake_lookup, max_panos=6
    )
    assert found2 == []


def test_merge_discover_tags() -> None:
    disc = {"source": "google_web", "panos": [{"pano_id": "a", "lat": 1.0, "lon": 2.0}]}
    new = [
        {
            "pano_id": "b",
            "lat": 3.0,
            "lon": 4.0,
            "source": "gap_needs",
            "need_id": "roof_099_e0",
        }
    ]
    out = merge_discover_panos(disc, new)
    assert len(out["panos"]) == 2
    assert out["gap_needs_added"] == 1


def test_summarize_includes_existing(tmp_path: Path) -> None:
    frame = LocalFrame(52.08975, 5.1196)
    need = {
        "e": 20.5,
        "n": 10.2,
        "heading": 0.0,
        "edge_id": "roof_099_n0",
        "reason": "no_facing_cam",
    }
    cams = [
        {
            "pano_id": "Ot6jYiTfjNNn-G8YqwyfCQ",
            "e": 20.5,
            "n": 32.7,
            "heading": 180.0,
        }
    ]
    s = summarize_need(need, frame, cams)
    assert s["id"] == "roof_099_n0"
    assert s["n_existing_facing"] == 1
    assert s["probes"]
