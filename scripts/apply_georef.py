#!/usr/bin/env python3
"""Apply align/georef.json SE(2) to recon artefacts + refresh scene cameras.

PC recipe after sat align (no densify re-run):

  uv run ps1hood align smoke-dense --align-prior sat
  # stage_align already seats + clips when prior poses exist; or:
  uv run python scripts/apply_georef.py smoke-dense --clip-sat-bbox
  # Studio hard-reload; BAG off; cams+façades on sat streets; MA inside Ortho
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# allow running from repo root without install
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ps1_hood.align.georef import (  # noqa: E402
    clip_recon_clouds_to_ortho,
    refresh_scene_cameras,
    seat_recon_artefacts,
)
from ps1_hood.capture.satellite import Ortho  # noqa: E402
from ps1_hood.geo import LocalFrame  # noqa: E402
from ps1_hood.project import open_project  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("name", help="run name under runs/")
    ap.add_argument(
        "--which",
        choices=("T_applied", "T_sat"),
        default="T_applied",
        help="Which transform in georef.json to apply (default T_applied)",
    )
    ap.add_argument(
        "--clip-sat-bbox",
        action="store_true",
        help="After SE(2), clip cloud.ply to Ortho ENU ± --sat-cloud-margin-m",
    )
    ap.add_argument(
        "--sat-cloud-margin-m",
        type=float,
        default=2.0,
        help="ENU margin (m) for --clip-sat-bbox (default 2)",
    )
    args = ap.parse_args()
    project = open_project(args.name)
    georef_path = project.align_dir / "georef.json"
    if not georef_path.is_file():
        print(f"missing {georef_path}; run: ps1hood align {args.name} --align-prior sat", file=sys.stderr)
        return 1
    georef = project.read_json(georef_path)
    key = args.which
    T = georef.get(key) or georef.get("T_sat")
    if not T:
        print(f"georef has no {key}/T_sat", file=sys.stderr)
        return 1
    stats = seat_recon_artefacts(project.recon_dir, T)
    if args.clip_sat_bbox:
        sat = project.read_json(project.satellite_dir / "ortho.json")
        spec = project.load_spec()
        frame = LocalFrame.from_bbox(spec.bbox)
        ortho = Ortho.load(sat, frame)
        clip = clip_recon_clouds_to_ortho(
            project.recon_dir,
            ortho.sw,
            ortho.sh,
            ortho.ee,
            ortho.nn,
            margin_m=args.sat_cloud_margin_m,
        )
        georef["cloud_clipped"] = clip
        project.write_json(georef_path, georef)
        print(f"clipped: kept={clip.get('kept')} dropped={clip.get('dropped')}")
    poses = project.read_json(project.align_dir / "poses.json")
    refresh_scene_cameras(project.recon_dir / "scene.json", poses, georef)
    print(f"applied {key}: {stats}")
    print(f"scene cameras refreshed; prior={georef.get('prior')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
