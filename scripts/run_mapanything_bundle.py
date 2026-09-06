#!/usr/bin/env python3
"""Run MapAnything on a ps1hood pose-locked bundle (cloud CUDA).

Never passes ignore_pose_inputs=True. Prefer --apache weights.

Example:
  python scripts/run_mapanything_bundle.py runs/smoke-dense/mapanything/bundle --apache
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from a checkout without install.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bundle", type=Path, help="path to mapanything/bundle (manifest.json)")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output PLY (default: <bundle>/../cloud.ply)",
    )
    p.add_argument(
        "--apache",
        action="store_true",
        default=True,
        help="use facebook/map-anything-apache (default)",
    )
    p.add_argument(
        "--research",
        action="store_true",
        help="use facebook/map-anything (CC-BY-NC) instead of apache",
    )
    p.add_argument("--minibatch-size", type=int, default=1)
    p.add_argument(
        "--import-recon",
        type=Path,
        default=None,
        help="if set, copy PLY to <run>/recon/cloud_mapanything.ply (+ cloud.ply)",
    )
    args = p.parse_args()
    apache = not args.research

    from ps1_hood.reconstruct import mapanything as ma

    bundle = args.bundle.resolve()
    if not (bundle / "manifest.json").is_file():
        print(f"missing manifest.json under {bundle}", file=sys.stderr)
        return 1
    out = args.out or (bundle.parent / "cloud.ply")
    try:
        meta = ma.run_mapanything_on_bundle(
            bundle,
            out_ply=out,
            apache=apache,
            minibatch_size=args.minibatch_size,
        )
    except Exception as exc:
        print(f"MapAnything failed: {exc}", file=sys.stderr)
        print(ma.INSTALL_HINT, file=sys.stderr)
        return 1

    if args.import_recon is not None:
        imported = ma.import_dense_ply_to_recon(Path(meta["path"]), args.import_recon)
        meta["import"] = imported

    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
