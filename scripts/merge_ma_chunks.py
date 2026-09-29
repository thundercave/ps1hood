#!/usr/bin/env python3
"""CLI: diagnose / keep-one / hybrid merge for MapAnything chunk PLYs.

Locked ENU only — no ICP, no pose invent, no XYZ averaging.

Examples (overnight recipe)::

  python scripts/merge_ma_chunks.py diagnose \\
      runs/.../chunks/chunk_*.ply --max-points 50000

  python scripts/merge_ma_chunks.py keep-one --radius 0.12 \\
      runs/.../chunks/chunk_*.ply --out runs/.../cloud_dedup012.ply

  python scripts/merge_ma_chunks.py hybrid \\
      --base runs/.../cloud.ply.bak_pre_chunks \\
      --nn-m 0.10 --chunk-nn chunk_04=0.15 \\
      runs/.../chunks/chunk_*.ply --out runs/.../cloud_hybrid.ply
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from ps1_hood.reconstruct.merge_ma_chunks import (
    diagnose_paths,
    format_diagnose_line,
    hybrid_merge_paths,
    keep_one_paths,
    parse_chunk_nn,
    resolve_chunk_paths,
    write_xyz_ply,
)


def _add_shared(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "chunks",
        nargs="+",
        help="chunk PLY paths and/or globs (order preserved after expansion)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output PLY (required for keep-one / hybrid)",
    )
    p.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="subsample overlap queries for diagnose speed",
    )
    p.add_argument("--seed", type=int, default=0, help="RNG seed (default 0)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("diagnose", help="overlap NN(A→B) median/p90 + verdict")
    _add_shared(d)

    k = sub.add_parser("keep-one", help="greedy radius keep-one (no XYZ average)")
    _add_shared(k)
    k.add_argument(
        "--radius",
        type=float,
        required=True,
        help="keep-one ball radius in metres (overnight: 0.12)",
    )

    h = sub.add_parser(
        "hybrid",
        help="append chunk points farther than --nn-m from current merge",
    )
    _add_shared(h)
    h.add_argument(
        "--base",
        type=Path,
        required=True,
        help="base cloud PLY (overnight: bak_pre_chunks)",
    )
    h.add_argument(
        "--nn-m",
        type=float,
        default=0.10,
        help="default NN threshold metres (default 0.10)",
    )
    h.add_argument(
        "--chunk-nn",
        action="append",
        default=[],
        metavar="CHUNK=NN",
        help="per-chunk NN override (repeatable); overnight: chunk_04=0.15",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = resolve_chunk_paths(args.chunks)

    if args.cmd == "diagnose":
        rows = diagnose_paths(paths, max_points=args.max_points, seed=args.seed)
        for r in rows:
            print(format_diagnose_line(r))
        return 0

    if args.out is None:
        print("error: --out is required for keep-one / hybrid", file=sys.stderr)
        return 2

    if args.cmd == "keep-one":
        xyz = keep_one_paths(paths, args.radius, seed=args.seed)
        write_xyz_ply(args.out, xyz)
        print(f"keep-one radius={args.radius}: wrote {len(xyz)} pts → {args.out}")
        return 0

    if args.cmd == "hybrid":
        overrides = parse_chunk_nn(args.chunk_nn)
        xyz = hybrid_merge_paths(
            args.base,
            paths,
            nn_m=args.nn_m,
            chunk_nn_overrides=overrides,
        )
        write_xyz_ply(args.out, xyz)
        print(
            f"hybrid nn_m={args.nn_m} overrides={overrides}: "
            f"wrote {len(xyz)} pts → {args.out}"
        )
        return 0

    print(f"unknown cmd {args.cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
