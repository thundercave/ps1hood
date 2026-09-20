"""Sat-absolute SE(2) georef helpers — one transform for cams + recon artefacts.

Sacred: Ortho = absolute XY; photo relative 3D preserved under a single SE(2).
BAG is not the hero; snap_camera_to_bag only under --align-prior bag.
"""

from __future__ import annotations

import json
import logging
import math

import numpy as np
import struct
from pathlib import Path
from typing import Any


from ps1_hood.geo import wrap_heading

log = logging.getLogger(__name__)

ALIGN_PRIORS = ("sat", "bag")


def summarize_se2(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
) -> dict[str, float]:
    """PR1 summary SE(2): centroid translation + mean yaw delta (degrees).

    Maps ``before`` → ``after`` in LocalFrame ENU. Scale fixed at 1.0.
    """
    if not before or not after or len(before) != len(after):
        return {"tx_m": 0.0, "ty_m": 0.0, "yaw_deg": 0.0, "s": 1.0}
    be = sum(float(p["e"]) for p in before) / len(before)
    bn = sum(float(p["n"]) for p in before) / len(before)
    ae = sum(float(p["e"]) for p in after) / len(after)
    an = sum(float(p["n"]) for p in after) / len(after)
    # Mean heading delta (circular)
    sins = 0.0
    coss = 0.0
    for b, a in zip(before, after):
        d = math.radians(float(a.get("heading", 0.0)) - float(b.get("heading", 0.0)))
        sins += math.sin(d)
        coss += math.cos(d)
    yaw = math.degrees(math.atan2(sins, coss)) if (sins or coss) else 0.0
    return {
        "tx_m": float(ae - be),
        "ty_m": float(an - bn),
        "yaw_deg": float(yaw),
        "s": 1.0,
        # Rotate about before-centroid, then translate by (tx, ty):
        # p' = R(p - pivot) + pivot + (tx, ty) == R(p - C_b) + C_a
        "pivot_e": float(be),
        "pivot_n": float(bn),
    }



def fit_se2(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
) -> dict[str, float]:
    """Least-squares SE(2) mapping before cam XY → after (Procrustes, s=1).

    Prefer this over ``summarize_se2`` when seating cloud/façades so one rigid
    motion matches the pose set. Falls back to centroid summary if <2 points.
    """
    if len(before) < 2 or len(before) != len(after):
        return summarize_se2(before, after)
    B = np.array([[float(p["e"]), float(p["n"])] for p in before], dtype=np.float64)
    A = np.array([[float(p["e"]), float(p["n"])] for p in after], dtype=np.float64)
    cb = B.mean(axis=0)
    ca = A.mean(axis=0)
    Bb = B - cb
    Aa = A - ca
    H = Bb.T @ Aa
    U, _S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    yaw = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    # Our apply_se2_xy uses: re = c*de + sn*dn; rn = -sn*de + c*dn
    # which is R = [[c, sn], [-sn, c]] — note arctan2(R10, R00) with that form
    # R_apply = [[c, sn], [-sn, c]] => R[0,0]=c, R[0,1]=sn, R[1,0]=-sn, R[1,1]=c
    # From SVD we got standard [[c,-s],[s,c]] or [[c,s],[-s,c]] depending on convention.
    # Re-derive yaw from the apply convention directly:
    c = float(R[0, 0])
    # Prefer matching apply_se2_xy: want R_app @ Bb ≈ Aa
    # Try yaw from atan2 of off-diagonal consistent with apply
    yaw = float(np.degrees(np.arctan2(R[0, 1], R[0, 0])))  # sn, c if R is [[c,sn],[-sn,c]]
    # Verify / correct via mean heading delta as soft check — trust SVD of apply form:
    # Rebuild R in apply convention from yaw
    rad = np.radians(yaw)
    c, sn = float(np.cos(rad)), float(np.sin(rad))
    R_app = np.array([[c, sn], [-sn, c]], dtype=np.float64)
    # If SVD R was the other convention, flip yaw sign
    err_pos = np.linalg.norm((R_app @ Bb.T).T - Aa)
    err_neg = np.linalg.norm((np.array([[c, -sn], [sn, c]]) @ Bb.T).T - Aa)
    if err_neg < err_pos:
        yaw = -yaw
        rad = np.radians(yaw)
        c, sn = float(np.cos(rad)), float(np.sin(rad))
        R_app = np.array([[c, sn], [-sn, c]], dtype=np.float64)
    tx = float(ca[0] - cb[0])  # with pivot=cb: p'=R(p-cb)+cb+(tx,ty) => ca = cb + (tx,ty) when mean(R Bb)=0
    ty = float(ca[1] - cb[1])
    return {
        "tx_m": tx,
        "ty_m": ty,
        "yaw_deg": yaw,
        "s": 1.0,
        "pivot_e": float(cb[0]),
        "pivot_n": float(cb[1]),
    }


def apply_se2_xy(
    e: float,
    n: float,
    T: dict[str, float],
    *,
    pivot_e: float | None = None,
    pivot_n: float | None = None,
) -> tuple[float, float]:
    """Apply SE(2): ``p' = R(p - pivot) + pivot + (tx, ty)``.

    Pivot defaults to ``T[pivot_e/n]`` (before-centroid from ``summarize_se2``),
    else LocalFrame origin. Leaves vertical (u) untouched at call sites.
    """
    pe = float(T["pivot_e"]) if pivot_e is None and "pivot_e" in T else float(0.0 if pivot_e is None else pivot_e)
    pn = float(T["pivot_n"]) if pivot_n is None and "pivot_n" in T else float(0.0 if pivot_n is None else pivot_n)
    s = float(T.get("s", 1.0))
    yaw = math.radians(float(T.get("yaw_deg", 0.0)))
    c, sn = math.cos(yaw), math.sin(yaw)
    de, dn = e - pe, n - pn
    re = c * de + sn * dn
    rn = -sn * de + c * dn
    return pe + s * re + float(T.get("tx_m", 0.0)), pn + s * rn + float(T.get("ty_m", 0.0))


def apply_se2_pose(
    pose: dict[str, Any],
    T: dict[str, float],
    *,
    pivot_e: float | None = None,
    pivot_n: float | None = None,
) -> dict[str, Any]:
    out = dict(pose)
    e2, n2 = apply_se2_xy(float(pose["e"]), float(pose["n"]), T, pivot_e=pivot_e, pivot_n=pivot_n)
    out["e"] = e2
    out["n"] = n2
    if "heading" in out:
        out["heading"] = wrap_heading(float(out["heading"]) + float(T.get("yaw_deg", 0.0)))
    if "travel_heading" in out:
        out["travel_heading"] = wrap_heading(
            float(out["travel_heading"]) + float(T.get("yaw_deg", 0.0))
        )
    return out


def georef_payload(
    *,
    prior: str,
    T_sat: dict[str, float],
    sat_score_mean: float | None = None,
    bag_snapped: int = 0,
    n_poses: int = 0,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "prior": prior,
        "frame": "ENU+LocalFrame",
        "T_sat": {
            "tx_m": float(T_sat.get("tx_m", 0.0)),
            "ty_m": float(T_sat.get("ty_m", 0.0)),
            "yaw_deg": float(T_sat.get("yaw_deg", 0.0)),
            "s": float(T_sat.get("s", 1.0)),
        },
        "overlay_registered": prior == "sat",
        "bag_snapped": int(bag_snapped),
        "n_poses": int(n_poses),
    }
    if sat_score_mean is not None:
        out["sat_score_mean"] = float(sat_score_mean)
    if extra:
        out.update({k: v for k, v in extra.items() if v is not None})
    return out


def _centroid_en(poses: list[dict[str, Any]]) -> tuple[float, float]:
    if not poses:
        return 0.0, 0.0
    return (
        sum(float(p["e"]) for p in poses) / len(poses),
        sum(float(p["n"]) for p in poses) / len(poses),
    )


def apply_se2_to_ascii_obj(path: Path, T: dict[str, float], *, pivot_e: float | None = None, pivot_n: float | None = None) -> int:
    """Transform ``v x y z`` lines — XY only, Z untouched. Returns #verts changed."""
    if not path.is_file():
        return 0
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    out: list[str] = []
    n = 0
    for line in lines:
        if line.startswith("v "):
            parts = line.split()
            if len(parts) >= 4:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                x2, y2 = apply_se2_xy(x, y, T, pivot_e=pivot_e, pivot_n=pivot_n)
                rest = " ".join(parts[4:])
                out.append(f"v {x2:.6f} {y2:.6f} {z:.6f}" + (f" {rest}" if rest else "") + "\n")
                n += 1
                continue
        out.append(line if line.endswith("\n") else line + "\n")
    path.write_text("".join(out), encoding="utf-8")
    return n


def apply_se2_to_ply(path: Path, T: dict[str, float], *, pivot_e: float | None = None, pivot_n: float | None = None) -> int:
    """In-place XY SE(2) on ascii or binary_little_endian PLY (x/y/z float first)."""
    if not path.is_file():
        return 0
    raw = path.read_bytes()
    header_end = raw.find(b"end_header")
    if header_end < 0:
        raise RuntimeError(f"not a PLY: {path}")
    header = raw[:header_end].decode("ascii", errors="replace")
    body = raw[header_end + len(b"end_header") :]
    nl = b"\n"
    if body.startswith(b"\r\n"):
        body = body[2:]
        nl = b"\r\n"
    elif body.startswith(b"\n"):
        body = body[1:]

    fmt = "ascii"
    n_verts = 0
    props: list[tuple[str, str]] = []
    in_vertex = False
    for line in header.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element" and parts[1] == "vertex":
            n_verts = int(parts[2])
            in_vertex = True
        elif parts[0] == "element":
            in_vertex = False
        elif in_vertex and parts[0] == "property":
            props.append((parts[1], parts[2]))

    if n_verts <= 0:
        return 0

    if fmt == "ascii":
        text = body.decode("ascii", errors="replace")
        lines = text.splitlines(keepends=True)
        out_lines: list[str] = []
        changed = 0
        for i, line in enumerate(lines):
            if changed >= n_verts:
                out_lines.append(line if line.endswith("\n") else line + "\n")
                continue
            parts = line.split()
            if len(parts) < 3:
                out_lines.append(line if line.endswith("\n") else line + "\n")
                continue
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            x2, y2 = apply_se2_xy(x, y, T, pivot_e=pivot_e, pivot_n=pivot_n)
            rest = " ".join(parts[3:])
            out_lines.append(
                f"{x2:.6f} {y2:.6f} {z:.6f}" + (f" {rest}" if rest else "") + "\n"
            )
            changed += 1
        path.write_bytes(raw[: header_end + len(b"end_header")] + nl + "".join(out_lines).encode("ascii"))
        return changed

    # binary — assume x,y,z are first three float32/float64 props
    type_map = {
        "char": "b",
        "uchar": "B",
        "int8": "b",
        "uint8": "B",
        "short": "h",
        "ushort": "H",
        "int16": "h",
        "uint16": "H",
        "int": "i",
        "uint": "I",
        "int32": "i",
        "uint32": "I",
        "float": "f",
        "float32": "f",
        "double": "d",
        "float64": "d",
    }
    endian = "<" if "little" in fmt else ">"
    fmt_chars = []
    for t, _name in props:
        if t not in type_map:
            raise RuntimeError(f"unsupported PLY prop type {t} in {path}")
        fmt_chars.append(type_map[t])
    if len(fmt_chars) < 3 or fmt_chars[0] not in "fd" or fmt_chars[1] not in "fd":
        raise RuntimeError(f"PLY {path} does not start with float x/y")
    vert_fmt = endian + "".join(fmt_chars)
    vert_size = struct.calcsize(vert_fmt)
    buf = bytearray(body)
    if len(buf) < n_verts * vert_size:
        raise RuntimeError(f"PLY body short: {path}")
    for i in range(n_verts):
        off = i * vert_size
        vals = list(struct.unpack_from(vert_fmt, buf, off))
        x2, y2 = apply_se2_xy(float(vals[0]), float(vals[1]), T, pivot_e=pivot_e, pivot_n=pivot_n)
        vals[0] = type(vals[0])(x2)
        vals[1] = type(vals[1])(y2)
        struct.pack_into(vert_fmt, buf, off, *vals)
    path.write_bytes(raw[: header_end + len(b"end_header")] + nl + bytes(buf))
    return n_verts


def apply_se2_to_planes_json(path: Path, T: dict[str, float], *, pivot_e: float | None = None, pivot_n: float | None = None) -> int:
    """Best-effort transform of planes.json vertex / centroid XY."""
    if not path.is_file():
        return 0
    data = json.loads(path.read_text(encoding="utf-8"))
    n = 0

    def _xf_point(pt: Any) -> Any:
        nonlocal n
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            e2, n2 = apply_se2_xy(float(pt[0]), float(pt[1]), T, pivot_e=pivot_e, pivot_n=pivot_n)
            out = list(pt)
            out[0], out[1] = e2, n2
            n += 1
            return out
        return pt

    planes = data if isinstance(data, list) else data.get("planes") or data.get("facades") or []
    if isinstance(planes, list):
        for pl in planes:
            if not isinstance(pl, dict):
                continue
            for key in ("centroid", "center", "origin"):
                if key in pl:
                    pl[key] = _xf_point(pl[key])
            for key in ("vertices", "corners", "polygon", "points"):
                if key in pl and isinstance(pl[key], list):
                    pl[key] = [_xf_point(p) for p in pl[key]]
            # plane (n, d): for pure translation, d' = d - n·t; with yaw, rotate n_xy
            if "n" in pl and isinstance(pl["n"], (list, tuple)) and len(pl["n"]) >= 2:
                nx, ny = float(pl["n"][0]), float(pl["n"][1])
                yaw = math.radians(float(T.get("yaw_deg", 0.0)))
                c, sn = math.cos(yaw), math.sin(yaw)
                nx2 = c * nx + sn * ny
                ny2 = -sn * nx + c * ny
                nlist = list(pl["n"])
                nlist[0], nlist[1] = nx2, ny2
                pl["n"] = nlist
                if "d" in pl:
                    # d' such that n'·x' = d' with x' = R x + t ≈ n·(R^T (x' - t))
                    tx, ty = float(T.get("tx_m", 0.0)), float(T.get("ty_m", 0.0))
                    pl["d"] = float(pl["d"]) - (nx2 * tx + ny2 * ty)
                    n += 1
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return n



def clip_ply_to_ortho_enu(
    path: Path,
    sw: float,
    sh: float,
    ee: float,
    nn: float,
    *,
    margin_m: float = 2.0,
    dest: Path | None = None,
) -> dict[str, Any]:
    """Drop PLY verts whose XY (ENU e,n) lie outside Ortho bbox ± margin.

    Does not touch façades.obj / quality metrics. Rewrites ``dest`` (default:
    in-place ``path``) with kept vertices only. Returns counts + bbox used.
    """
    if not path.is_file():
        return {"kept": 0, "dropped": 0, "margin_m": float(margin_m), "skipped": True}
    e_lo, e_hi = float(sw) - margin_m, float(ee) + margin_m
    n_lo, n_hi = float(sh) - margin_m, float(nn) + margin_m
    raw = path.read_bytes()
    header_end = raw.find(b"end_header")
    if header_end < 0:
        raise RuntimeError(f"not a PLY: {path}")
    header = raw[:header_end].decode("ascii", errors="replace")
    body = raw[header_end + len(b"end_header") :]
    nl = b"\n"
    if body.startswith(b"\r\n"):
        body = body[2:]
        nl = b"\r\n"
    elif body.startswith(b"\n"):
        body = body[1:]

    fmt = "ascii"
    n_verts = 0
    props: list[tuple[str, str]] = []
    in_vertex = False
    header_lines = header.splitlines()
    for line in header_lines:
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element" and parts[1] == "vertex":
            n_verts = int(parts[2])
            in_vertex = True
        elif parts[0] == "element":
            in_vertex = False
        elif in_vertex and parts[0] == "property":
            props.append((parts[1], parts[2]))

    if n_verts <= 0:
        return {
            "kept": 0,
            "dropped": 0,
            "margin_m": float(margin_m),
            "bbox_enu": [e_lo, n_lo, e_hi, n_hi],
        }

    out_path = dest if dest is not None else path
    dropped = 0

    if fmt == "ascii":
        text_body = body.decode("ascii", errors="replace")
        lines = text_body.splitlines(keepends=True)
        kept_str: list[str] = []
        for i, line in enumerate(lines):
            if i >= n_verts:
                break
            parts = line.split()
            if len(parts) < 3:
                dropped += 1
                continue
            x, y = float(parts[0]), float(parts[1])
            if e_lo <= x <= e_hi and n_lo <= y <= n_hi:
                kept_str.append(line if line.endswith("\n") else line + "\n")
            else:
                dropped += 1
        new_header_lines = []
        for line in header_lines:
            parts = line.strip().split()
            if parts and parts[0] == "element" and parts[1] == "vertex":
                new_header_lines.append(f"element vertex {len(kept_str)}")
            else:
                new_header_lines.append(line.rstrip("\n\r"))
        header_out = ("\n".join(new_header_lines) + "\nend_header").encode("ascii")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(header_out + nl + "".join(kept_str).encode("ascii"))
        kept = len(kept_str)
    else:
        type_map = {
            "char": "b",
            "uchar": "B",
            "int8": "b",
            "uint8": "B",
            "short": "h",
            "ushort": "H",
            "int16": "h",
            "uint16": "H",
            "int": "i",
            "uint": "I",
            "int32": "i",
            "uint32": "I",
            "float": "f",
            "float32": "f",
            "double": "d",
            "float64": "d",
        }
        endian = "<" if "little" in fmt else ">"
        fmt_chars = []
        for t, _name in props:
            if t not in type_map:
                raise RuntimeError(f"unsupported PLY prop type {t} in {path}")
            fmt_chars.append(type_map[t])
        if len(fmt_chars) < 3 or fmt_chars[0] not in "fd" or fmt_chars[1] not in "fd":
            raise RuntimeError(f"PLY {path} does not start with float x/y")
        vert_fmt = endian + "".join(fmt_chars)
        vert_size = struct.calcsize(vert_fmt)
        if len(body) < n_verts * vert_size:
            raise RuntimeError(f"PLY body short: {path}")
        kept_blobs: list[bytes] = []
        for i in range(n_verts):
            off = i * vert_size
            chunk = body[off : off + vert_size]
            vals = struct.unpack_from(vert_fmt, body, off)
            x, y = float(vals[0]), float(vals[1])
            if e_lo <= x <= e_hi and n_lo <= y <= n_hi:
                kept_blobs.append(bytes(chunk))
            else:
                dropped += 1
        new_header_lines = []
        for line in header_lines:
            parts = line.strip().split()
            if parts and parts[0] == "element" and parts[1] == "vertex":
                new_header_lines.append(f"element vertex {len(kept_blobs)}")
            else:
                new_header_lines.append(line.rstrip("\n\r"))
        header_out = ("\n".join(new_header_lines) + "\nend_header").encode("ascii")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(header_out + nl + b"".join(kept_blobs))
        kept = len(kept_blobs)

    log.info(
        "clip_ply_to_ortho_enu %s: kept=%s dropped=%s margin=%.1fm",
        out_path.name,
        kept,
        dropped,
        margin_m,
    )
    return {
        "kept": int(kept),
        "dropped": int(dropped),
        "margin_m": float(margin_m),
        "bbox_enu": [float(e_lo), float(n_lo), float(e_hi), float(n_hi)],
        "path": str(out_path),
        "source": str(path),
    }


def clip_recon_clouds_to_ortho(
    recon_dir: Path,
    sw: float,
    sh: float,
    ee: float,
    nn: float,
    *,
    margin_m: float = 2.0,
    ply_names: tuple[str, ...] = ("cloud.ply", "cloud_photo.ply"),
    write_sidecar: bool = True,
) -> dict[str, Any]:
    """Clip MA product clouds to Ortho ENU ± margin after sat seat.

    Writes ``*_satclipped.ply`` sidecars and replaces the live product PLY so
    Studio picks up the cleaned cloud. Façades are never clipped here.
    """
    summary: dict[str, Any] = {"margin_m": float(margin_m), "clouds": {}}
    total_dropped = 0
    total_kept = 0
    for name in ply_names:
        p = recon_dir / name
        if not p.is_file():
            continue
        sidecar = recon_dir / name.replace(".ply", "_satclipped.ply")
        dest = sidecar if write_sidecar else p
        try:
            stats = clip_ply_to_ortho_enu(
                p, sw, sh, ee, nn, margin_m=margin_m, dest=dest
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("georef clip: skip %s (%s)", p, exc)
            summary["clouds"][name] = {"error": str(exc)}
            continue
        if write_sidecar and dest != p and dest.is_file():
            p.write_bytes(dest.read_bytes())
        summary["clouds"][name] = stats
        total_dropped += int(stats.get("dropped", 0))
        total_kept += int(stats.get("kept", 0))
    summary["kept"] = total_kept
    summary["dropped"] = total_dropped
    return summary



DEFAULT_ZCLEAN_MARGIN_M = 1.5
DEFAULT_ZCLEAN_AABB_INSET_M = 0.5
SINK_DROP_BELOW_GROUND_M = 1.0

DEFAULT_OFFTILE_DILATE_M = 10.0
DEFAULT_OFFTILE_FAR_M = 15.0
DEFAULT_OFFTILE_Z_OUT_M = 8.0
DEFAULT_OFFTILE_CAM_CORRIDOR_M = 6.0


def load_roof_shell_aabbs(
    roofs_json: Path,
    *,
    aabb_inset_m: float = DEFAULT_ZCLEAN_AABB_INSET_M,
) -> list[dict[str, float]]:
    """Load ``kind=roof`` shells as inset AABB + shell_z for soft Z gate.

    Uses existing ``assign_shell_z`` product ``roofs.json`` — never invents Z from BAG.
    """
    if not roofs_json.is_file():
        return []
    data = json.loads(roofs_json.read_text(encoding="utf-8"))
    shells = data.get("shells") if isinstance(data, dict) else data
    if not isinstance(shells, list):
        return []
    out: list[dict[str, float]] = []
    inset = float(aabb_inset_m)
    for sh in shells:
        if not isinstance(sh, dict) or sh.get("kind") != "roof":
            continue
        aabb = sh.get("aabb_enu") or []
        if len(aabb) < 2:
            continue
        try:
            es = [float(p[0]) for p in aabb]
            ns = [float(p[1]) for p in aabb]
            z = float(sh["z"])
        except (TypeError, ValueError, KeyError, IndexError):
            continue
        if not math.isfinite(z):
            continue
        e_lo, e_hi = min(es) + inset, max(es) - inset
        n_lo, n_hi = min(ns) + inset, max(ns) - inset
        if e_hi <= e_lo or n_hi <= n_lo:
            # Degenerate after inset — keep un-inset AABB (tiny roofs)
            e_lo, e_hi = min(es), max(es)
            n_lo, n_hi = min(ns), max(ns)
        out.append(
            {
                "id": str(sh.get("id") or ""),
                "e_lo": float(e_lo),
                "e_hi": float(e_hi),
                "n_lo": float(n_lo),
                "n_hi": float(n_hi),
                "shell_z": float(z),
                "ground_z": float(sh["ground_z"]) if sh.get("ground_z") is not None else float("nan"),
            }
        )
    return out


def _roof_shell_z_for_xy(
    e: float,
    n: float,
    roofs: list[dict[str, float]],
) -> float | None:
    """Max shell_z among roof AABBs containing (e,n); None if outside all roofs."""
    matched: list[float] = []
    for r in roofs:
        if r["e_lo"] <= e <= r["e_hi"] and r["n_lo"] <= n <= r["n_hi"]:
            matched.append(r["shell_z"])
    if not matched:
        return None
    return max(matched)


def zclean_ply_against_roof_aabbs(
    path: Path,
    roofs: list[dict[str, float]],
    *,
    margin_m: float = DEFAULT_ZCLEAN_MARGIN_M,
    drop_sinks: bool = False,
    ground_z: float | None = None,
    dest: Path | None = None,
) -> dict[str, Any]:
    """Drop PLY verts inside sat roof AABBs with ``z > shell_z + margin_m``.

    Points outside all roof footprints are kept (yards/street untouched).
    Does not touch façades.obj / roofs.obj. Writes ``dest`` (default in-place).
    """
    if not path.is_file():
        return {
            "kept": 0,
            "dropped": 0,
            "margin_m": float(margin_m),
            "skipped": True,
            "reason": "missing_ply",
        }
    if not roofs:
        return {
            "kept": 0,
            "dropped": 0,
            "margin_m": float(margin_m),
            "skipped": True,
            "reason": "no_roof_shells",
        }

    raw = path.read_bytes()
    header_end = raw.find(b"end_header")
    if header_end < 0:
        raise RuntimeError(f"not a PLY: {path}")
    header = raw[:header_end].decode("ascii", errors="replace")
    body = raw[header_end + len(b"end_header") :]
    nl = b"\n"
    if body.startswith(b"\r\n"):
        body = body[2:]
        nl = b"\r\n"
    elif body.startswith(b"\n"):
        body = body[1:]

    fmt = "ascii"
    n_verts = 0
    props: list[tuple[str, str]] = []
    in_vertex = False
    header_lines = header.splitlines()
    for line in header_lines:
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element" and parts[1] == "vertex":
            n_verts = int(parts[2])
            in_vertex = True
        elif parts[0] == "element":
            in_vertex = False
        elif in_vertex and parts[0] == "property":
            props.append((parts[1], parts[2]))

    if n_verts <= 0:
        return {"kept": 0, "dropped": 0, "margin_m": float(margin_m), "n_roofs": len(roofs)}

    margin = float(margin_m)
    gnd = float(ground_z) if ground_z is not None and math.isfinite(float(ground_z)) else None
    # Fallback ground from shells
    if gnd is None:
        gs = [r["ground_z"] for r in roofs if math.isfinite(r.get("ground_z", float("nan")))]
        if gs:
            gnd = float(sum(gs) / len(gs))

    def keep_xyz(x: float, y: float, z: float) -> bool:
        shell_z = _roof_shell_z_for_xy(x, y, roofs)
        if shell_z is None:
            return True  # outside roof footprints
        if z > shell_z + margin:
            return False
        if drop_sinks and gnd is not None and z < gnd - SINK_DROP_BELOW_GROUND_M:
            return False
        return True

    out_path = dest if dest is not None else path
    dropped = 0

    if fmt == "ascii":
        text_body = body.decode("ascii", errors="replace")
        lines = text_body.splitlines(keepends=True)
        kept_str: list[str] = []
        for i, line in enumerate(lines):
            if i >= n_verts:
                break
            parts = line.split()
            if len(parts) < 3:
                dropped += 1
                continue
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            if keep_xyz(x, y, z):
                kept_str.append(line if line.endswith("\n") else line + "\n")
            else:
                dropped += 1
        new_header_lines = []
        for line in header_lines:
            parts = line.strip().split()
            if parts and parts[0] == "element" and parts[1] == "vertex":
                new_header_lines.append(f"element vertex {len(kept_str)}")
            else:
                new_header_lines.append(line.rstrip("\n\r"))
        header_out = ("\n".join(new_header_lines) + "\nend_header").encode("ascii")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(header_out + nl + "".join(kept_str).encode("ascii"))
        kept = len(kept_str)
    else:
        type_map = {
            "char": "b",
            "uchar": "B",
            "int8": "b",
            "uint8": "B",
            "short": "h",
            "ushort": "H",
            "int16": "h",
            "uint16": "H",
            "int": "i",
            "uint": "I",
            "int32": "i",
            "uint32": "I",
            "float": "f",
            "float32": "f",
            "double": "d",
            "float64": "d",
        }
        endian = "<" if "little" in fmt else ">"
        fmt_chars = []
        for t, _name in props:
            if t not in type_map:
                raise RuntimeError(f"unsupported PLY prop type {t} in {path}")
            fmt_chars.append(type_map[t])
        if len(fmt_chars) < 3 or fmt_chars[0] not in "fd" or fmt_chars[1] not in "fd":
            raise RuntimeError(f"PLY {path} does not start with float x/y")
        if fmt_chars[2] not in "fd":
            raise RuntimeError(f"PLY {path} z is not float/double")
        vert_fmt = endian + "".join(fmt_chars)
        vert_size = struct.calcsize(vert_fmt)
        if len(body) < n_verts * vert_size:
            raise RuntimeError(f"PLY body short: {path}")
        kept_blobs: list[bytes] = []
        for i in range(n_verts):
            off = i * vert_size
            chunk = body[off : off + vert_size]
            vals = struct.unpack_from(vert_fmt, body, off)
            x, y, z = float(vals[0]), float(vals[1]), float(vals[2])
            if keep_xyz(x, y, z):
                kept_blobs.append(bytes(chunk))
            else:
                dropped += 1
        new_header_lines = []
        for line in header_lines:
            parts = line.strip().split()
            if parts and parts[0] == "element" and parts[1] == "vertex":
                new_header_lines.append(f"element vertex {len(kept_blobs)}")
            else:
                new_header_lines.append(line.rstrip("\n\r"))
        header_out = ("\n".join(new_header_lines) + "\nend_header").encode("ascii")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(header_out + nl + b"".join(kept_blobs))
        kept = len(kept_blobs)

    log.info(
        "zclean_ply_against_roof_aabbs %s: kept=%s dropped=%s margin=%.1fm roofs=%s",
        out_path.name,
        kept,
        dropped,
        margin,
        len(roofs),
    )
    return {
        "kept": int(kept),
        "dropped": int(dropped),
        "margin_m": float(margin),
        "n_roofs": len(roofs),
        "drop_sinks": bool(drop_sinks),
        "path": str(out_path),
        "source": str(path),
    }


def zclean_recon_clouds(
    recon_dir: Path,
    *,
    margin_m: float = DEFAULT_ZCLEAN_MARGIN_M,
    aabb_inset_m: float = DEFAULT_ZCLEAN_AABB_INSET_M,
    drop_sinks: bool = False,
    ply_names: tuple[str, ...] = ("cloud.ply",),
    replace_product: bool = False,
) -> dict[str, Any]:
    """Opt-in soft Z gate: write ``cloud_zclean.ply`` (+ bak); leave product cloud.

    Sacred: does **not** hungry-XY-clip; façades/roofs shells untouched; default
    product stays unclipped ``cloud.ply`` unless ``replace_product=True``.
    """
    roofs_path = Path(recon_dir) / "roofs.json"
    roofs = load_roof_shell_aabbs(roofs_path, aabb_inset_m=aabb_inset_m)
    summary: dict[str, Any] = {
        "margin_m": float(margin_m),
        "aabb_inset_m": float(aabb_inset_m),
        "drop_sinks": bool(drop_sinks),
        "n_roofs": len(roofs),
        "replace_product": bool(replace_product),
        "clouds": {},
    }
    if not roofs:
        summary["skipped"] = True
        summary["reason"] = "no_roof_shells"
        summary["kept"] = 0
        summary["dropped"] = 0
        return summary

    total_kept = 0
    total_dropped = 0
    for name in ply_names:
        src = Path(recon_dir) / name
        if not src.is_file():
            continue
        # Sidecar: cloud.ply → cloud_zclean.ply; cloud_photo.ply → cloud_photo_zclean.ply
        if name == "cloud.ply":
            dest_name = "cloud_zclean.ply"
        elif name.endswith(".ply"):
            dest_name = name[:-4] + "_zclean.ply"
        else:
            dest_name = name + "_zclean"
        dest = Path(recon_dir) / dest_name
        if dest.is_file():
            bak = Path(str(dest) + ".bak")
            bak.write_bytes(dest.read_bytes())
        # Bak source once when replacing product (reversible)
        src_bak = Path(str(src) + ".bak")
        if replace_product and src.is_file() and not src_bak.is_file():
            src_bak.write_bytes(src.read_bytes())

        try:
            stats = zclean_ply_against_roof_aabbs(
                src,
                roofs,
                margin_m=margin_m,
                drop_sinks=drop_sinks,
                dest=dest,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("zclean: skip %s (%s)", src, exc)
            summary["clouds"][name] = {"error": str(exc)}
            continue

        if replace_product and dest.is_file():
            src.write_bytes(dest.read_bytes())

        summary["clouds"][name] = stats
        total_kept += int(stats.get("kept", 0))
        total_dropped += int(stats.get("dropped", 0))

    summary["kept"] = total_kept
    summary["dropped"] = total_dropped
    return summary


def _filter_ply_by_xyz(
    path: Path,
    keep_xyz,
    *,
    dest: Path | None = None,
) -> tuple[int, int, Path]:
    """Rewrite PLY keeping verts where ``keep_xyz(x,y,z)`` is true.

    Returns ``(kept, dropped, out_path)``. Supports ascii + binary_little/big.
    """
    raw = path.read_bytes()
    header_end = raw.find(b"end_header")
    if header_end < 0:
        raise RuntimeError(f"not a PLY: {path}")
    header = raw[:header_end].decode("ascii", errors="replace")
    body = raw[header_end + len(b"end_header") :]
    nl = b"\n"
    if body.startswith(b"\r\n"):
        body = body[2:]
        nl = b"\r\n"
    elif body.startswith(b"\n"):
        body = body[1:]

    fmt = "ascii"
    n_verts = 0
    props: list[tuple[str, str]] = []
    in_vertex = False
    header_lines = header.splitlines()
    for line in header_lines:
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element" and parts[1] == "vertex":
            n_verts = int(parts[2])
            in_vertex = True
        elif parts[0] == "element":
            in_vertex = False
        elif in_vertex and parts[0] == "property":
            props.append((parts[1], parts[2]))

    out_path = dest if dest is not None else path
    if n_verts <= 0:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(raw)
        return 0, 0, out_path

    dropped = 0
    if fmt == "ascii":
        text_body = body.decode("ascii", errors="replace")
        lines = text_body.splitlines(keepends=True)
        kept_str: list[str] = []
        for i, line in enumerate(lines):
            if i >= n_verts:
                break
            parts = line.split()
            if len(parts) < 3:
                dropped += 1
                continue
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            if keep_xyz(x, y, z):
                kept_str.append(line if line.endswith("\n") else line + "\n")
            else:
                dropped += 1
        new_header_lines = []
        for line in header_lines:
            parts = line.strip().split()
            if parts and parts[0] == "element" and parts[1] == "vertex":
                new_header_lines.append(f"element vertex {len(kept_str)}")
            else:
                new_header_lines.append(line.rstrip("\n\r"))
        header_out = ("\n".join(new_header_lines) + "\nend_header").encode("ascii")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(header_out + nl + "".join(kept_str).encode("ascii"))
        kept = len(kept_str)
    else:
        type_map = {
            "char": "b",
            "uchar": "B",
            "int8": "b",
            "uint8": "B",
            "short": "h",
            "ushort": "H",
            "int16": "h",
            "uint16": "H",
            "int": "i",
            "uint": "I",
            "int32": "i",
            "uint32": "I",
            "float": "f",
            "float32": "f",
            "double": "d",
            "float64": "d",
        }
        endian = "<" if "little" in fmt else ">"
        fmt_chars = []
        for t, _name in props:
            if t not in type_map:
                raise RuntimeError(f"unsupported PLY prop type {t} in {path}")
            fmt_chars.append(type_map[t])
        if len(fmt_chars) < 3 or fmt_chars[0] not in "fd" or fmt_chars[1] not in "fd":
            raise RuntimeError(f"PLY {path} does not start with float x/y")
        if fmt_chars[2] not in "fd":
            raise RuntimeError(f"PLY {path} z is not float/double")
        vert_fmt = endian + "".join(fmt_chars)
        vert_size = struct.calcsize(vert_fmt)
        if len(body) < n_verts * vert_size:
            raise RuntimeError(f"PLY body short: {path}")
        kept_blobs: list[bytes] = []
        for i in range(n_verts):
            off = i * vert_size
            chunk = body[off : off + vert_size]
            vals = struct.unpack_from(vert_fmt, body, off)
            x, y, z = float(vals[0]), float(vals[1]), float(vals[2])
            if keep_xyz(x, y, z):
                kept_blobs.append(bytes(chunk))
            else:
                dropped += 1
        new_header_lines = []
        for line in header_lines:
            parts = line.strip().split()
            if parts and parts[0] == "element" and parts[1] == "vertex":
                new_header_lines.append(f"element vertex {len(kept_blobs)}")
            else:
                new_header_lines.append(line.rstrip("\n\r"))
        header_out = ("\n".join(new_header_lines) + "\nend_header").encode("ascii")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(header_out + nl + b"".join(kept_blobs))
        kept = len(kept_blobs)

    return int(kept), int(dropped), out_path


def load_support_shell_aabbs(
    roofs_json: Path,
    *,
    kinds: tuple[str, ...] = ("roof", "yard", "street"),
    dilate_m: float = 0.0,
) -> list[dict[str, float]]:
    """Load roof∪yard∪street shell AABBs, optionally dilated (soft support).

    Unlike ``load_roof_shell_aabbs``, does **not** inset — dilation expands
    footprints so yards/street stay inside the keep mask.
    """
    if not roofs_json.is_file():
        return []
    data = json.loads(roofs_json.read_text(encoding="utf-8"))
    shells = data.get("shells") if isinstance(data, dict) else data
    if not isinstance(shells, list):
        return []
    kind_set = {str(k) for k in kinds}
    dilate = float(dilate_m)
    out: list[dict[str, float]] = []
    for sh in shells:
        if not isinstance(sh, dict) or str(sh.get("kind") or "") not in kind_set:
            continue
        aabb = sh.get("aabb_enu") or []
        if len(aabb) < 2:
            continue
        try:
            es = [float(p[0]) for p in aabb]
            ns = [float(p[1]) for p in aabb]
        except (TypeError, ValueError, IndexError):
            continue
        e_lo, e_hi = min(es) - dilate, max(es) + dilate
        n_lo, n_hi = min(ns) - dilate, max(ns) + dilate
        z_raw = sh.get("z")
        try:
            z = float(z_raw) if z_raw is not None else float("nan")
        except (TypeError, ValueError):
            z = float("nan")
        g_raw = sh.get("ground_z")
        try:
            gz = float(g_raw) if g_raw is not None else float("nan")
        except (TypeError, ValueError):
            gz = float("nan")
        out.append(
            {
                "id": str(sh.get("id") or ""),
                "kind": str(sh.get("kind") or ""),
                "e_lo": float(e_lo),
                "e_hi": float(e_hi),
                "n_lo": float(n_lo),
                "n_hi": float(n_hi),
                "shell_z": float(z),
                "ground_z": float(gz),
            }
        )
    return out


def load_cam_corridor_aabbs(
    poses_path: Path,
    *,
    half_m: float = DEFAULT_OFFTILE_CAM_CORRIDOR_M,
) -> list[dict[str, float]]:
    """Axis-aligned pads around cam ENU (e,n) so the drive path stays in support."""
    if half_m <= 0 or not poses_path.is_file():
        return []
    raw = json.loads(poses_path.read_text(encoding="utf-8"))
    poses = raw if isinstance(raw, list) else (raw.get("poses") or raw.get("cameras") or [])
    if not isinstance(poses, list):
        return []
    r = float(half_m)
    out: list[dict[str, float]] = []
    for i, p in enumerate(poses):
        if not isinstance(p, dict):
            continue
        try:
            e = float(p.get("e", p.get("e_gps")))
            n = float(p.get("n", p.get("n_gps")))
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(e) and math.isfinite(n)):
            continue
        out.append(
            {
                "id": f"cam_{i}",
                "kind": "cam",
                "e_lo": e - r,
                "e_hi": e + r,
                "n_lo": n - r,
                "n_hi": n + r,
                "shell_z": float("nan"),
                "ground_z": float("nan"),
            }
        )
    return out


def dist_xy_to_support(e: float, n: float, aabbs: list[dict[str, float]]) -> float:
    """Min XY distance to any AABB; 0 if inside at least one."""
    if not aabbs:
        return float("inf")
    best = float("inf")
    for a in aabbs:
        dx = 0.0
        if e < a["e_lo"]:
            dx = a["e_lo"] - e
        elif e > a["e_hi"]:
            dx = e - a["e_hi"]
        dy = 0.0
        if n < a["n_lo"]:
            dy = a["n_lo"] - n
        elif n > a["n_hi"]:
            dy = n - a["n_hi"]
        d = math.hypot(dx, dy)
        if d < best:
            best = d
            if best == 0.0:
                return 0.0
    return float(best)


def resolve_local_ground_z(
    roofs_json: Path,
    *,
    cloud_zs: list[float] | None = None,
) -> float:
    """Ground reference for off-tile z_out gate — not BAG.

    Order: ``cam_u_median - 2.5``, else median yard shell z, else cloud p10, else 0.
    """
    cam_u = None
    yard_zs: list[float] = []
    if roofs_json.is_file():
        data = json.loads(roofs_json.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            raw_cam = data.get("cam_u_median")
            try:
                if raw_cam is not None and math.isfinite(float(raw_cam)):
                    cam_u = float(raw_cam)
            except (TypeError, ValueError):
                cam_u = None
            shells = data.get("shells") or []
            if isinstance(shells, list):
                for sh in shells:
                    if not isinstance(sh, dict) or sh.get("kind") != "yard":
                        continue
                    try:
                        z = float(sh["z"])
                    except (TypeError, ValueError, KeyError):
                        continue
                    if math.isfinite(z):
                        yard_zs.append(z)
    if cam_u is not None:
        return float(cam_u) - 2.5
    if yard_zs:
        yard_zs.sort()
        return float(yard_zs[len(yard_zs) // 2])
    if cloud_zs:
        zs = sorted(float(z) for z in cloud_zs if math.isfinite(float(z)))
        if zs:
            idx = max(0, min(len(zs) - 1, int(0.10 * (len(zs) - 1))))
            return float(zs[idx])
    return 0.0


def offtile_ply_against_support(
    path: Path,
    support: list[dict[str, float]],
    *,
    far_m: float = DEFAULT_OFFTILE_FAR_M,
    z_out_m: float = DEFAULT_OFFTILE_Z_OUT_M,
    local_ground_z: float = 0.0,
    dest: Path | None = None,
) -> dict[str, Any]:
    """Soft support gate: keep in dilated support; drop far halo / sky smear.

    Sacred: not Ortho XY±2 m; façades untouched; yards inside support kept.
    """
    if not path.is_file():
        return {
            "kept": 0,
            "dropped": 0,
            "dropped_far": 0,
            "dropped_z": 0,
            "far_m": float(far_m),
            "z_out_m": float(z_out_m),
            "local_ground_z": float(local_ground_z),
            "skipped": True,
            "reason": "missing_ply",
        }
    if not support:
        return {
            "kept": 0,
            "dropped": 0,
            "dropped_far": 0,
            "dropped_z": 0,
            "far_m": float(far_m),
            "z_out_m": float(z_out_m),
            "local_ground_z": float(local_ground_z),
            "skipped": True,
            "reason": "no_support",
        }

    far = float(far_m)
    z_out = float(z_out_m)
    gnd = float(local_ground_z)
    dropped_far = 0
    dropped_z = 0

    def keep_xyz(x: float, y: float, z: float) -> bool:
        nonlocal dropped_far, dropped_z
        d = dist_xy_to_support(x, y, support)
        if d <= 0.0:
            return True
        if z > gnd + z_out:
            dropped_z += 1
            return False
        if d > far:
            dropped_far += 1
            return False
        return True

    kept, dropped, out_path = _filter_ply_by_xyz(path, keep_xyz, dest=dest)
    log.info(
        "offtile_ply_against_support %s: kept=%s dropped=%s (far=%s z=%s) "
        "far_m=%.1f z_out_m=%.1f gnd=%.2f support=%s",
        out_path.name,
        kept,
        dropped,
        dropped_far,
        dropped_z,
        far,
        z_out,
        gnd,
        len(support),
    )
    return {
        "kept": int(kept),
        "dropped": int(dropped),
        "dropped_far": int(dropped_far),
        "dropped_z": int(dropped_z),
        "far_m": float(far),
        "z_out_m": float(z_out),
        "local_ground_z": float(gnd),
        "n_support": len(support),
        "path": str(out_path),
        "source": str(path),
    }


def offtile_recon_clouds(
    recon_dir: Path,
    *,
    dilate_m: float = DEFAULT_OFFTILE_DILATE_M,
    far_m: float = DEFAULT_OFFTILE_FAR_M,
    z_out_m: float = DEFAULT_OFFTILE_Z_OUT_M,
    cam_corridor_m: float = DEFAULT_OFFTILE_CAM_CORRIDOR_M,
    poses_path: Path | None = None,
    ply_names: tuple[str, ...] = ("cloud.ply",),
    replace_product: bool = False,
) -> dict[str, Any]:
    """Opt-in soft support gate: write ``cloud_offtile.ply`` (+ bak).

    Dilates roof∪yard∪street (~10 m); drops pts farther than ``far_m`` from
    support or with z ≫ local ground (+``z_out_m``). Never Ortho±2 m XY clip;
    façades/roofs shells untouched; default product stays ``cloud.ply``.
    """
    roofs_path = Path(recon_dir) / "roofs.json"
    support = load_support_shell_aabbs(
        roofs_path, kinds=("roof", "yard", "street"), dilate_m=dilate_m
    )
    n_shell = len(support)
    if poses_path is None:
        # recon/ -> run root -> align/poses.json
        guess = Path(recon_dir).parent / "align" / "poses.json"
        poses_path = guess if guess.is_file() else None
    n_cam = 0
    if poses_path is not None and float(cam_corridor_m) > 0:
        cams = load_cam_corridor_aabbs(Path(poses_path), half_m=float(cam_corridor_m))
        n_cam = len(cams)
        support = support + cams

    local_gnd = resolve_local_ground_z(roofs_path)
    summary: dict[str, Any] = {
        "dilate_m": float(dilate_m),
        "far_m": float(far_m),
        "z_out_m": float(z_out_m),
        "cam_corridor_m": float(cam_corridor_m),
        "local_ground_z": float(local_gnd),
        "n_support_shells": int(n_shell),
        "n_cam_pads": int(n_cam),
        "n_support": len(support),
        "replace_product": bool(replace_product),
        "clouds": {},
    }
    if not support:
        summary["skipped"] = True
        summary["reason"] = "no_support"
        summary["kept"] = 0
        summary["dropped"] = 0
        summary["dropped_far"] = 0
        summary["dropped_z"] = 0
        return summary

    total_kept = 0
    total_dropped = 0
    total_far = 0
    total_z = 0
    for name in ply_names:
        src = Path(recon_dir) / name
        if not src.is_file():
            continue
        if name == "cloud.ply":
            dest_name = "cloud_offtile.ply"
        elif name.endswith(".ply"):
            dest_name = name[:-4] + "_offtile.ply"
        else:
            dest_name = name + "_offtile"
        dest = Path(recon_dir) / dest_name
        if dest.is_file():
            bak = Path(str(dest) + ".bak")
            bak.write_bytes(dest.read_bytes())
        src_bak = Path(str(src) + ".bak")
        if replace_product and src.is_file() and not src_bak.is_file():
            src_bak.write_bytes(src.read_bytes())

        try:
            stats = offtile_ply_against_support(
                src,
                support,
                far_m=far_m,
                z_out_m=z_out_m,
                local_ground_z=local_gnd,
                dest=dest,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("offtile: skip %s (%s)", src, exc)
            summary["clouds"][name] = {"error": str(exc)}
            continue

        if replace_product and dest.is_file():
            src.write_bytes(dest.read_bytes())

        summary["clouds"][name] = stats
        total_kept += int(stats.get("kept", 0))
        total_dropped += int(stats.get("dropped", 0))
        total_far += int(stats.get("dropped_far", 0))
        total_z += int(stats.get("dropped_z", 0))

    summary["kept"] = total_kept
    summary["dropped"] = total_dropped
    summary["dropped_far"] = total_far
    summary["dropped_z"] = total_z
    return summary



def seat_recon_artefacts(
    recon_dir: Path,
    T: dict[str, float],
    *,
    pivot_e: float | None = None,
    pivot_n: float | None = None,
    ply_names: tuple[str, ...] = ("cloud.ply", "cloud_photo.ply"),
) -> dict[str, int]:
    """Apply one SE(2) to product PLY/OBJ/planes so Studio matches sat-seated cams."""
    stats: dict[str, int] = {}
    for name in ply_names:
        p = recon_dir / name
        if p.is_file():
            try:
                stats[name] = apply_se2_to_ply(p, T, pivot_e=pivot_e, pivot_n=pivot_n)
            except Exception as exc:  # noqa: BLE001
                log.warning("georef: skip %s (%s)", p, exc)
                stats[name] = -1
    obj = recon_dir / "facades.obj"
    if obj.is_file():
        stats["facades.obj"] = apply_se2_to_ascii_obj(obj, T, pivot_e=pivot_e, pivot_n=pivot_n)
    planes = recon_dir / "planes.json"
    if planes.is_file():
        stats["planes.json"] = apply_se2_to_planes_json(planes, T, pivot_e=pivot_e, pivot_n=pivot_n)
    return stats


def refresh_scene_cameras(
    scene_path: Path,
    poses: list[dict[str, Any]],
    georef: dict[str, Any] | None,
) -> bool:
    """Update scene.json camera ENU + georef without a full densify re-run."""
    if not scene_path.is_file():
        return False
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    by_pano = {p.get("pano_id"): p for p in poses}
    cams = scene.get("cameras") or []
    for cam in cams:
        src = by_pano.get(cam.get("pano_id"))
        if not src:
            # fall back: match by index order if same length
            continue
        cam["e"] = src["e"]
        cam["n"] = src["n"]
        cam["u"] = src.get("u", cam.get("u"))
        cam["heading"] = src["heading"]
        if "sat_score" in src:
            cam["sat_score"] = src["sat_score"]
        if "sat_ncc" in src:
            cam["sat_ncc"] = src["sat_ncc"]
        if "sat_edge" in src:
            cam["sat_edge"] = src["sat_edge"]
        if "residual_m" in src:
            cam["residual_m"] = src["residual_m"]
        cam["bag_snapped"] = bool(src.get("bag_snapped"))
    if len(cams) == len(poses) and not any(c.get("pano_id") in by_pano for c in cams):
        for cam, src in zip(cams, poses):
            cam["e"], cam["n"] = src["e"], src["n"]
            cam["heading"] = src["heading"]
            cam["u"] = src.get("u", cam.get("u"))
    if georef is not None:
        scene["georef"] = georef
    scene_path.write_text(json.dumps(scene, indent=2), encoding="utf-8")
    return True
