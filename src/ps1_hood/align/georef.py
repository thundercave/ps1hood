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
        out.update(extra)
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
