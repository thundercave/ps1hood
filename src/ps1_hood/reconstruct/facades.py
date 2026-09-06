"""Very rough vertical-plane facades + ground quad — the PS1-looking pass."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _read_ply_xyz(path: Path) -> np.ndarray:
    pts = []
    with path.open("r", encoding="ascii") as fh:
        header = True
        for line in fh:
            if header:
                if line.strip() == "end_header":
                    header = False
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            pts.append((float(parts[0]), float(parts[1]), float(parts[2])))
    return np.array(pts, dtype=np.float64)


def extract_facades(ply_path: Path, dest_obj: Path, n_planes: int = 12) -> dict:
    """RANSAC vertical planes. Enough to get a blocky hood standing up."""
    xyz = _read_ply_xyz(ply_path)
    if len(xyz) < 200:
        raise RuntimeError("not enough points for facade extraction")
    remaining = xyz.copy()
    planes: list[dict] = []
    rng = np.random.default_rng(0)
    for _ in range(n_planes):
        if len(remaining) < 80:
            break
        best_inliers = None
        best_n = None
        best_d = None
        for _try in range(80):
            i0, i1 = rng.integers(0, len(remaining), size=2)
            p0, p1 = remaining[i0], remaining[i1]
            # force vertical: normal in XY
            v = p1[:2] - p0[:2]
            if np.linalg.norm(v) < 0.2:
                continue
            nrm = np.array([-v[1], v[0]], dtype=np.float64)
            nrm /= np.linalg.norm(nrm)
            d = float(nrm @ p0[:2])
            dist = np.abs(remaining[:, :2] @ nrm - d)
            inl = dist < 0.45
            if best_inliers is None or int(inl.sum()) > int(best_inliers.sum()):
                best_inliers = inl
                best_n = nrm
                best_d = d
        if best_inliers is None or int(best_inliers.sum()) < 60:
            break
        pts = remaining[best_inliers]
        planes.append(
            {
                "nx": float(best_n[0]),
                "ny": float(best_n[1]),
                "d": float(best_d),
                "min": pts.min(axis=0).tolist(),
                "max": pts.max(axis=0).tolist(),
                "count": int(len(pts)),
            }
        )
        remaining = remaining[~best_inliers]

    dest_obj.parent.mkdir(parents=True, exist_ok=True)
    _write_obj(dest_obj, planes, xyz)
    return {"path": str(dest_obj), "planes": len(planes)}


def _write_obj(path: Path, planes: list[dict], xyz: np.ndarray) -> None:
    # ground quad from point AABB
    mn, mx = xyz.min(axis=0), xyz.max(axis=0)
    verts: list[tuple[float, float, float]] = [
        (mn[0], mn[1], 0.0),
        (mx[0], mn[1], 0.0),
        (mx[0], mx[1], 0.0),
        (mn[0], mx[1], 0.0),
    ]
    faces: list[tuple[int, int, int, int]] = [(1, 2, 3, 4)]
    for pl in planes:
        nx, ny, d = pl["nx"], pl["ny"], pl["d"]
        # tangent along the facade
        tx, ty = -ny, nx
        c = np.array(pl["min"][:2]) * 0.5 + np.array(pl["max"][:2]) * 0.5
        # project centre onto the plane
        c = c - (c @ np.array([nx, ny]) - d) * np.array([nx, ny])
        half = 0.5 * np.linalg.norm(np.array(pl["max"][:2]) - np.array(pl["min"][:2]))
        z0 = max(0.0, float(pl["min"][2]))
        z1 = max(z0 + 2.0, float(pl["max"][2]))
        p1 = (c[0] - tx * half, c[1] - ty * half, z0)
        p2 = (c[0] + tx * half, c[1] + ty * half, z0)
        p3 = (c[0] + tx * half, c[1] + ty * half, z1)
        p4 = (c[0] - tx * half, c[1] - ty * half, z1)
        base = len(verts)
        verts.extend([p1, p2, p3, p4])
        faces.append((base + 1, base + 2, base + 3, base + 4))
    with path.open("w", encoding="ascii") as fh:
        fh.write("# ps1-hood facade blockout\n")
        for v in verts:
            fh.write(f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n")
        for f in faces:
            fh.write(f"f {f[0]} {f[1]} {f[2]} {f[3]}\n")
