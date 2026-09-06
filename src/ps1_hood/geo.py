"""WGS84 geodetic math and a local east-north-up frame for one bbox."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


@dataclass(frozen=True)
class BBox:
    south: float
    west: float
    north: float
    east: float

    def __post_init__(self) -> None:
        if self.south >= self.north:
            raise ValueError("bbox south must be < north")
        if self.west >= self.east:
            raise ValueError("bbox west must be < east")
        if not (-90.0 <= self.south <= 90.0 and -90.0 <= self.north <= 90.0):
            raise ValueError("latitude out of range")
        if not (-180.0 <= self.west <= 180.0 and -180.0 <= self.east <= 180.0):
            raise ValueError("longitude out of range")

    def contains(self, lat: float, lon: float, slop_deg: float = 0.0) -> bool:
        return (
            self.south - slop_deg <= lat <= self.north + slop_deg
            and self.west - slop_deg <= lon <= self.east + slop_deg
        )

    def contains_m(self, lat: float, lon: float, slop_m: float = 0.0) -> bool:
        """True if the point is inside the rectangle, or within slop_m of its border."""
        if self.contains(lat, lon):
            return True
        if slop_m <= 0.0:
            return False
        clat = min(max(lat, self.south), self.north)
        clon = min(max(lon, self.west), self.east)
        return haversine_m(lat, lon, clat, clon) <= slop_m

    def padded(self, metres: float) -> "BBox":
        lat, _ = self.center()
        dlat = metres / 111_320.0
        dlon = metres / max(1e-6, 111_320.0 * math.cos(math.radians(lat)))
        return BBox(self.south - dlat, self.west - dlon, self.north + dlat, self.east + dlon)

    def union_point(self, lat: float, lon: float) -> "BBox":
        return BBox(
            min(self.south, lat),
            min(self.west, lon),
            max(self.north, lat),
            max(self.east, lon),
        )

    def center(self) -> tuple[float, float]:
        return ((self.south + self.north) / 2.0, (self.west + self.east) / 2.0)

    def width_m(self) -> float:
        lat, _ = self.center()
        return haversine_m(lat, self.west, lat, self.east)

    def height_m(self) -> float:
        _, lon = self.center()
        return haversine_m(self.south, lon, self.north, lon)

    def area_deg2(self) -> float:
        return abs(self.north - self.south) * abs(self.east - self.west)

    def as_dict(self) -> dict[str, float]:
        return {
            "south": self.south,
            "west": self.west,
            "north": self.north,
            "east": self.east,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BBox":
        return cls(
            south=float(data["south"]),
            west=float(data["west"]),
            north=float(data["north"]),
            east=float(data["east"]),
        )


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial bearing from point 1 to 2, degrees clockwise from north."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlmb = math.radians(lon2 - lon1)
    x = math.sin(dlmb) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dlmb)
    return wrap_heading(math.degrees(math.atan2(x, y)))


def wrap_heading(deg: float) -> float:
    return deg % 360.0


def heading_diff(a: float, b: float) -> float:
    """Smallest signed difference b - a in (-180, 180]."""
    d = (b - a + 180.0) % 360.0 - 180.0
    return 180.0 if d == -180.0 else d


def geodetic_to_ecef(lat: float, lon: float, alt: float = 0.0) -> tuple[float, float, float]:
    phi = math.radians(lat)
    lam = math.radians(lon)
    sphi, cphi = math.sin(phi), math.cos(phi)
    slam, clam = math.sin(lam), math.cos(lam)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sphi * sphi)
    x = (n + alt) * cphi * clam
    y = (n + alt) * cphi * slam
    z = (n * (1.0 - WGS84_E2) + alt) * sphi
    return x, y, z


def ecef_to_geodetic(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Closed-form approximation good to millimetres at street scale."""
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    phi = math.atan2(z, p * (1.0 - WGS84_E2))
    for _ in range(8):
        sphi = math.sin(phi)
        n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sphi * sphi)
        alt = p / math.cos(phi) - n
        phi = math.atan2(z, p * (1.0 - WGS84_E2 * n / (n + alt)))
    return math.degrees(phi), math.degrees(lon), alt


@dataclass
class LocalFrame:
    """East-north-up metres, origin at the bbox centre on the ellipsoid."""

    lat0: float
    lon0: float
    alt0: float = 0.0

    def __post_init__(self) -> None:
        self._origin = geodetic_to_ecef(self.lat0, self.lon0, self.alt0)
        phi = math.radians(self.lat0)
        lam = math.radians(self.lon0)
        sphi, cphi = math.sin(phi), math.cos(phi)
        slam, clam = math.sin(lam), math.cos(lam)
        # rows are unit vectors of ENU expressed in ECEF
        self._r = (
            (-slam, clam, 0.0),
            (-sphi * clam, -sphi * slam, cphi),
            (cphi * clam, cphi * slam, sphi),
        )

    @classmethod
    def from_bbox(cls, bbox: BBox) -> "LocalFrame":
        lat, lon = bbox.center()
        return cls(lat, lon)

    def to_enu(self, lat: float, lon: float, alt: float = 0.0) -> tuple[float, float, float]:
        x, y, z = geodetic_to_ecef(lat, lon, alt)
        dx = x - self._origin[0]
        dy = y - self._origin[1]
        dz = z - self._origin[2]
        r = self._r
        e = r[0][0] * dx + r[0][1] * dy + r[0][2] * dz
        n = r[1][0] * dx + r[1][1] * dy + r[1][2] * dz
        u = r[2][0] * dx + r[2][1] * dy + r[2][2] * dz
        return e, n, u

    def to_geodetic(self, e: float, n: float, u: float = 0.0) -> tuple[float, float, float]:
        r = self._r
        dx = r[0][0] * e + r[1][0] * n + r[2][0] * u
        dy = r[0][1] * e + r[1][1] * n + r[2][1] * u
        dz = r[0][2] * e + r[1][2] * n + r[2][2] * u
        return ecef_to_geodetic(
            self._origin[0] + dx, self._origin[1] + dy, self._origin[2] + dz
        )


def sample_polyline_m(
    coords: list[tuple[float, float]], spacing_m: float
) -> list[tuple[float, float, float]]:
    """Densify a lon/lat polyline. Returns (lat, lon, heading_deg) samples."""
    if spacing_m <= 0:
        raise ValueError("spacing_m must be > 0")
    if len(coords) < 2:
        return []
    out: list[tuple[float, float, float]] = []
    leftover = 0.0
    for i in range(len(coords) - 1):
        lon1, lat1 = coords[i]
        lon2, lat2 = coords[i + 1]
        seg = haversine_m(lat1, lon1, lat2, lon2)
        if seg < 1e-3:
            continue
        hdg = bearing_deg(lat1, lon1, lat2, lon2)
        dist = leftover
        while dist <= seg:
            t = dist / seg
            lat = lat1 + t * (lat2 - lat1)
            lon = lon1 + t * (lon2 - lon1)
            out.append((lat, lon, hdg))
            dist += spacing_m
        leftover = dist - seg
    if not out:
        lon1, lat1 = coords[0]
        lon2, lat2 = coords[-1]
        out.append((lat1, lon1, bearing_deg(lat1, lon1, lat2, lon2)))
    return out


def nearest_point_on_segment(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> tuple[float, float, float]:
    """Project P onto AB. Returns (qx, qy, t in [0,1])."""
    abx, aby = bx - ax, by - ay
    denom = abx * abx + aby * aby
    if denom < 1e-12:
        return ax, ay, 0.0
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / denom))
    return ax + t * abx, ay + t * aby, t


def snap_to_polylines_enu(
    x: float,
    y: float,
    lines: list[list[tuple[float, float]]],
) -> tuple[float, float, float, float]:
    """Snap (x,y) to the closest ENU polyline.

    Returns (sx, sy, heading_deg, distance_m).
    """
    best = (x, y, 0.0, float("inf"))
    for line in lines:
        for i in range(len(line) - 1):
            ax, ay = line[i]
            bx, by = line[i + 1]
            qx, qy, _ = nearest_point_on_segment(x, y, ax, ay, bx, by)
            d = math.hypot(qx - x, qy - y)
            if d < best[3]:
                hdg = wrap_heading(math.degrees(math.atan2(bx - ax, by - ay)))
                best = (qx, qy, hdg, d)
    return best


def camera_rotation_cv(heading_deg: float, pitch_deg: float = 0.0) -> list[list[float]]:
    """World-from-camera rotation, OpenCV convention (X right, Y down, Z fwd).

    World is ENU (X east, Y north, Z up). Street View pitch is 0 at the
    horizon and positive looking up.
    """
    h = math.radians(heading_deg)
    p = math.radians(pitch_deg)
    sh, ch = math.sin(h), math.cos(h)
    sp, cp = math.sin(p), math.cos(p)
    # camera Z (forward) in ENU
    fx, fy, fz = sh * cp, ch * cp, sp
    # camera X (right): heading + 90°, horizon
    rx, ry, rz = ch, -sh, 0.0
    # camera Y (down) = forward × right
    dx = fy * rz - fz * ry
    dy = fz * rx - fx * rz
    dz = fx * ry - fy * rx
    # columns are camera axes in world
    return [
        [rx, dx, fx],
        [ry, dy, fy],
        [rz, dz, fz],
    ]


def mat_vec(m: list[list[float]], v: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    )


def transpose3(m: list[list[float]]) -> list[list[float]]:
    return [[m[j][i] for j in range(3)] for i in range(3)]


@lru_cache(maxsize=1)
def _rd_transformers():
    from pyproj import Transformer

    # EPSG:28992 is RD New. 7415 is RD+NAP; XY is the same.
    # Kadaster polynomials drift tens to hundreds of metres away from
    # Amersfoort; PROJ stays centimetre-level in Groningen/Haarlem.
    to_wgs = Transformer.from_crs(28992, 4326, always_xy=True)
    to_rd = Transformer.from_crs(4326, 28992, always_xy=True)
    return to_wgs, to_rd


def rd_to_wgs84(x: float, y: float) -> tuple[float, float]:
    """EPSG:28992 / 7415 (RD New) → WGS84."""
    to_wgs, _ = _rd_transformers()
    lon, lat = to_wgs.transform(x, y)
    return float(lat), float(lon)


def wgs84_to_rd(lat: float, lon: float) -> tuple[float, float]:
    """WGS84 → RD New metres."""
    _, to_rd = _rd_transformers()
    x, y = to_rd.transform(lon, lat)
    return float(x), float(y)
