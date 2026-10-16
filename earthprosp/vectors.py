"""Distance and density covariates from vector data (faults, plate boundaries, points).

Lines are densified to vertices every ~`step_km` and indexed in a haversine BallTree,
so "distance to the nearest fault" and "fault length within r km" are nearest-
neighbour / radius queries on points. At H3 res 6 (3.7 km edge) a 2 km step is
well below the cell size.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import BallTree

R_EARTH_KM = 6371.0


def densify_lines(geoms, step_km: float = 2.0) -> np.ndarray:
    """(n, 2) lat/lon vertices along every LineString / MultiLineString, ~step_km apart."""
    out = []
    for g in geoms:
        if g is None or g.is_empty:
            continue
        parts = getattr(g, "geoms", [g])
        for p in parts:
            xy = np.asarray(p.coords)[:, :2]
            if len(xy) < 2:
                out.append(xy[:, ::-1])
                continue
            for (x0, y0), (x1, y1) in zip(xy[:-1], xy[1:]):
                dx = ((x1 - x0 + 180) % 360) - 180  # shortest way across the antimeridian
                km = np.hypot(dx * 111.32 * np.cos(np.radians((y0 + y1) / 2)), (y1 - y0) * 110.57)
                n = max(int(np.ceil(km / step_km)), 1)
                t = np.arange(n) / n
                lon = ((x0 + t * dx + 180) % 360) - 180
                out.append(np.stack([y0 + t * (y1 - y0), lon], 1))
            out.append(np.array([[xy[-1, 1], xy[-1, 0]]]))
    return np.concatenate(out) if out else np.empty((0, 2))


class PointIndex:
    """Haversine BallTree over lat/lon points (degrees)."""

    def __init__(self, lat, lon):
        self.n = len(lat)
        self.tree = BallTree(np.radians(np.stack([lat, lon], 1)), metric="haversine")

    def distance_km(self, lat, lon, k: int = 1) -> np.ndarray:
        d, _ = self.tree.query(np.radians(np.stack([lat, lon], 1)), k=k)
        return (d[:, k - 1] * R_EARTH_KM).astype(np.float32)

    def count_within(self, lat, lon, radius_km: float, chunk: int = 200_000) -> np.ndarray:
        q = np.radians(np.stack([lat, lon], 1))
        out = np.zeros(len(q), np.int32)
        for s in range(0, len(q), chunk):
            out[s:s + chunk] = self.tree.query_radius(q[s:s + chunk], radius_km / R_EARTH_KM, count_only=True)
        return out


def line_features(prefix: str, geoms, lat, lon, radii_km=(25, 100), step_km: float = 2.0,
                  cap_km: float = 2000.0) -> dict[str, np.ndarray]:
    """Distance to nearest line (log-scaled) and line length density within each radius."""
    v = densify_lines(geoms, step_km)
    if len(v) == 0:
        return {}
    idx = PointIndex(v[:, 0], v[:, 1])
    d = np.minimum(idx.distance_km(lat, lon), cap_km)
    out = {f"{prefix}_dist_km": d, f"{prefix}_logdist": np.log1p(d).astype(np.float32)}
    for r in radii_km:
        length = idx.count_within(lat, lon, r) * step_km
        out[f"{prefix}_len_density_{r}km"] = (length / (np.pi * r ** 2) * 1000).astype(np.float32)  # m per km^2
    return out


def point_features(prefix: str, plat, plon, lat, lon, radii_km=(50,), cap_km: float = 3000.0,
                   weights=None) -> dict[str, np.ndarray]:
    idx = PointIndex(np.asarray(plat), np.asarray(plon))
    d = np.minimum(idx.distance_km(lat, lon), cap_km)
    out = {f"{prefix}_dist_km": d, f"{prefix}_logdist": np.log1p(d).astype(np.float32)}
    for r in radii_km:
        out[f"{prefix}_count_{r}km"] = idx.count_within(lat, lon, r).astype(np.float32)
    return out
