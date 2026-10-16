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


def density_grid(plat, plon, weight_per_point: float, radius_km: float, res_deg: float = 0.1):
    """Smoothed point density (per km^2) on a global grid: counts per pixel, divided by
    pixel area, Gaussian-smoothed with sigma = radius / 2 (in km, isotropic on the ground).

    A radius query per cell costs O(neighbours) and is far too slow for millions of cells
    and dense fault traces; a smoothed raster sampled at the cells is O(1) per cell."""
    from .geophysics import gaussian_lowpass
    from .rasters import GeoGrid
    lat_edges = np.arange(-90, 90 + res_deg / 2, res_deg)
    lon_edges = np.arange(-180, 180 + res_deg / 2, res_deg)
    H, _, _ = np.histogram2d(plat, plon, bins=[lat_edges, lon_edges])
    lat_c = (lat_edges[:-1] + lat_edges[1:]) / 2
    lon_c = (lon_edges[:-1] + lon_edges[1:]) / 2
    area = (res_deg * 110.57) * (res_deg * 111.32 * np.maximum(np.cos(np.radians(lat_c)), 0.01))
    dens = H * weight_per_point / area[:, None]
    return gaussian_lowpass(GeoGrid(lat_c, lon_c, dens), radius_km / 2)


def line_features(prefix: str, geoms, lat, lon, radii_km=(25, 100), step_km: float = 2.0,
                  cap_km: float = 2000.0) -> dict[str, np.ndarray]:
    """Distance to nearest line (capped, and log-scaled) and line length per area (m/km^2)
    within each radius (Gaussian-weighted)."""
    v = densify_lines(geoms, step_km)
    if len(v) == 0:
        return {}
    idx = PointIndex(v[:, 0], v[:, 1])
    d = np.minimum(idx.distance_km(lat, lon), cap_km)
    out = {f"{prefix}_dist_km": d, f"{prefix}_logdist": np.log1p(d).astype(np.float32)}
    for r in radii_km:
        g = density_grid(v[:, 0], v[:, 1], step_km * 1000.0, r)  # metres of trace per km^2
        out[f"{prefix}_len_density_{r}km"] = np.maximum(g.sample(lat, lon), 0).astype(np.float32)
    return out


def point_features(prefix: str, plat, plon, lat, lon, radii_km=(50,), cap_km: float = 3000.0,
                   weights=None) -> dict[str, np.ndarray]:
    idx = PointIndex(np.asarray(plat), np.asarray(plon))
    d = np.minimum(idx.distance_km(lat, lon), cap_km)
    out = {f"{prefix}_dist_km": d, f"{prefix}_logdist": np.log1p(d).astype(np.float32)}
    for r in radii_km:
        # Expected count within r (Gaussian-weighted density x disc area).
        g = density_grid(np.asarray(plat), np.asarray(plon), 1.0, r, res_deg=0.25)
        out[f"{prefix}_count_{r}km"] = np.maximum(g.sample(lat, lon) * np.pi * r ** 2, 0).astype(np.float32)
    return out
