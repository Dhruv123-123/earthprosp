"""Global lat/lon rasters: load, sample at points, and summarise onto H3 cells.

Most global geoscience grids (gravity, topography, crustal models, heat flow) ship as
regular geographic grids. `GeoGrid` wraps one such grid with bilinear sampling that
handles the antimeridian wrap, and `cell_values` reduces a grid onto H3 cells by
averaging a small stencil of points per cell (centre + 6 points halfway to the
vertices), which is enough when the grid is coarser than or similar to the cell.
"""

from __future__ import annotations

from dataclasses import dataclass

import h3
import numpy as np


@dataclass
class GeoGrid:
    lat: np.ndarray     # (H,) ascending
    lon: np.ndarray     # (W,) ascending, spanning ~360 degrees
    values: np.ndarray  # (H, W)
    name: str = ""

    def __post_init__(self):
        self.lat = np.asarray(self.lat, np.float64)
        self.lon = np.asarray(self.lon, np.float64)
        v = np.asarray(self.values, np.float64)
        if self.lat[0] > self.lat[-1]:
            self.lat, v = self.lat[::-1], v[::-1]
        if self.lon[0] > self.lon[-1]:
            self.lon, v = self.lon[::-1], v[:, ::-1]
        self.values = v

    @property
    def dlat(self) -> float:
        return float(self.lat[1] - self.lat[0])

    @property
    def dlon(self) -> float:
        return float(self.lon[1] - self.lon[0])

    def sample(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Bilinear interpolation; longitudes wrap on global grids, latitudes clamp."""
        lat = np.asarray(lat, np.float64)
        lon = np.asarray(lon, np.float64)
        v = self.values
        H, W = v.shape
        if abs(self.lon[-1] - self.lon[0] - 360) < 1e-6:
            W -= 1  # last column repeats the first (-180..180 inclusive)
        periodic = abs(W * self.dlon - 360) < 1e-6
        fy = np.clip((lat - self.lat[0]) / self.dlat, 0, H - 1)
        if periodic:
            fx = np.mod((lon - self.lon[0]) % 360 / self.dlon, W)
        else:
            fx = np.clip((lon - self.lon[0]) / self.dlon, 0, W - 1)
        y0 = np.minimum(np.floor(fy).astype(int), H - 1)
        x0 = np.minimum(np.floor(fx).astype(int), W - 1)
        y1 = np.minimum(y0 + 1, H - 1)
        x1 = (x0 + 1) % W if periodic else np.minimum(x0 + 1, W - 1)
        wy, wx = fy - y0, fx - x0
        return ((1 - wy) * (1 - wx) * v[y0, x0] + (1 - wy) * wx * v[y0, x1]
                + wy * (1 - wx) * v[y1, x0] + wy * wx * v[y1, x1])

    def with_values(self, values: np.ndarray, name: str = "") -> "GeoGrid":
        return GeoGrid(self.lat, self.lon, values, name or self.name)


def cell_stencil(cells, halfway: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """(N, 7) lat/lon points per cell: the centre and 6 points halfway to each vertex."""
    n = len(cells)
    lat = np.zeros((n, 7))
    lon = np.zeros((n, 7))
    for i, c in enumerate(cells):
        a, b = h3.cell_to_latlng(c)
        lat[i, 0], lon[i, 0] = a, b
        bd = h3.cell_to_boundary(c)
        for j in range(6):
            va, vb = bd[j % len(bd)]
            dlon = ((vb - b + 180) % 360) - 180
            f = 0.5 if halfway else 1.0
            lat[i, j + 1] = a + f * (va - a)
            lon[i, j + 1] = b + f * dlon
    return lat, lon


def cell_values(grid: GeoGrid, cells=None, stencil=None, reducers=("mean",)) -> dict[str, np.ndarray]:
    """Reduce a grid onto cells. Pass a precomputed `stencil` to avoid recomputing it."""
    lat, lon = stencil if stencil is not None else cell_stencil(cells)
    v = grid.sample(lat.ravel(), lon.ravel()).reshape(lat.shape)
    out = {}
    for r in reducers:
        if r == "mean":
            out[r] = np.nanmean(v, 1)
        elif r == "std":
            out[r] = np.nanstd(v, 1)
        elif r == "range":
            out[r] = np.nanmax(v, 1) - np.nanmin(v, 1)
        elif r == "centre":
            out[r] = v[:, 0]
        else:
            raise ValueError(r)
    return out


def from_netcdf(path: str, var: str, lat_name: str = "latitude", lon_name: str = "longitude",
                with_coords: tuple[str, ...] = ()) -> GeoGrid | tuple:
    """Load one variable; `with_coords` also returns extra 2-D coordinates (e.g. height)."""
    import xarray as xr

    ds = xr.open_dataset(path)
    g = GeoGrid(ds[lat_name].values, ds[lon_name].values, ds[var].values, var)
    if not with_coords:
        return g
    extra = []
    for c in with_coords:
        v = ds[c].values
        if ds[lat_name].values[0] > ds[lat_name].values[-1]:
            v = v[::-1]
        extra.append(v)
    return (g, *extra)
