"""Subduction geometry, seismicity and tectonic provinces.

Porphyry copper, epithermal gold and many skarns form above a subducting slab where it
reaches ~100 km depth and the mantle wedge melts; the plate-boundary distances already in
the feature set know where trenches are, but not how far behind the trench the arc sits,
which depends on slab dip. Three public datasets fill that in:

* Slab2 (Hayes et al. 2018, USGS): slab-top depth grids for 27 subduction zones at 0.05 deg.
* Earthquake hypocentres: ISC-GEM (1904-2018, M >= ~5.5) and the USGS ComCat M4.5+
  catalogue (2000-2024). Intermediate-depth (70-300 km) seismicity outlines slabs where
  Slab2 has no model, and shallow crustal seismicity marks active deformation.
* World geologic provinces (USGS world provinces, rasterised in Keller's StatGeochem):
  accreted arc, island arc, continental arc, collisional orogen, extensional, rift, plume,
  shield, platform, basin.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

from . import vectors
from .rasters import GeoGrid

PROVINCES = {10: "accreted_arc", 11: "island_arc", 12: "continental_arc", 13: "collisional_orogen",
             20: "extensional", 21: "rift", 22: "plume", 31: "shield", 32: "platform", 33: "basin"}


def read_catalog(seis_dir: str) -> pd.DataFrame:
    """lat, lon, depth_km, mag: ComCat M4.5+ from 2000, ISC-GEM before 2000 (no overlap)."""
    parts = []
    cc = (glob.glob(os.path.join(seis_dir, "comcat_m45_2000_2024.csv"))
          + glob.glob(os.path.join(seis_dir, "query_M4.5*2000-2024.csv")))[:1]
    if cc:
        cc = cc[0]
        d = pd.read_csv(cc, usecols=["time", "latitude", "longitude", "depth", "mag", "type"])
        d = d[d.type == "earthquake"]
        parts.append(pd.DataFrame({"lat": d.latitude, "lon": d.longitude, "depth_km": d.depth, "mag": d.mag,
                                   "year": pd.to_datetime(d.time, errors="coerce", utc=True).dt.year}))
    ig = os.path.join(seis_dir, "isc-gem-cat.csv")
    if os.path.exists(ig):
        d = pd.read_csv(ig, comment="#", header=None, skipinitialspace=True, usecols=[0, 1, 2, 7, 10])
        d.columns = ["date", "lat", "lon", "depth_km", "mag"]
        year = pd.to_datetime(d.date.str.strip(), errors="coerce").dt.year
        d = d.assign(year=year)
        parts.append(d[d.year < 2000 if parts else slice(None)].drop(columns="date"))
    if not parts:
        raise FileNotFoundError(f"no earthquake catalogue in {seis_dir}")
    out = pd.concat(parts, ignore_index=True)
    for c in ("lat", "lon", "depth_km", "mag"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.dropna(subset=["lat", "lon", "depth_km"]).reset_index(drop=True)


def seismicity_features(cat: pd.DataFrame, lat, lon) -> dict[str, np.ndarray]:
    out = {}
    bands = {"shallow": (cat.depth_km <= 40, 100), "intermediate": ((cat.depth_km > 70) & (cat.depth_km <= 300), 200),
             "deep": (cat.depth_km > 300, 300)}
    for name, (m, r) in bands.items():
        p = cat[m]
        g = vectors.density_grid(p.lat.to_numpy(), p.lon.to_numpy(), 1.0, r, res_deg=0.25)
        # events per 10,000 km^2, log-scaled: the raw density spans five orders of magnitude
        out[f"seis_{name}_logdensity_{r}km"] = np.log1p(np.maximum(g.sample(lat, lon), 0) * 1e4).astype(np.float32)
    inter = cat[bands["intermediate"][0]]
    d = vectors.PointIndex(inter.lat.to_numpy(), inter.lon.to_numpy()).distance_km(lat, lon)
    out["seis_intermediate_logdist"] = np.log1p(np.minimum(d, 3000)).astype(np.float32)
    # Kernel mean hypocentre depth of sub-crustal events (> 40 km) within ~100 km: a slab
    # depth estimate where Slab2 has no model. NaN where fewer than ~1 event supports it.
    sub = cat[cat.depth_km > 40]
    den = vectors.density_grid(sub.lat.to_numpy(), sub.lon.to_numpy(), 1.0, 200, res_deg=0.25)
    num = vectors.density_grid(sub.lat.to_numpy(), sub.lon.to_numpy(), 1.0, 200, res_deg=0.25,
                               weights=sub.depth_km.to_numpy())
    dv = den.sample(lat, lon)
    val = num.sample(lat, lon) / np.where(dv > 0, dv, np.nan)
    out["seis_subcrustal_depth_km"] = np.where(dv * np.pi * 100 ** 2 >= 1, val, np.nan).astype(np.float32)
    return out


def slab_grid(slab_dir: str, res: float = 0.05) -> GeoGrid:
    """Mosaic of Slab2 depth grids (km, positive down; NaN where no slab). Where zones
    overlap the shallower slab wins."""
    files = sorted(glob.glob(os.path.join(slab_dir, "*_slab2_dep_*.xyz")))
    if not files:
        raise FileNotFoundError(f"no Slab2 depth grids in {slab_dir}")
    lat_c = np.round(np.arange(-90, 90 + res / 2, res), 4)
    lon_c = np.round(np.arange(-180, 180, res), 4)
    D = np.full((len(lat_c), len(lon_c)), np.nan, np.float32)
    for f in files:
        x = pd.read_csv(f, header=None, names=["lon", "lat", "dep"]).dropna()
        lo = (x.lon.to_numpy() + 180) % 360 - 180
        i = np.clip(np.rint((x.lat.to_numpy() + 90) / res).astype(int), 0, len(lat_c) - 1)
        j = np.rint((lo + 180) / res).astype(int) % len(lon_c)
        dep = -x.dep.to_numpy(np.float32)
        cur = D[i, j]
        D[i, j] = np.where(np.isnan(cur) | (dep < cur), dep, cur)
    return GeoGrid(lat_c, lon_c, D, "slab_depth_km")


def slab_features(slab: GeoGrid, lat, lon) -> dict[str, np.ndarray]:
    D = slab.values
    ii, jj = np.nonzero(np.isfinite(D))
    plat, plon, dep = slab.lat[ii], slab.lon[jj], D[ii, jj]
    # Nearest-neighbour depth instead of bilinear: most zones are gridded at 0.1 deg, so the
    # 0.05 deg mosaic has holes, and edges would be lost to NaN mixing.
    idx = vectors.PointIndex(plat, plon)
    dist, nn = idx.tree.query(np.radians(np.stack([lat, lon], 1)), k=1)
    dist = dist[:, 0] * vectors.R_EARTH_KM
    depth_nn = dep[nn[:, 0]]
    out = {"slab_depth_km": np.where(dist < 15, depth_nn, np.nan).astype(np.float32),
           "slab_dist_km": np.minimum(dist, 3000).astype(np.float32)}
    # Arc front: distance to where the slab top is 90-150 km deep, signed negative on the
    # trench side (slab shallower than 90 km beneath the cell).
    band = (dep >= 90) & (dep <= 150)
    d_arc = vectors.PointIndex(plat[band], plon[band]).distance_km(lat, lon)
    fore = np.isfinite(out["slab_depth_km"]) & (out["slab_depth_km"] < 90)
    out["slab_arc_signed_km"] = np.where(fore, -d_arc, d_arc).clip(-3000, 3000).astype(np.float32)
    # Depth to the slab at the nearest slab point within 300 km (the slab the cell sits behind).
    out["slab_nearest_depth_km"] = np.where(dist < 300, depth_nn, np.nan).astype(np.float32)
    return out


def province_grid(path: str) -> tuple[GeoGrid, np.ndarray]:
    """USGS world geologic provinces raster from StatGeochem: 0.1 deg latitude (90 to -90) by
    1/6 deg longitude (-180 to 180). Returns a template grid and the integer code array."""
    import h5py
    codes = h5py.File(path, "r")["geolprov"][()].T  # HDF5 written column-major by Julia
    H, W = codes.shape
    lat = np.linspace(90, -90, H)
    lon = np.linspace(-180, 180, W)
    return GeoGrid(lat, lon, np.zeros((H, W)), "prov"), codes[::-1]  # GeoGrid flips lat ascending


def province_features(path: str, lat, lon, sigma_km: float = 25.0) -> dict[str, np.ndarray]:
    from .geophysics import gaussian_lowpass
    tmpl, codes = province_grid(path)
    out = {}
    for code, name in PROVINCES.items():
        g = gaussian_lowpass(tmpl.with_values((codes == code).astype(np.float64)), sigma_km)
        out[f"prov_{name}"] = np.clip(g.sample(lat, lon), 0, 1).astype(np.float32)
    return out
