"""Assemble per-cell geoscience covariates from public global datasets.

Each *family* is a function `(cells, lat, lon, stencil, raw_dir) -> dict[name, array]`
registered in FAMILIES. `build` runs the requested families, skips any whose raw data
is missing (with a message), and caches the table as parquet keyed by H3 id, so adding
a family later does not recompute the others.

Families and sources (all public; see earthprosp/sources.py for URLs and licences):
  gravity     EIGEN-6C4 gravity + ETOPO1 (fatiando-data, 10'): Bouguer, residuals, HGM, tilt
  topo        ETOPO1 regional elevation / relief / slope (10')
  faults      GEM Global Active Faults (Styron & Pagani 2020): distance + density by kinematics
  plates      Bird (2003) PB2002 boundaries: distance to subduction, collision, rift, transform;
              convergence rate at the nearest convergent step; inside orogen
Further families (terrain from Copernicus DEM, land cover from ESA WorldCover, magnetics,
crust and lithosphere models) plug in the same way.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from . import grid, rasters, vectors

FAMILIES: dict[str, callable] = {}


def family(name):
    def deco(fn):
        FAMILIES[name] = fn
        return fn
    return deco


def _log(msg):
    print(f"[covariates {time.strftime('%H:%M:%S')}] {msg}", flush=True)


@family("gravity")
def gravity_family(cells, lat, lon, stencil, raw):
    from . import geophysics as gp
    g, h = rasters.from_netcdf(f"{raw}/fatiando/earth-gravity-10arcmin.nc", "gravity", with_coords=("height",))
    topo = rasters.from_netcdf(f"{raw}/fatiando/earth-topography-10arcmin.nc", "topography")
    layers = gp.gravity_stack(gp.disturbance_from_gravity(g, h), topo)
    out = gp.cell_features(layers, stencil)
    # Within-cell variability of the residual: a crude texture measure at 18 km grid spacing.
    out["grav_resid_cell_range"] = rasters.cell_values(layers["grav_bouguer_resid200"], stencil=stencil,
                                                       reducers=("range",))["range"].astype(np.float32)
    return out


@family("topo")
def topo_family(cells, lat, lon, stencil, raw):
    from . import geophysics as gp
    topo = rasters.from_netcdf(f"{raw}/fatiando/earth-topography-10arcmin.nc", "topography")
    return gp.cell_features(gp.topography_stack(topo), stencil)


@family("faults")
def faults_family(cells, lat, lon, stencil, raw):
    import geopandas as gpd
    path = f"{raw}/wri/faults/gem_active_faults_harmonized.shp"
    f = gpd.read_file(path)
    st = f["slip_type"].fillna("").str.lower()
    groups = {
        "fault_all": np.ones(len(f), bool),
        "fault_normal": st.str.contains("normal").to_numpy(),
        "fault_reverse": (st.str.contains("reverse") | st.str.contains("thrust")).to_numpy(),
        "fault_strikeslip": (st.str.contains("dextral") | st.str.contains("sinistral")
                             | st.str.contains("strike")).to_numpy(),
    }
    out = {}
    for name, m in groups.items():
        out.update(vectors.line_features(name, f.geometry[m], lat, lon,
                                         radii_km=(25, 100) if name == "fault_all" else (50,)))
    return out


@family("plates")
def plates_family(cells, lat, lon, stencil, raw):
    import geopandas as gpd
    import shapely
    s = gpd.read_file(f"{raw}/plates/PB2002_steps.json")
    cls = s["STEPCLASS"].fillna("")
    groups = {
        "plate_subduction": cls.isin(["SUB"]).to_numpy(),
        "plate_collision": cls.isin(["CCB", "OCB"]).to_numpy(),
        "plate_rift": cls.isin(["CRB", "OSR"]).to_numpy(),
        "plate_transform": cls.isin(["CTF", "OTF"]).to_numpy(),
    }
    out = {}
    for name, m in groups.items():
        out.update(vectors.line_features(name, s.geometry[m], lat, lon, radii_km=(), cap_km=3000))
    # Convergence rate (mm/yr) at the nearest subduction/collision step: fast subduction
    # feeds arc magmatism (porphyry, epithermal).
    conv = (groups["plate_subduction"] | groups["plate_collision"])
    mid = np.array([[g.interpolate(0.5, normalized=True).y, g.interpolate(0.5, normalized=True).x]
                    for g in s.geometry[conv]])
    idx = vectors.PointIndex(mid[:, 0], mid[:, 1])
    _, nn = idx.tree.query(np.radians(np.stack([lat, lon], 1)), k=1)
    vel = pd.to_numeric(s.loc[conv, "VELOCITYDI"], errors="coerce").abs().to_numpy()
    out["plate_conv_rate_nearest"] = vel[nn[:, 0]].astype(np.float32)
    o = gpd.read_file(f"{raw}/plates/PB2002_orogens.json")
    geom = shapely.union_all(o.geometry.to_numpy())
    out["plate_in_orogen"] = shapely.contains_xy(geom, lon, lat).astype(np.float32)
    return out


def build(cells, families=None, raw_dir: str = "data/raw", cache: str | None = "data/covariates.parquet",
          refresh: tuple = ()) -> pd.DataFrame:
    """Covariate table indexed by H3 cell. Cached per column: existing families are reused."""
    cells = np.asarray(cells, dtype=object)
    families = list(families or FAMILIES)
    have = None
    if cache and os.path.exists(cache):
        have = pd.read_parquet(cache)
        if len(have) != len(cells) or not (have.index.to_numpy() == cells).all():
            _log("cache cell set differs; rebuilding")
            have = None
    out = have if have is not None else pd.DataFrame(index=pd.Index(cells, name="h3"))
    lat, lon = grid.cell_centers(cells)
    stencil = None
    for fam in families:
        done = [c for c in out.columns if c.startswith(_prefix(fam))]
        if done and fam not in refresh:
            continue
        if stencil is None:
            _log(f"stencil for {len(cells):,} cells")
            stencil = rasters.cell_stencil(cells)
        t0 = time.time()
        try:
            cols = FAMILIES[fam](cells, lat, lon, stencil, raw_dir)
        except FileNotFoundError as exc:
            _log(f"{fam}: skipped, missing data ({exc})")
            continue
        for k, v in cols.items():
            out[k] = np.asarray(v, np.float32)
        _log(f"{fam}: {len(cols)} columns in {time.time() - t0:.0f}s")
        if cache:
            out.to_parquet(cache)
    return out


def _prefix(fam: str) -> str:
    return {"gravity": "grav_", "topo": "topo_", "faults": "fault_", "plates": "plate_"}.get(fam, fam + "_")


def family_of(column: str) -> str:
    for fam in FAMILIES:
        if column.startswith(_prefix(fam)):
            return fam
    return column.split("_")[0]
