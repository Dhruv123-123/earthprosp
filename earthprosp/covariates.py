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
  litho       LITHO1.0 LAB / Moho / sediment thickness, LAB gradient and signed distance to the
              100 and 170 km LAB contours; CRUST1.0 crust type (Archean ... arc ... rift)
  volcano     Holocene volcanoes: distance and counts within 100 / 300 km
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


@family("litho")
def litho_family(cells, lat, lon, stencil, raw):
    import os
    from . import lithosphere
    path = f"{raw}/litho1/litho_data.npz"
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return lithosphere.features(lat, lon, path, f"{raw}/litho1/Crust1pt0-regionalisation.npz")


@family("volcano")
def volcano_family(cells, lat, lon, stencil, raw):
    """Holocene volcanoes (NOAA/Smithsonian list): proximity to active magmatic arcs,
    the heat engine of porphyry and epithermal systems."""
    v = pd.read_csv(f"{raw}/github/volcano_db.csv", encoding="latin1")
    v = v.dropna(subset=["Latitude", "Longitude"])
    return vectors.point_features("volcano", v.Latitude.to_numpy(), v.Longitude.to_numpy(), lat, lon,
                                  radii_km=(100, 300))


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
        done = [c for c in out.columns if c.startswith(_prefix(fam))]  # str.startswith takes tuples
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


@family("magnetics")
def magnetics_family(cells, lat, lon, stencil, raw):
    """Crustal total-field anomaly synthesised from NOAA EMM2017 (degree 790, EMAG2-based):
    magnetite-bearing intrusions, greenstone belts, BIF and the structures that offset them."""
    from . import geophysics as gp
    from . import magnetics
    tmi = magnetics.total_field_anomaly(f"{raw}/magnetics/emm2017.wmm.cof")
    return gp.cell_features(magnetics.magnetic_stack(tmi), stencil)


@family("heatflow")
def heatflow_family(cells, lat, lon, stencil, raw):
    """IHFC Global Heat Flow Database 2023, continental sites: Gaussian kernel mean of
    surface heat flow at 150 and 400 km. Where no site is within reach the value is NaN
    (the trees treat it as missing), rather than an extrapolated guess. Site density is
    deliberately not a feature: boreholes cluster where people explore."""
    d = pd.read_csv(f"{raw}/heatflow/IHFC_2023_GHFDB.csv", sep=";", encoding="latin1", low_memory=False,
                    usecols=["q", "lat", "lng", "A_dom"])
    q = pd.to_numeric(d.q, errors="coerce")
    d = d[(d.A_dom == "continental") & q.between(1, 250)].assign(q=q)
    plat, plon = d.lat.to_numpy(float), d.lng.to_numpy(float)
    out = {}
    for sigma2 in (150, 400):  # density_grid smooths with sigma = radius / 2
        r = 2 * sigma2
        num = vectors.density_grid(plat, plon, 1.0, r, res_deg=0.25)
        H = vectors.density_grid(plat, plon, 1.0, r, res_deg=0.25, weights=d.q.to_numpy(float))
        den = num.sample(lat, lon)
        val = H.sample(lat, lon) / np.where(den > 0, den, np.nan)
        # Support: expected number of sites within ~sigma; below 0.5 the mean is unreliable.
        support = den * np.pi * sigma2 ** 2
        out[f"heatflow_q_{sigma2}km"] = np.where(support >= 0.5, val, np.nan).astype(np.float32)
    return out


def _prefix(fam: str) -> str:
    return {"gravity": "grav_", "topo": "topo_", "faults": "fault_", "plates": "plate_",
            "litho": ("litho_", "crust_"), "magnetics": "mag_"}.get(fam, fam + "_")


def family_of(column: str) -> str:
    for fam in FAMILIES:
        if column.startswith(_prefix(fam)):
            return fam
    return column.split("_")[0]
