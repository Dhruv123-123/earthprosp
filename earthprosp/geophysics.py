"""Potential-field covariates from global gravity grids.

Input: the EIGEN-6C4 gravity disturbance and ETOPO1 topography at 10 arc-minutes
(fatiando-data, CC-BY), both on the same -180..180 / -90..90 grid. From them:

* **Bouguer anomaly**: disturbance minus a simple-slab correction (2670 kg/m^3 on land,
  water replaced by crust offshore). The Bouguer field maps density contrasts in the
  crust: mafic belts, granites, basins.
* **Regional / residual separation** with Gaussian low-pass filters (sigma 50 and 200 km).
  The residual isolates upper-crustal sources at district-to-belt scale.
* **Horizontal gradient magnitude (HGM)** of the Bouguer field and of its residual. Ridges
  of HGM mark steep density contrasts: faults, intrusive contacts, terrane boundaries,
  the "worms" of mineral-systems analysis (Hobbs 2000; Hornby et al. 1999).
* **First vertical derivative and tilt angle** (Miller & Singh 1994), computed with FFT
  in latitude bands so the east-west grid spacing is near-constant within each band.
  Tilt normalises amplitude, so weak and strong sources outline equally.

Everything is returned as GeoGrids on the input grid; `cell_features` samples them at
H3 cells. All filters work in kilometres, not degrees.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .rasters import GeoGrid, cell_values

G = 6.674e-11
RHO_CRUST = 2670.0
RHO_WATER = 1030.0
EARTH_KM_PER_DEG = 111.195


def normal_gravity(lat_deg: np.ndarray, height_m: np.ndarray | float) -> np.ndarray:
    """WGS84 normal gravity (mGal) at geodetic latitude and ellipsoidal height.

    Somigliana's closed form on the ellipsoid plus the second-order free-air height
    correction (Hofmann-Wellenhof & Moritz 2006, eq. 2-215)."""
    a, f, m = 6378137.0, 1 / 298.257223563, 0.00344978650684
    s2 = np.sin(np.radians(lat_deg)) ** 2
    g0 = 978032.53359 * (1 + 0.00193185265241 * s2) / np.sqrt(1 - 0.00669437999013 * s2)
    h = np.asarray(height_m, np.float64)
    return g0 * (1 - 2 / a * (1 + f + m - 2 * f * s2) * h + 3 * h ** 2 / a ** 2)


def disturbance_from_gravity(gravity: GeoGrid, height_m) -> GeoGrid:
    """Gravity disturbance (mGal) = observed gravity - normal gravity at the same point."""
    h = np.broadcast_to(np.asarray(height_m, np.float64), gravity.values.shape)
    lat = np.broadcast_to(gravity.lat[:, None], gravity.values.shape)
    return gravity.with_values(gravity.values - normal_gravity(lat, h), "disturbance")


def bouguer(disturbance: GeoGrid, topography: GeoGrid) -> GeoGrid:
    """Simple Bouguer anomaly (mGal) from a gravity disturbance and topography/bathymetry (m)."""
    h = topography.values
    rho = np.where(h >= 0, RHO_CRUST, RHO_CRUST - RHO_WATER)
    slab = 2 * np.pi * G * rho * h * 1e5  # m/s^2 -> mGal
    return disturbance.with_values(disturbance.values - slab, "bouguer")


def _sigma_px(grid: GeoGrid, sigma_km: float) -> tuple[np.ndarray, float]:
    """Gaussian sigma in pixels: constant in latitude, latitude-dependent in longitude."""
    sy = sigma_km / (EARTH_KM_PER_DEG * grid.dlat)
    sx = sigma_km / (EARTH_KM_PER_DEG * grid.dlon * np.maximum(np.cos(np.radians(grid.lat)), 0.05))
    return sx, sy


def gaussian_lowpass(grid: GeoGrid, sigma_km: float) -> GeoGrid:
    """Gaussian smoothing with a sigma fixed in km (wider in degrees toward the poles).

    Latitude smoothing is one separable pass; longitude smoothing is done row by row with
    that row's sigma and periodic wrap, so the kernel is isotropic on the ground."""
    v = np.nan_to_num(grid.values, nan=float(np.nanmean(grid.values)))
    sx, sy = _sigma_px(grid, sigma_km)
    v = ndimage.gaussian_filter1d(v, sy, axis=0, mode="nearest")
    out = np.empty_like(v)
    for i in range(v.shape[0]):
        s = min(sx[i], v.shape[1] / 4)
        out[i] = ndimage.gaussian_filter1d(v[i], s, mode="wrap")
    return grid.with_values(out)


def gradients(grid: GeoGrid) -> tuple[np.ndarray, np.ndarray]:
    """d/dx (east) and d/dy (north) in units per km."""
    v = grid.values
    dy = np.gradient(v, axis=0) / (EARTH_KM_PER_DEG * grid.dlat)
    dx_px = (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / 2
    dx = dx_px / (EARTH_KM_PER_DEG * grid.dlon * np.maximum(np.cos(np.radians(grid.lat)), 0.05))[:, None]
    return dx, dy


def hgm(grid: GeoGrid) -> GeoGrid:
    dx, dy = gradients(grid)
    return grid.with_values(np.hypot(dx, dy), f"hgm_{grid.name}")


def vertical_derivative(grid: GeoGrid, band_deg: float = 20.0, pad_deg: float = 10.0) -> GeoGrid:
    """First vertical derivative (units per km) by FFT, |k| multiplication, in latitude bands.

    Each band is padded by `pad_deg` on both sides (reflected at the poles), treated as
    a flat grid with the band-centre east-west spacing, transformed, and its core kept."""
    v = np.nan_to_num(grid.values, nan=float(np.nanmean(grid.values)))
    H, W = v.shape
    out = np.zeros_like(v)
    nb = int(round(band_deg / grid.dlat))
    npad = int(round(pad_deg / grid.dlat))
    vpad = np.pad(v, ((npad, npad), (0, 0)), mode="reflect")
    for s in range(0, H, nb):
        e = min(s + nb, H)
        block = vpad[s:e + 2 * npad]
        lat_c = grid.lat[(s + e) // 2]
        dx = EARTH_KM_PER_DEG * grid.dlon * max(np.cos(np.radians(lat_c)), 0.05)
        dy = EARTH_KM_PER_DEG * grid.dlat
        ky = np.fft.fftfreq(block.shape[0], d=dy) * 2 * np.pi
        kx = np.fft.fftfreq(block.shape[1], d=dx) * 2 * np.pi
        k = np.hypot(*np.meshgrid(ky, kx, indexing="ij"))
        f = np.fft.fft2(block - block.mean())
        out[s:e] = np.real(np.fft.ifft2(f * k))[npad:npad + (e - s)]
    return grid.with_values(out, f"vd_{grid.name}")


def tilt(grid: GeoGrid) -> GeoGrid:
    """Tilt angle atan(VD / HGM) in radians, in [-pi/2, pi/2]."""
    dz = vertical_derivative(grid).values
    h = hgm(grid).values
    return grid.with_values(np.arctan2(dz, h), f"tilt_{grid.name}")


def gravity_stack(disturbance: GeoGrid, topography: GeoGrid) -> dict[str, GeoGrid]:
    """All gravity-derived layers, keyed by covariate name."""
    bg = bouguer(disturbance, topography)
    reg200 = gaussian_lowpass(bg, 200.0)
    reg50 = gaussian_lowpass(bg, 50.0)
    resid = bg.with_values(bg.values - reg200.values, "bouguer_resid200")
    band = bg.with_values(reg50.values - reg200.values, "bouguer_band50_200")
    hg = hgm(reg50)
    hg_res = hgm(resid)
    out = {
        "grav_bouguer": bg,
        "grav_bouguer_reg200": reg200,
        "grav_bouguer_resid200": resid,
        "grav_bouguer_band50_200": band,
        "grav_hgm": hg,
        "grav_hgm_resid": hg_res,
        "grav_tilt": tilt(resid),
        "grav_vd": vertical_derivative(resid),
        "grav_free_air": disturbance,
    }
    # Local maxima of HGM ~ "worms": distance-free proxy = HGM relative to its 100 km mean.
    hg_mean = gaussian_lowpass(hg, 100.0)
    out["grav_hgm_ridge"] = hg.with_values(hg.values / np.maximum(hg_mean.values, 1e-6))
    return out


def topography_stack(topography: GeoGrid) -> dict[str, GeoGrid]:
    """Long-wavelength topography: regional elevation and relief (complements the DEM)."""
    t = topography
    reg = gaussian_lowpass(t, 100.0)
    return {
        "topo_etopo": t,
        "topo_reg100": reg,
        "topo_resid100": t.with_values(t.values - reg.values),
        "topo_hgm": hgm(gaussian_lowpass(t, 25.0)),
    }


def cell_features(layers: dict[str, GeoGrid], stencil) -> dict[str, np.ndarray]:
    return {k: cell_values(g, stencil=stencil)["mean"].astype(np.float32) for k, g in layers.items()}
