"""Lithosphere and crust covariates from LITHO1.0 (Pasyanos et al. 2014) and the
CRUST1.0 crustal-type regionalisation (Laske et al. 2013).

Both ship inside the `litho1pt0` PyPI wheel (data/litho_data.npz: 19 layers x 9
properties on a 40,962-node icosahedral mesh; data/Crust1pt0-regionalisation.npz:
crust type on a 1-degree grid). We read the arrays directly; no stripy dependency.

Why these matter for ore:
* **LAB depth and its edges.** Sediment-hosted Zn-Pb and Cu (and many magmatic Ni-PGE
  deposits) cluster within ~200 km of the 170 km LAB contour, the edge of thick cratonic
  lithosphere (Hoggard et al. 2020, Nature Geoscience; Hoggard et al. 2022). We give
  the LAB depth, its horizontal gradient, and the signed distance to the 170 km contour.
* **Moho depth / crustal thickness** separate thickened orogens from thinned rifts.
* **Sediment thickness** says how deep basement is: ore under 5 km of cover is out of
  reach, and the surface embedding sees only the cover.
* **Crust type** is a free crustal-age map: Archean, Proterozoic, Phanerozoic, arcs,
  orogens, rifts, each with a distinctive metal endowment (Au in Archean greenstones,
  porphyries in young arcs).
"""

from __future__ import annotations

import numpy as np

from . import geophysics as gp
from . import vectors
from .rasters import GeoGrid

LAYER = {"LID-BOTTOM": 1, "LID-TOP": 2, "CRUST3-BOTTOM": 3, "CRUST1-TOP": 8,
         "SEDS3-BOTTOM": 9, "SEDS1-TOP": 14, "WATER-TOP": 16}
DEPTH, VS = 0, 3

# CRUST1.0 regionalisation index (see litho1pt0.c1_region_descriptor) -> coarse group.
CRUST_GROUPS = {
    "archean": [2, 3, 4],
    "proterozoic": [5, 6, 7, 8],
    "platform": [0, 1, 20, 21],
    "arc": [9, 10, 11, 12],
    "orogen": [15, 16, 17, 18, 19],
    "extended_rift": [13, 14, 22, 31],
    "phanerozoic": [23, 24],
    "oceanic_margin": [25, 26, 27, 28, 29, 30, 32, 33, 34, 35],
}


class Litho1:
    def __init__(self, path: str):
        z = np.load(path)
        self.data = z["litho1_all_data"]
        m = z["litho1_mesh_coords"]
        self.idx = vectors.PointIndex(m[:, 0], m[:, 2])

    def layer(self, name: str, lat, lon, quantity: int = DEPTH, k: int = 3) -> np.ndarray:
        """Inverse-distance interpolation of one layer property from the k nearest nodes."""
        d, j = self.idx.tree.query(np.radians(np.stack([lat, lon], 1)), k=k)
        w = 1.0 / np.maximum(d, 1e-9)
        w /= w.sum(1, keepdims=True)
        return (self.data[LAYER[name], quantity][j] * w).sum(1)


def lab_grid(l1: Litho1, step: float = 0.5) -> GeoGrid:
    lat = np.arange(-90, 90 + step / 2, step)
    lon = np.arange(-180, 180, step)
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    v = l1.layer("LID-BOTTOM", LA.ravel(), LO.ravel()).reshape(LA.shape) / 1000.0
    return GeoGrid(lat, lon, v, "lab_km")


def contour_points(g: GeoGrid, level: float) -> np.ndarray:
    """(n, 2) lat/lon of pixel pairs straddling `level` (a contour, pixel-accurate)."""
    v = g.values - level
    s = np.sign(v)
    pts = []
    ys, xs = np.nonzero(s[:, :-1] != s[:, 1:])
    pts.append(np.stack([g.lat[ys], (g.lon[xs] + g.lon[xs + 1]) / 2], 1))
    ys, xs = np.nonzero(s[:-1] != s[1:])
    pts.append(np.stack([(g.lat[ys] + g.lat[ys + 1]) / 2, g.lon[xs]], 1))
    return np.concatenate(pts)


def features(lat, lon, litho_path: str, regional_path: str) -> dict[str, np.ndarray]:
    l1 = Litho1(litho_path)
    surf = l1.layer("CRUST1-TOP", lat, lon)
    sed_top = l1.layer("SEDS1-TOP", lat, lon)
    moho = l1.layer("CRUST3-BOTTOM", lat, lon)
    lab = l1.layer("LID-BOTTOM", lat, lon)
    out = {
        "litho_lab_km": lab / 1000.0,
        "litho_moho_km": moho / 1000.0,
        "litho_crust_thick_km": (moho - np.minimum(sed_top, surf)) / 1000.0,
        "litho_sed_thick_km": np.maximum(surf - sed_top, 0) / 1000.0,
        "litho_mantle_lid_km": (lab - moho) / 1000.0,
        "litho_lid_vs": l1.layer("LID-TOP", lat, lon, quantity=VS),
    }
    g = lab_grid(l1)
    sm = gp.gaussian_lowpass(g, 100.0)
    out["litho_lab_grad"] = gp.hgm(sm).sample(lat, lon)  # km of LAB per km laterally
    for level in (100.0, 170.0):
        pts = contour_points(sm, level)
        d = vectors.PointIndex(pts[:, 0], pts[:, 1]).distance_km(lat, lon)
        sign = np.where(sm.sample(lat, lon) >= level, 1.0, -1.0)  # + on the thick side
        out[f"litho_lab{int(level)}_signed_km"] = np.clip(sign * d, -2000, 2000)
    # CRUST1.0 crust type
    reg = np.load(regional_path)["latlonDescriptor"]
    i = np.clip(((90.0 - np.asarray(lat)) // 1).astype(int), 0, 179)  # rows start at +90
    j = ((np.asarray(lon) + 180) // 1).astype(int) % 360  # columns start at -180
    t = reg[i, j]
    out["crust_type_code"] = t.astype(np.float32)
    for name, ids in CRUST_GROUPS.items():
        out[f"crust_{name}"] = np.isin(t, ids).astype(np.float32)
    return {k: np.asarray(v, np.float32) for k, v in out.items()}
