import numpy as np

from earthprosp import lithosphere as L
from earthprosp.rasters import GeoGrid


def test_contour_points_straddle_level():
    lat = np.arange(-10, 10.5, 1.0)
    lon = np.arange(-180, 180, 1.0)
    g = GeoGrid(lat, lon, np.tile(lon[None, :], (len(lat), 1)))  # value = longitude
    pts = L.contour_points(g, 0.5)
    assert len(pts) > 0
    assert np.allclose(pts[:, 1][np.abs(pts[:, 1]) < 90], 0.5)


def test_crust_groups_are_disjoint_and_cover_all_types():
    ids = [i for v in L.CRUST_GROUPS.values() for i in v]
    assert len(ids) == len(set(ids)) == 36
