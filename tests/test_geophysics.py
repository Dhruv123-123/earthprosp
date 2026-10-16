import numpy as np

from earthprosp import geophysics as gp
from earthprosp.rasters import GeoGrid, cell_stencil, cell_values


def _grid(fn, step=0.5):
    lat = np.arange(-90, 90 + step / 2, step)
    lon = np.arange(-180, 180, step)
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    return GeoGrid(lat, lon, fn(LA, LO))


def test_normal_gravity_reference_values():
    # WGS84 normal gravity on the ellipsoid: equator and pole.
    np.testing.assert_allclose(gp.normal_gravity(0.0, 0.0), 978032.53359, atol=1e-3)
    np.testing.assert_allclose(gp.normal_gravity(90.0, 0.0), 983218.49378, atol=1e-2)
    # Free-air gradient ~0.3086 mGal/m near the surface.
    d = gp.normal_gravity(45.0, 0.0) - gp.normal_gravity(45.0, 100.0)
    assert 30.5 < d < 31.0


def test_sample_wraps_antimeridian():
    g = _grid(lambda la, lo: np.cos(np.radians(lo)))
    v = g.sample(np.array([0.0, 0.0]), np.array([179.9, -179.9]))
    np.testing.assert_allclose(v, -1.0, atol=1e-3)


def test_lowpass_preserves_constant_and_smooths_noise():
    rng = np.random.default_rng(0)
    g = _grid(lambda la, lo: 5 + rng.normal(size=la.shape))
    s = gp.gaussian_lowpass(g, 200.0)
    mid = np.abs(g.lat) < 60
    assert abs(s.values[mid].mean() - 5) < 0.05
    assert s.values[mid].std() < 0.2 * g.values[mid].std()


def test_vertical_derivative_of_plane_wave():
    # For f = cos(k x), the vertical derivative magnitude is |k| f.
    wl_km = 1000.0
    k = 2 * np.pi / wl_km
    g = _grid(lambda la, lo: np.cos(k * lo * gp.EARTH_KM_PER_DEG), step=0.25)
    vd = gp.vertical_derivative(g).values
    row = np.argmin(np.abs(g.lat))  # equator: x spacing is exact there
    np.testing.assert_allclose(np.abs(vd[row]).max(), k, rtol=0.05)


def test_hgm_of_linear_ramp():
    g = _grid(lambda la, lo: la * gp.EARTH_KM_PER_DEG * 2.0)  # 2 units per km northward
    h = gp.hgm(g).values
    np.testing.assert_allclose(np.median(h[np.abs(g.lat) < 80]), 2.0, rtol=1e-6)


def test_cell_values_on_constant_grid():
    import h3
    g = _grid(lambda la, lo: np.full(la.shape, 3.0))
    cells = [h3.latlng_to_cell(10, 20, 6), h3.latlng_to_cell(-40, 179.99, 6)]
    out = cell_values(g, stencil=cell_stencil(cells), reducers=("mean", "std"))
    np.testing.assert_allclose(out["mean"], 3.0)
    np.testing.assert_allclose(out["std"], 0.0, atol=1e-12)
