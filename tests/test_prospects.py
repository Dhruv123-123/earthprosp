import numpy as np

from earthprosp import prospects


def test_nms_separates_and_orders():
    lat = np.array([0.0, 0.01, 0.5, 0.51, 1.0])
    lon = np.zeros(5)
    score = np.array([0.9, 0.95, 0.5, 0.8, 0.1])
    k = prospects.nms(lat, lon, score, min_km=5.0, n=10)
    assert list(k) == [1, 3, 4]
    d = prospects.haversine_km(lat[k][:, None], lon[k][:, None], lat[k][None], lon[k][None])
    assert (d[~np.eye(len(k), dtype=bool)] >= 5).all()


def test_nms_limit_and_footprint():
    lat = np.linspace(0, 1, 50)
    lon = np.zeros(50)
    k = prospects.nms(lat, lon, -lat, min_km=0.1, n=3)
    assert list(k) == [0, 1, 2]
    f = prospects.footprint(lat, lon, [0], km=5.0)
    assert f[0] == 3  # 0, 2.27 and 4.54 km
