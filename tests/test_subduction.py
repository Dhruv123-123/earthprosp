import numpy as np
import pandas as pd

from earthprosp import subduction as S


def _slab(tmp_path):
    # A planar slab dipping east under 0..4 E, 0..2 N: depth = 30 km per degree.
    lon, lat = np.meshgrid(np.arange(0, 4.01, 0.1), np.arange(0, 2.01, 0.1))
    dep = -30.0 * lon
    pd.DataFrame({"lon": (lon.ravel() + 360) % 360, "lat": lat.ravel(), "dep": dep.ravel()}).to_csv(
        tmp_path / "tst_slab2_dep_01.01.18.xyz", header=False, index=False)
    return S.slab_grid(str(tmp_path))


def test_slab_depth_and_arc_side(tmp_path):
    g = _slab(tmp_path)
    f = S.slab_features(g, np.array([1.0, 1.0, 1.0, 1.0]), np.array([1.0, 3.5, 4.5, 10.0]))
    assert abs(f["slab_depth_km"][0] - 30) < 2 and abs(f["slab_depth_km"][1] - 105) < 2
    assert np.isnan(f["slab_depth_km"][3]) and f["slab_dist_km"][3] > 600
    # trench side of the 90-150 km band is negative, over the band ~0, behind it positive
    assert f["slab_arc_signed_km"][0] < -150 and abs(f["slab_arc_signed_km"][1]) < 15
    assert f["slab_arc_signed_km"][3] > 600


def test_read_catalog_merges_without_overlap(tmp_path):
    d = tmp_path / "seis"
    d.mkdir()
    pd.DataFrame({"time": ["2005-01-01T00:00:00Z", "2010-01-01T00:00:00Z"], "latitude": [1, 2],
                  "longitude": [3, 4], "depth": [10, 120], "mag": [5, 6], "type": ["earthquake"] * 2}
                 ).to_csv(d / "comcat_m45_2000_2024.csv", index=False)
    rows = ["# ISC-GEM", " 1990-01-01 00:00:00.00 , 5.0 , 6.0 , 1 , 1 , 1 , A , 33.0 , 1 , A , 6.0",
            " 2005-01-01 00:00:00.00 , 1.0 , 3.0 , 1 , 1 , 1 , A , 10.0 , 1 , A , 5.0"]
    (d / "isc-gem-cat.csv").write_text("\n".join(rows) + "\n")
    cat = S.read_catalog(str(d))
    assert len(cat) == 3 and sorted(cat.year) == [1990, 2005, 2010]
