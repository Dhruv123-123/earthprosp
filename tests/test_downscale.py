import os
import sys

import h3
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import downscale  # noqa: E402


def test_aggregate_pools_children_exactly():
    """Pixel-weighted mean and std of children equal those of the pooled pixels."""
    rng = np.random.default_rng(0)
    p = h3.latlng_to_cell(40.0, -116.0, 6)
    kids = list(h3.cell_to_children(p, 8))[:3]
    px = [rng.normal(size=(n, 2)) for n in (5, 9, 2)]
    mean = np.array([x.mean(0) for x in px], np.float32)
    std = np.array([x.std(0) for x in px], np.float32)
    n = np.array([len(x) for x in px])
    cells, M, S = downscale.aggregate(downscale.parents(np.array(kids, object), 6), mean, std, n)
    allpx = np.concatenate(px)
    assert list(cells) == [p]
    np.testing.assert_allclose(M[0], allpx.mean(0), atol=1e-5)
    np.testing.assert_allclose(S[0], allpx.std(0), atol=1e-5)


def test_extra_res8_aligns_and_prefixes(tmp_path):
    c = [h3.latlng_to_cell(40.0 + i * 0.1, -116.0, 8) for i in range(3)]
    pd.DataFrame({"h3": c[:2], "s2_clay": [1.0, 2.0], "s2_n": [5, 6]}).to_parquet(tmp_path / "x.parquet")
    out = downscale.extra_res8(str(tmp_path / "x.parquet"), np.array(c[::-1], object))
    assert list(out.columns) == ["r8_s2_clay"]  # pixel counts are dropped
    assert np.isnan(out.iloc[0, 0]) and out.iloc[1:, 0].tolist() == [2.0, 1.0]
