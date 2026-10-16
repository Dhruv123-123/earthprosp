import h3
import numpy as np

from earthprosp import evaluate, pu, spatial_cv
from earthprosp.grid import cell_centers


def _patch(n_ring=25, seed=0):
    rng = np.random.default_rng(seed)
    cells = np.array(sorted(h3.grid_disk(h3.latlng_to_cell(40.0, -117.0, 6), n_ring)), dtype=object)
    lat, lon = cell_centers(cells)
    X = rng.normal(size=(len(cells), 6)).astype(np.float32)
    truth = (X[:, 0] + 0.5 * X[:, 1]) > 1.5
    labelled = truth & (rng.random(len(cells)) < 0.5)
    return cells, lat, lon, X, truth, labelled


def test_pu_bagging_recovers_planted_signal():
    cells, lat, lon, X, truth, lab = _patch()
    m = pu.PUBaggingGBM(n_bags=4, params=dict(n_estimators=60, learning_rate=0.1, num_leaves=15,
                                               min_child_samples=10, verbose=-1)).fit(X, lab)
    s = m.predict(X)
    hidden = truth & ~lab
    # Unlabelled true positives rank far above the unlabelled background.
    assert evaluate.capture_curve(s[~lab], hidden[~lab])["capture_auc"] > 0.85
    assert m.predict_bags(X).shape == (4, len(X))
    c = m.contributions(X[:50])
    assert c.shape == (50, X.shape[1] + 1)


def test_buffered_folds_drop_training_cells_near_test():
    cells, lat, lon, *_ = _patch()
    folds = spatial_cv.buffered_block_folds(cells, lat, lon, k=3, block_res=3, buffer_km=10)
    for tr, te in folds:
        assert len(np.intersect1d(tr, te)) == 0
        d = spatial_cv.geo_nn_distance_km(lat[te], lon[te], lat[tr], lon[tr])
        assert d.min() > 10
    # Every cell is a test cell exactly once.
    assert sorted(np.concatenate([te for _, te in folds])) == list(range(len(cells)))


def test_area_of_applicability_flags_extrapolation():
    rng = np.random.default_rng(0)
    Xtr = rng.normal(size=(2000, 4))
    X = np.vstack([rng.normal(size=(100, 4)), rng.normal(size=(100, 4)) + 8])
    inside, di, thr = spatial_cv.area_of_applicability(Xtr, X)
    assert inside[:100].mean() > 0.9 and inside[100:].mean() < 0.05


def test_rank_normalise_and_target_group():
    r = pu.rank_normalise(np.array([3.0, 1.0, 2.0]))
    np.testing.assert_allclose(r, [1.0, 0.0, 0.5])
    cells, *_ = _patch(5)
    w = pu.target_group_propensity(cells, np.ones(len(cells)))
    np.testing.assert_allclose(w, 1.0)
