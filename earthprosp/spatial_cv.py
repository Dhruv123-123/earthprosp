"""Spatial cross-validation with buffers, and the area of applicability.

Deposit labels are strongly clustered (districts, belts), so a test cell next to a
training deposit is not an honest test. Three tools:

* `buffered_block_folds`: whole H3 blocks per fold (as before) **plus** a dead zone:
  training cells within `buffer_km` of any test block are dropped from training
  (Roberts et al. 2017; Ploton et al. 2020). The test set is unchanged.
* `dissimilarity_index` / `area_of_applicability`: Meyer & Pebesma (2021). For each
  prediction cell, the distance in (importance-weighted, standardised) feature space to
  the nearest training cell, divided by the mean nearest-neighbour distance among
  training cells. Cells with DI above the training-data threshold lie outside the area
  of applicability: the model is extrapolating there and its score deserves less trust.
* `nndm_like_folds`: a cheap kNNDM-style check (Linnenbrink et al. 2024) that reports
  how the test-to-train nearest-neighbour distance distribution of a fold design
  compares with the prediction-to-train distribution, so fold sizes can be chosen to
  mimic the actual mapping task.
"""

from __future__ import annotations

import h3
import numpy as np
from sklearn.neighbors import BallTree, NearestNeighbors

R_EARTH_KM = 6371.0


def block_ids(cells, block_res: int) -> np.ndarray:
    return np.array([h3.cell_to_parent(c, block_res) for c in cells], dtype=object)


def buffered_block_folds(cells, lat, lon, k: int = 5, block_res: int = 1, buffer_km: float = 50.0,
                         seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """[(train_idx, test_idx)]: blocks assigned to folds, train excludes a buffer around test."""
    groups = block_ids(cells, block_res)
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    fold_of = {g: i % k for i, g in enumerate(uniq)}
    f = np.array([fold_of[g] for g in groups])
    pts = np.radians(np.stack([lat, lon], 1))
    out = []
    for i in range(k):
        te = np.where(f == i)[0]
        tr = np.where(f != i)[0]
        if buffer_km > 0:
            # Only training cells near a block boundary can be inside the buffer: test the
            # distance from every training cell to the test set.
            tree = BallTree(pts[te], metric="haversine")
            d, _ = tree.query(pts[tr], k=1)
            tr = tr[d[:, 0] * R_EARTH_KM > buffer_km]
        out.append((tr, te))
    return out


def _scaler(Xtr, weights=None):
    mu = np.nanmean(Xtr, 0)
    sd = np.nanstd(Xtr, 0) + 1e-9
    w = np.ones(Xtr.shape[1]) if weights is None else np.asarray(weights, np.float64)
    w = w / (w.sum() + 1e-12) * len(w)
    return lambda Z: np.nan_to_num((np.asarray(Z, np.float64) - mu) / sd * w)


def dissimilarity_index(X_train, X, weights=None, max_train: int = 50_000, seed: int = 0):
    """DI of each row of X relative to X_train, and the training-data threshold.

    Returns (di, threshold): threshold = upper whisker (Q3 + 1.5 IQR) of the training
    cells' own DI computed leave-one-out, as in Meyer & Pebesma (2021)."""
    rng = np.random.default_rng(seed)
    Xtr = np.asarray(X_train, np.float32)
    if len(Xtr) > max_train:
        Xtr = Xtr[rng.choice(len(Xtr), max_train, replace=False)]
    scale = _scaler(Xtr.astype(np.float64), weights)
    A = scale(Xtr)
    nn = NearestNeighbors(n_neighbors=2).fit(A)
    d_tr = nn.kneighbors(A)[0][:, 1]
    dbar = d_tr.mean()
    di_tr = d_tr / dbar
    q1, q3 = np.percentile(di_tr, [25, 75])
    thr = q3 + 1.5 * (q3 - q1)
    out = np.zeros(len(X), np.float32)
    for s in range(0, len(X), 100_000):  # X may be huge: scale chunk by chunk
        out[s:s + 100_000] = nn.kneighbors(scale(X[s:s + 100_000]), n_neighbors=1)[0][:, 0] / dbar
    return out, float(thr)


def area_of_applicability(X_train, X, weights=None, **kw):
    di, thr = dissimilarity_index(X_train, X, weights, **kw)
    return di <= thr, di, thr


def geo_nn_distance_km(src_lat, src_lon, dst_lat, dst_lon) -> np.ndarray:
    tree = BallTree(np.radians(np.stack([src_lat, src_lon], 1)), metric="haversine")
    d, _ = tree.query(np.radians(np.stack([dst_lat, dst_lon], 1)), k=1)
    return d[:, 0] * R_EARTH_KM


def nndm_like_folds(lat, lon, positive, folds, pred_mask=None, n_sample: int = 20_000, seed: int = 0):
    """Compare geographic NN distances: CV test positives -> training positives, against
    prediction cells -> all positives. Returns quantiles (km) of both distributions."""
    rng = np.random.default_rng(seed)
    cv = []
    for tr, te in folds:
        tp, ep = tr[positive[tr]], te[positive[te]]
        if len(tp) and len(ep):
            cv.append(geo_nn_distance_km(lat[tp], lon[tp], lat[ep], lon[ep]))
    cv = np.concatenate(cv) if cv else np.array([np.nan])
    cand = np.where(pred_mask)[0] if pred_mask is not None else np.arange(len(lat))
    s = rng.choice(cand, min(n_sample, len(cand)), replace=False)
    pos = np.where(positive)[0]
    pr = geo_nn_distance_km(lat[pos], lon[pos], lat[s], lon[s])
    q = [10, 25, 50, 75, 90]
    return {"cv_test_to_train_km": dict(zip(q, np.percentile(cv, q).round(1))),
            "prediction_to_train_km": dict(zip(q, np.percentile(pr, q).round(1)))}
