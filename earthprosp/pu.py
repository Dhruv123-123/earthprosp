"""Positive-unlabeled learning with gradient boosting: bagging, calibration, prior.

`PUBaggingGBM` follows Mordelet & Vert (2014): each bag trains a LightGBM classifier on
all positives against a fresh random sample of unlabeled cells, and the ensemble
averages the bags. Compared with treating all unlabeled cells as negatives it

* never asks one model to separate positives from millions of mostly-unknown cells,
* gives a spread across bags that is a cheap epistemic-uncertainty estimate, and
* scales: each bag sees ~(1 + ratio) x |P| rows.

Unlabeled cells can be drawn uniformly or from a *target-group background*
(Phillips et al. 2009): weight each unlabeled cell by a sampling propensity (e.g. how
much survey effort its region has had), so the "negatives" share the spatial bias of
the positives and the model cannot win by learning "where people looked".

`elkan_noto_c` estimates the label frequency c = P(labeled | positive) on held-out
positives (Elkan & Noto 2008); dividing scores by c turns a PU classifier into an
estimate of P(deposit | x) under the selected-completely-at-random assumption.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PUBaggingGBM:
    n_bags: int = 16
    unlabeled_ratio: float = 5.0          # unlabeled rows per positive in each bag
    max_unlabeled: int = 400_000
    params: dict = field(default_factory=lambda: dict(
        n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=40,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0, max_bin=63,
        verbose=-1))
    seed: int = 0
    n_jobs: int = 4
    models: list = field(default_factory=list, repr=False)
    feature_names: list | None = None

    def fit(self, X, positive, weight=None, sample_prob=None, feature_names=None, rows=None, cols=None):
        """positive: bool (n,); weight: per-positive weight (ignored for unlabeled);
        sample_prob: optional per-cell probability for drawing unlabeled cells.

        rows / cols restrict training to a subset of X without copying it: only the rows
        drawn into each bag are gathered (X can be millions of rows by hundreds of columns)."""
        import lightgbm as lgb
        rng = np.random.default_rng(self.seed)
        positive = np.asarray(positive, bool)
        w = np.ones(len(positive), np.float32) if weight is None else np.asarray(weight, np.float32)
        universe = np.arange(len(positive)) if rows is None else np.asarray(rows)
        pos = universe[positive[universe]]
        unl = universe[~positive[universe]]
        p = None
        if sample_prob is not None:
            p = np.asarray(sample_prob, np.float64)[unl]
            p = p / p.sum()
        n_u = int(min(self.max_unlabeled, self.unlabeled_ratio * len(pos), len(unl)))
        # Positives and unlabeled get equal total weight in every bag.
        pw = w[pos] / w[pos].mean()
        uw = len(pos) / n_u
        self.models, self.feature_names = [], feature_names
        for b in range(self.n_bags):
            u = rng.choice(unl, n_u, replace=False, p=p)
            idx = np.concatenate([pos, u])
            y = np.concatenate([np.ones(len(pos)), np.zeros(n_u)])
            sw = np.concatenate([pw, np.full(n_u, uw)])
            m = lgb.LGBMClassifier(random_state=self.seed + b, n_jobs=self.n_jobs, **self.params)
            xb = X[np.sort(idx)] if cols is None else X[np.sort(idx)][:, cols]
            order = np.argsort(np.argsort(idx))  # undo the sort (sorted gathers are faster)
            m.fit(np.asarray(xb, np.float32)[order], y, sample_weight=sw)
            self.models.append(m)
        return self

    def predict_bags(self, X, chunk: int = 500_000) -> np.ndarray:
        """(n_bags, n) scores in [0, 1]."""
        out = np.zeros((len(self.models), len(X)), np.float32)
        for s in range(0, len(X), chunk):
            xb = np.asarray(X[s:s + chunk], np.float32)
            for b, m in enumerate(self.models):
                out[b, s:s + chunk] = m.predict_proba(xb)[:, 1]
        return out

    def predict(self, X) -> np.ndarray:
        return self.predict_bags(X).mean(0)

    def contributions(self, X, chunk: int = 200_000) -> np.ndarray:
        """Mean SHAP (TreeSHAP) contributions over bags, in log-odds: (n, n_features + 1)."""
        out = None
        for s in range(0, len(X), chunk):
            xb = np.asarray(X[s:s + chunk], np.float32)
            c = np.mean([m.booster_.predict(xb, pred_contrib=True) for m in self.models], 0)
            out = c if out is None else np.concatenate([out, c])
        return out

    def feature_importance(self, kind: str = "gain") -> np.ndarray:
        return np.mean([m.booster_.feature_importance(kind) for m in self.models], 0)


def elkan_noto_c(scores_on_heldout_positives: np.ndarray) -> float:
    """Label frequency estimate c = E[g(x) | y = 1] on held-out labelled positives."""
    s = np.asarray(scores_on_heldout_positives, np.float64)
    return float(np.clip(s.mean(), 1e-6, 1.0))


def rank_normalise(score: np.ndarray) -> np.ndarray:
    """Map scores to their percentile in [0, 1] (for ensembling models on different scales)."""
    order = np.argsort(np.argsort(score, kind="stable"), kind="stable")
    return (order / max(len(score) - 1, 1)).astype(np.float32)


def target_group_propensity(cells, effort_count: np.ndarray, region_res: int = 2,
                            floor: float = 0.05) -> np.ndarray:
    """Sampling weight for unlabeled cells proportional to regional survey effort.

    effort_count: per-cell count of *any* mineral record (occurrences included), a
    proxy for how much a region has been looked at. Aggregated to H3 `region_res`
    parents (~86,000 km^2 at res 2) and floored so unexplored regions still contribute."""
    import h3
    import pandas as pd
    reg = np.array([h3.cell_to_parent(c, region_res) for c in cells], dtype=object)
    dens = pd.Series(np.asarray(effort_count, np.float64)).groupby(reg).transform("mean").to_numpy()
    dens = dens / max(dens.mean(), 1e-12)
    return np.maximum(dens, floor)
