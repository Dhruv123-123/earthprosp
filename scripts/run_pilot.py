"""Spatially blocked CV of prospectivity models on AEF cell bags + MRDS labels.

    python scripts/run_pilot.py --bags data/westus_2024.npz --country USA --out outputs/westus

Compares, on held-out spatial blocks (H3 res-3 parents, ~12,000 km^2):
  knn        distance to nearest training deposit (pure spatial interpolation, no AEF)
  lr_mean    logistic regression on the mean AEF embedding
  gbm        gradient boosting on bag statistics + neighbourhood context
  nn         adapter + gated-attention MIL + PU/composition heads (+ context covariates)
  nn_ssl     same, with self-supervised adaptation on all cells first
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import BallTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from earthprosp import aef, evaluate, features, grid, labels  # noqa: E402
from earthprosp import commodities as C  # noqa: E402
from earthprosp.train import TrainConfig, fit, predict, spatial_folds  # noqa: E402


def knn_score(tr_ll, tr_pos, te_ll):
    tree = BallTree(np.radians(tr_ll[tr_pos]), metric="haversine")
    d, _ = tree.query(np.radians(te_ll), k=1)
    return -d[:, 0]


def gbm_composition(X_tr, comp_tr, w_tr, X_te):
    y = comp_tr.argmax(1)
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0)
    clf.fit(X_tr, y, sample_weight=w_tr)
    out = np.full((len(X_te), C.N_CLASSES), 1e-6)
    out[:, clf.classes_] = clf.predict_proba(X_te)
    return out / out.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bags", required=True)
    ap.add_argument("--mrds", default="data/mrds.csv")
    ap.add_argument("--country", default=None, help="ISO A3 to restrict cells to (label coverage)")
    ap.add_argument("--countries-geojson", default="data/ne_countries.geojson")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--pretrain-epochs", type=int, default=10)
    ap.add_argument("--models", default="knn,lr_mean,gbm,nn,nn_ssl")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    models = a.models.split(",")

    cb = aef.CellBags.load(a.bags)
    if a.country:
        keep = grid.in_country(cb.cells, a.countries_geojson, a.country)
        cb = aef.CellBags(cb.cells[keep], cb.bags[keep], cb.mask[keep])
    lab = labels.align(cb.cells, labels.cell_labels(labels.load_mrds(a.mrds)))
    pos, conf, comp = lab["positive"], lab["conf"], lab["composition"]
    print(f"{len(cb.cells):,} cells, {pos.sum():,} positive ({pos.mean():.1%}), "
          f"{(conf == 1).sum():,} with a producer")

    emb = cb.mean_embedding()
    ctx = features.neighbourhood_context(cb.cells, emb)
    X = np.concatenate([features.bag_stats(cb), ctx], 1)
    cov = (ctx - ctx.mean(0)) / (ctx.std(0) + 1e-6)
    lat, lon = grid.cell_centers(cb.cells)
    ll = np.stack([lat, lon], 1)
    folds = spatial_folds(grid.spatial_blocks(cb.cells, 3), a.folds)
    prior_comp = None

    rows = []
    oof = {m: np.zeros(len(cb.cells)) for m in models}
    for k, te in enumerate(folds):
        tr = np.setdiff1d(np.arange(len(cb.cells)), te)
        prior_comp = comp[tr][pos[tr]].mean(0)
        scores, comps = {}, {}
        t0 = time.time()
        if "knn" in models:
            scores["knn"] = knn_score(ll[tr], pos[tr], ll[te])
        if "lr_mean" in models:
            lr = LogisticRegression(max_iter=2000, C=1.0).fit(emb[tr], pos[tr])
            scores["lr_mean"] = lr.predict_proba(emb[te])[:, 1]
        if "gbm" in models:
            w = np.where(pos[tr], np.maximum(conf[tr], 0.3), 1.0)
            g = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=0)
            g.fit(X[tr], pos[tr], sample_weight=w)
            scores["gbm"] = g.predict_proba(X[te])[:, 1]
            p = pos[tr]
            comps["gbm"] = gbm_composition(X[tr][p], comp[tr][p], conf[tr][p], X[te][pos[te]])
        for name, pre in (("nn", 0), ("nn_ssl", a.pretrain_epochs)):
            if name not in models:
                continue
            cfg = TrainConfig(epochs=a.epochs, pretrain_epochs=pre, prior=None, seed=k, verbose=False)
            m = fit(cb.bags[tr], cb.mask[tr], pos[tr], conf[tr], comp[tr], cov[tr], cfg,
                    unlabeled_bags=cb.bags[te], unlabeled_mask=cb.mask[te], unlabeled_cov=cov[te])
            s = predict(m, cb.bags[te], cb.mask[te], cov[te])
            scores[name] = 1 - s[:, 0]
            q = s[pos[te], 1:]
            comps[name] = q / q.sum(1, keepdims=True)
        comps["class_prior"] = np.tile(prior_comp, (pos[te].sum(), 1))

        for name, sc in scores.items():
            oof[name][te] = sc
            r = {"fold": k, "model": name, "task": "presence_all", **evaluate.capture_curve(sc, pos[te])}
            rows.append(r)
            rows.append({"fold": k, "model": name, "task": "presence_producer",
                         **evaluate.capture_curve(sc, conf[te] >= 1.0)})
        for name, q in comps.items():
            rows.append({"fold": k, "model": name, "task": "composition",
                         **evaluate.composition_metrics(q, comp[te][pos[te]], conf[te][pos[te]])})
        print(f"fold {k}: {len(te):,} test cells, {time.time() - t0:.0f}s")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(a.out, "cv_folds.csv"), index=False)
    metric_cols = [c for c in res.columns if c not in ("fold", "model", "task")]
    summary = res.groupby(["task", "model"])[metric_cols].agg(["mean", "std"]).round(3)
    summary = summary.dropna(axis=1, how="all")
    print(summary.to_string())
    summary.to_csv(os.path.join(a.out, "cv_summary.csv"))
    np.savez_compressed(os.path.join(a.out, "oof_scores.npz"), cells=cb.cells.astype(str),
                        positive=pos, conf=conf, **oof)
    with open(os.path.join(a.out, "run.json"), "w") as f:
        json.dump({"args": vars(a), "n_cells": int(len(cb.cells)), "n_pos": int(pos.sum())}, f, indent=2)


if __name__ == "__main__":
    main()
