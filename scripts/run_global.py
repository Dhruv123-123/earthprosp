"""Global prospectivity: features -> spatial CV + continent hold-outs -> final global split.

    python scripts/run_global.py --stats data/global_2024.npz --out outputs/global

Steps (each cached under data/ so the script can be re-run cheaply):
  1. per-cell features: AEF mean + std (from the streaming aggregation) + 10/25 km context
  2. labels: MRDS + USGS major deposits + porphyry Cu + sediment-hosted Cu + VMS
  3. label-propensity reweighting of positives (US is ~10x more densely labeled)
  4. spatial 5-fold CV on H3 res-1 blocks (~600,000 km^2) + whole-continent hold-outs
  5. final model on everything -> split for every land cell
"""

import argparse
import json
import os
import sys
import time

import h3
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import BallTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from earthprosp import commodities as C  # noqa: E402
from earthprosp import evaluate, features, grid, labels  # noqa: E402
from earthprosp.train import TrainConfig, fit_features, predict_features, spatial_folds  # noqa: E402

# Continent hold-out boxes (lon_w, lat_s, lon_e, lat_n): coarse but unambiguous.
HOLDOUTS = {
    "south_america": (-82, -56, -34, 13),
    "australia": (112, -44, 154, -10),
    "africa": (-18, -35, 52, 37),
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_features(stats_path, cache):
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        return z["cells"].astype(object), z["X"]
    z = np.load(stats_path)
    cells = z["cells"].astype(object)
    mean, std = z["mean"].astype(np.float32), z["std"].astype(np.float32)
    log(f"context features for {len(cells):,} cells")
    ctx = features.neighbourhood_context(cells, mean)
    X = np.concatenate([mean, std, ctx], 1).astype(np.float16)
    np.savez(cache, cells=cells.astype(str), X=X)
    return cells, X


def propensity_weights(cells, pos, conf, region_res=1, power=0.5, clip=(0.25, 4.0)):
    """Down-weight positives in densely-labeled regions, up-weight sparse ones.

    Labeling propensity varies ~10x between the US and most of the world. Without this,
    the model partly learns "looks like the western US". w = conf * (global_rate /
    region_rate)^power, clipped."""
    reg = np.array([h3.cell_to_parent(c, region_res) for c in cells], dtype=object)
    df = pd.DataFrame({"reg": reg, "pos": pos})
    rate = df.groupby("reg")["pos"].transform("mean").to_numpy()
    g = pos.mean()
    w = np.where(pos, np.clip((g / np.maximum(rate, 1e-9)) ** power, *clip), 0.0)
    return (conf * w).astype(np.float32)


def in_box(lat, lon, box):
    w, s, e, n = box
    return (lon >= w) & (lon <= e) & (lat >= s) & (lat <= n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", default="data/global_2024.npz")
    ap.add_argument("--features-cache", default="data/global_features.npz")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--unlabeled-per-epoch", type=int, default=1_000_000)
    ap.add_argument("--prior", type=float, default=0.05)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--skip-cv", action="store_true")
    ap.add_argument("--positive-min-conf", type=float, default=1.0,
                    help="records below this confidence (prospects/occurrences) are treated as unlabeled")
    ap.add_argument("--propensity-power", type=float, default=1.0)
    ap.add_argument("--propensity-clip", type=float, nargs=2, default=(0.05, 20.0))
    ap.add_argument("--load-model", action="store_true", help="reuse <out>/model.pt for the final prediction")
    ap.add_argument("--alpha-far", type=float, default=None,
                    help="composition shrinkage far from labels (default: fitted on hold-outs, else run.json)")
    ap.add_argument("--out", default="outputs/global")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    cells, X = build_features(a.stats, a.features_cache)
    log(f"{len(cells):,} cells, {X.shape[1]} features")
    rec = labels.global_records()
    lab_tab = labels.cell_labels_from_records(rec)
    lab = lab_tab.reindex(pd.Index(cells, name="cell"))
    conf = lab["conf"].fillna(0).to_numpy(np.float32)
    # One labelling standard worldwide: MRDS outside the US is mostly producers, so
    # US prospects/occurrences would otherwise make the US look uniquely "prospective".
    pos = lab["positive"].fillna(False).to_numpy(bool) & (conf >= a.positive_min_conf)
    major = lab["major"].fillna(False).to_numpy(bool)
    comp = lab[C.CLASSES].fillna(0).to_numpy(np.float32)
    w = propensity_weights(cells, pos, conf, power=a.propensity_power, clip=tuple(a.propensity_clip))
    lat, lon = grid.cell_centers(cells)
    us = in_box(lat, lon, (-125, 24, -66, 50))
    log(f"positives {pos.sum():,} ({pos.mean():.2%}); world-class {major.sum():,}; "
        f"non-US positives {(pos & ~us).sum():,}; label records matched {len(lab_tab):,}")
    cfg = TrainConfig(d=256, epochs=a.epochs, batch=4096, lr=2e-3, prior=a.prior, dropout=0.2)

    def evaluate_split(tag, te, s_nn, extra_scores):
        rows = []
        targets = {"all": pos[te], "non_us": pos[te] & ~us[te], "world_class": major[te]}
        for model, sc in {"nn": 1 - s_nn[:, 0], **extra_scores}.items():
            for tname, t in targets.items():
                if t.sum() >= 5:
                    rows.append({"split": tag, "model": model, "target": tname, "n_pos": int(t.sum()),
                                 **evaluate.capture_curve(sc, t)})
        p = pos[te]
        q = s_nn[p, 1:] / s_nn[p, 1:].sum(1, keepdims=True)
        rows.append({"split": tag, "model": "nn", "target": "composition", "n_pos": int(p.sum()),
                     **evaluate.composition_metrics(q, comp[te][p], conf[te][p])})
        prior_q = np.tile(comp[np.setdiff1d(np.arange(len(cells)), te)][pos[np.setdiff1d(np.arange(len(cells)), te)]].mean(0), (p.sum(), 1))
        rows.append({"split": tag, "model": "class_prior", "target": "composition", "n_pos": int(p.sum()),
                     **evaluate.composition_metrics(prior_q, comp[te][p], conf[te][p])})
        return rows

    holdout_comp = []

    def run_split(tag, te):
        t0 = time.time()
        tr = np.setdiff1d(np.arange(len(cells)), te)
        m = fit_features(X[tr], pos[tr], w[tr], comp[tr], cfg, unlabeled_subsample=a.unlabeled_per_epoch)
        s = predict_features(m, X[te])
        tree = BallTree(np.radians(np.stack([lat[tr][pos[tr]], lon[tr][pos[tr]]], 1)), metric="haversine")
        knn = -tree.query(np.radians(np.stack([lat[te], lon[te]], 1)), k=1)[0][:, 0]
        # GBM baseline on a subsample (positive-vs-unlabeled with the same weights).
        rng = np.random.default_rng(0)
        sub = np.concatenate([tr[pos[tr]], rng.choice(tr[~pos[tr]], min(500_000, (~pos[tr]).sum()), replace=False)])
        g = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=0)
        g.fit(X[sub].astype(np.float32), pos[sub], sample_weight=np.where(pos[sub], w[sub], 1.0))
        gbm = g.predict_proba(X[te].astype(np.float32))[:, 1]
        rows = evaluate_split(tag, te, s, {"knn": knn, "gbm": gbm})
        if tag.startswith("holdout"):
            p = pos[te]
            holdout_comp.append((s[p, 1:] / s[p, 1:].sum(1, keepdims=True), comp[te][p], conf[te][p],
                                 comp[tr][pos[tr]].mean(0)))
        log(f"{tag}: {len(te):,} test cells, {time.time() - t0:.0f}s")
        return rows

    if not a.skip_cv:
        rows = []
        for k, te in enumerate(spatial_folds(grid.spatial_blocks(cells, 1), a.folds)):
            rows += run_split(f"cv{k}", te)
        for name, box in HOLDOUTS.items():
            rows += run_split(f"holdout_{name}", np.where(in_box(lat, lon, box))[0])
        res = pd.DataFrame(rows)
        res.to_csv(os.path.join(a.out, "cv_results.csv"), index=False)
        res["cv"] = res.split.str.startswith("cv")
        cols = ["capture@1%", "capture@5%", "capture@10%", "capture@20%", "capture_auc",
                "top1_acc", "cross_entropy", "mass_on_present"]
        summ = pd.concat([
            res[res.cv].groupby(["model", "target"])[cols].mean().assign(split="cv_mean"),
            res[~res.cv].set_index(["model", "target"])[cols + ["split"]],
        ]).round(3)
        summ.to_csv(os.path.join(a.out, "cv_summary.csv"))
        print(summ.to_string(), flush=True)

    # Composition shrinkage toward the global class mix, alpha chosen on the continent
    # hold-outs (the regime that matters for unexplored ground): q' = a*q + (1-a)*prior.
    alpha = 1.0
    if holdout_comp:
        best = None
        for al in np.linspace(0, 1, 21):
            ce = np.mean([evaluate.composition_metrics(al * q + (1 - al) * pr, t, cw)["cross_entropy"]
                          for q, t, cw, pr in holdout_comp])
            if best is None or ce < best[0]:
                best = (ce, al)
        alpha = float(best[1])
        shr = [evaluate.composition_metrics(alpha * q + (1 - alpha) * pr, t, cw) for q, t, cw, pr in holdout_comp]
        log(f"composition shrinkage alpha={alpha:.2f}: holdout top1 "
            f"{np.mean([r['top1_acc'] for r in shr]):.3f}, CE {np.mean([r['cross_entropy'] for r in shr]):.3f}")

    import torch

    from earthprosp.model import FeatureProspectivityModel
    if a.alpha_far is not None:
        alpha = a.alpha_far
    elif not holdout_comp and os.path.exists(os.path.join(a.out, "run.json")):
        alpha = json.load(open(os.path.join(a.out, "run.json"))).get("composition_alpha", alpha)
    if a.load_model:
        m = FeatureProspectivityModel(X.shape[1], C.N_CLASSES, cfg.d, cfg.dropout)
        m.load_state_dict(torch.load(os.path.join(a.out, "model.pt")))
    else:
        log("final model on all cells")
        m = fit_features(X, pos, w, comp, cfg, unlabeled_subsample=a.unlabeled_per_epoch)
        torch.save(m.state_dict(), os.path.join(a.out, "model.pt"))
    s = predict_features(m, X)
    # Composition shrinkage by label support. Spatial CV (held-out blocks <~800 km from
    # training labels) shows the composition head beats the global class mix; whole-
    # continent hold-outs (thousands of km from labels) show it does not. So shrink
    # toward the global mix with distance to the nearest labeled deposit:
    # alpha = 1 within 500 km, alpha_far beyond 1500 km, linear in between.
    tree = BallTree(np.radians(np.stack([lat[pos], lon[pos]], 1)), metric="haversine")
    dist_km = tree.query(np.radians(np.stack([lat, lon], 1)), k=1)[0][:, 0] * 6371.0
    al = np.clip(1 - (dist_km - 500) / 1000, 0, 1) * (1 - alpha) + alpha
    prior_all = comp[pos].mean(0)
    q = s[:, 1:] / np.maximum(s[:, 1:].sum(1, keepdims=True), 1e-9)
    s[:, 1:] = (1 - s[:, :1]) * (al[:, None] * q + (1 - al[:, None]) * prior_all)
    top5 = (1 - s[:, 0]) >= np.quantile(1 - s[:, 0], 0.95)
    bias = {"us_share_land": float(us.mean()), "us_share_top5pct": float((us & top5).sum() / top5.sum())}
    for name, box in HOLDOUTS.items():
        mb = in_box(lat, lon, box)
        bias[f"{name}_world_class_in_global_top5pct"] = f"{int((mb & major & top5).sum())}/{int((mb & major).sum())}"
    log(f"bias check: {bias}")
    df = pd.DataFrame({"h3": cells.astype(str), "lat": lat.astype(np.float32), "lon": lon.astype(np.float32),
                       "p_prospective": (1 - s[:, 0]).astype(np.float32)})
    for i, c in enumerate(C.CLASSES):
        df[f"p_{c}"] = s[:, i + 1].astype(np.float32)
    df["km_to_nearest_label"] = dist_km.astype(np.float32)
    df["composition_trust"] = al.astype(np.float32)
    df["top_class"] = pd.Categorical(np.array(C.CLASSES)[s[:, 1:].argmax(1)], categories=C.CLASSES)
    df["known_positive"] = pos
    df["known_any_record"] = lab["positive"].fillna(False).to_numpy(bool)
    df["world_class_known"] = major
    df.to_parquet(os.path.join(a.out, "global_split_res6.parquet"), index=False, compression="zstd")
    with open(os.path.join(a.out, "run.json"), "w") as f:
        json.dump({"args": vars(a), "n_cells": int(len(cells)), "n_pos": int(pos.sum()),
                   "n_world_class": int(major.sum()), "composition_alpha": alpha, "bias": bias}, f, indent=2)
    log("done")


if __name__ == "__main__":
    main()
