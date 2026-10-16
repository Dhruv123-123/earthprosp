"""v3 global prospectivity: AEF + geoscience covariates, MinMod labels, PU-bagged GBM + NN.

    python scripts/run_v3.py --out outputs/v3

Stages (each cached under data/ so re-runs are cheap):
  1. features   AEF mean/std + 10/25 km context (v2 cache) and the covariate families
                (gravity, topo, faults, plates, litho, volcano) + streamed rasters
                (WorldCover, DEM, Sentinel-2) when their parquet files exist.
  2. labels     global_records_v3 -> per-cell positives (producers + deposit compilations),
                world-class flag, commodity and mineral-system mixes, survey effort.
  3. CV         buffered spatial 5-fold CV on H3 res-1 blocks + whole-continent hold-outs.
                Models: knn (distance to known deposits), PU-bagged LightGBM on AEF only,
                on covariates only, and on both; the v2 feature NN; and a rank ensemble.
                Targets: all positives, non-US positives, world-class deposits, occurrence-
                only cells (no producer), OSM-mapped metal mines outside the label set, and
                positives in cells without a mapped mine footprint (leakage test).
  4. final      models on all cells -> presence (prior-matched), uncertainty (bag spread),
                area of applicability, commodity split and mineral-system split, top drivers.
"""

import argparse
import json
import os
import sys
import time

import h3
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from earthprosp import commodities as C  # noqa: E402
from earthprosp import covariates, evaluate, features, grid, labels, pu, smoothing, spatial_cv  # noqa: E402
from earthprosp import mineral_systems as MS  # noqa: E402

HOLDOUTS = {
    "south_america": (-82, -56, -34, 13),
    "australia": (112, -44, 154, -10),
    "africa": (-18, -35, 52, 37),
}
US_BOX = (-125, 24, -66, 50)
RASTER_PRODUCTS = ("worldcover", "dem", "s2")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def in_box(lat, lon, box):
    w, s, e, n = box
    return (lon >= w) & (lon <= e) & (lat >= s) & (lat <= n)


# ---------------------------------------------------------------------------
# 1. features
# ---------------------------------------------------------------------------
def aef_features(stats_path, cache):
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


def aef_names():
    return ([f"aef_mean_A{i:02d}" for i in range(64)] + [f"aef_std_A{i:02d}" for i in range(64)]
            + [f"aef_ctx10_A{i:02d}" for i in range(64)] + [f"aef_ctx25_A{i:02d}" for i in range(64)])


def covariate_table(cells, data_dir):
    # One cache per cell set, so a regional smoke run never overwrites the global cache.
    cov = covariates.build(cells, cache=os.path.join(data_dir, f"covariates_{len(cells)}.parquet"),
                           raw_dir=os.path.join(data_dir, "raw"))
    for p in RASTER_PRODUCTS:
        path = os.path.join(data_dir, f"{p}_res6.parquet")
        if os.path.exists(path):
            r = pd.read_parquet(path)
            r.index = r.index.astype(object)
            cov = cov.join(r.reindex(cov.index))
            log(f"joined {p}: {r.shape[1]} columns, {cov.index.isin(r.index).mean():.1%} of cells covered")
    return cov


# ---------------------------------------------------------------------------
# 2. labels
# ---------------------------------------------------------------------------
def build_labels(cells, data_dir):
    rec = labels.global_records_v3(os.path.join(data_dir, "raw"), os.path.join(data_dir, "minmod_sites.parquet"))
    strict = labels.cell_labels_v3(rec, min_conf=1.0).reindex(pd.Index(cells, name="cell"))
    loose = labels.cell_labels_v3(rec, min_conf=0.0).reindex(pd.Index(cells, name="cell"))
    lab = {
        "pos": strict["positive"].fillna(False).to_numpy(bool),
        "world_class": strict["world_class"].fillna(False).to_numpy(bool),
        "conf": strict["conf"].fillna(0).to_numpy(np.float32),
        "comp": strict[C.CLASSES].fillna(0).to_numpy(np.float32),
        "sys": strict[[f"sys_{s}" for s in MS.SYSTEMS]].fillna(0).to_numpy(np.float32),
        "has_sys": strict["has_system"].fillna(False).to_numpy(bool),
        "any_record": loose["positive"].fillna(False).to_numpy(bool),
        "effort": loose["n_any"].fillna(0).to_numpy(np.float32),
    }
    lab["occurrence_only"] = lab["any_record"] & ~lab["pos"]
    if "contained_cu_kt" in rec:
        cc = grid.latlng_to_cells(rec.latitude.to_numpy(), rec.longitude.to_numpy())
        cu = pd.Series(rec["contained_cu_kt"].fillna(0).to_numpy(), index=cc).groupby(level=0).sum()
        lab["cu_kt"] = cu.reindex(cells).fillna(0).to_numpy(np.float32)
    else:
        lab["cu_kt"] = np.zeros(len(cells), np.float32)
    osm_path = os.path.join(data_dir, "overture_mines.parquet")
    lab["osm_metal"] = np.zeros(len(cells), bool)
    lab["mine_footprint"] = np.zeros(len(cells), bool)
    if os.path.exists(osm_path):
        o = labels.osm_mine_records(osm_path)
        oc = set(grid.latlng_to_cells(o.latitude.to_numpy(), o.longitude.to_numpy()))
        lab["osm_metal"] = np.isin(cells, list(oc))
        allm = pd.read_parquet(osm_path)
        allm = allm[allm["class"] == "quarry"]
        fc = set(grid.latlng_to_cells(allm.lat.to_numpy(), allm.lon.to_numpy()))
        lab["mine_footprint"] = np.isin(cells, list(fc))
    return lab, rec


def propensity_weights(cells, pos, conf, region_res=1, power=1.0, clip=(0.05, 20.0)):
    """v2 propensity reweighting: positives in densely labelled H3 res-1 regions count less."""
    reg = np.array([h3.cell_to_parent(c, region_res) for c in cells], dtype=object)
    rate = pd.Series(pos.astype(float)).groupby(reg).transform("mean").to_numpy()
    w = np.where(pos, np.clip((pos.mean() / np.maximum(rate, 1e-9)) ** power, *clip), 0.0)
    return (conf * w).astype(np.float32)


# ---------------------------------------------------------------------------
# 3/4. models
# ---------------------------------------------------------------------------
def knn_score(lat, lon, tr_pos_idx, te_idx):
    d = spatial_cv.geo_nn_distance_km(lat[tr_pos_idx], lon[tr_pos_idx], lat[te_idx], lon[te_idx])
    return -d


CV_PARAMS = dict(n_estimators=200, learning_rate=0.1)  # half the trees, double the rate: CV only ranks models


def fit_gbm(X, cols, tr, pos, w, sample_prob, n_bags, seed=0, fast=False):
    m = pu.PUBaggingGBM(n_bags=n_bags, seed=seed)
    if fast:
        m.params = {**m.params, **CV_PARAMS}
    return m.fit(X, pos, w, sample_prob=sample_prob, rows=tr, cols=np.asarray(cols))


def _cols(X, cols):
    """X[:, cols] without a copy when cols is every column (X is several GB globally)."""
    return X if len(cols) == X.shape[1] and (np.asarray(cols) == np.arange(X.shape[1])).all() else X[:, cols]


def _sub(X, rows, cols):
    """X[rows][:, cols] as float32 without materialising X[:, cols] for every row."""
    return np.asarray(X[rows][:, cols], np.float32)


def soft_multiclass(X, rows, Y, weight, n_classes, seed=0, n_estimators=300):
    """LightGBM multiclass with soft labels: one row per (cell, class) weighted by its share."""
    import lightgbm as lgb
    r, k = np.nonzero(Y[rows] > 0)
    sw = Y[rows][r, k] * weight[rows][r]
    present = np.unique(k)
    m = lgb.LGBMClassifier(objective="multiclass", n_estimators=n_estimators, learning_rate=0.05,
                           num_leaves=31, min_child_samples=20, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.6, random_state=seed, verbose=-1, n_jobs=4)
    m.fit(np.asarray(X[rows[r]], np.float32), k, sample_weight=sw)
    return m, present, n_classes


def predict_soft(model_tuple, X, chunk=500_000):
    m, present, K = model_tuple
    out = np.zeros((len(X), K), np.float32)
    for s in range(0, len(X), chunk):
        out[s:s + chunk][:, m.classes_] = m.predict_proba(np.asarray(X[s:s + chunk], np.float32))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", default="data/global_2024.npz")
    ap.add_argument("--data", default="data")
    ap.add_argument("--features-cache", default="data/global_features.npz")
    ap.add_argument("--out", default="outputs/v3")
    ap.add_argument("--bags", type=int, default=8)
    ap.add_argument("--cv-bags", type=int, default=4)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--buffer-km", type=float, default=50.0)
    ap.add_argument("--skip-cv", action="store_true")
    ap.add_argument("--with-nn", action="store_true", help="also run the v2 feature NN in CV (needs ~2x RAM)")
    ap.add_argument("--nn-epochs", type=int, default=12)
    ap.add_argument("--prior", type=float, default=0.05)
    ap.add_argument("--system-heads", action="store_true",
                    help="also fit one PU model per mineral system and compare it with presence x share")
    ap.add_argument("--target-group", action="store_true",
                    help="draw unlabeled cells in proportion to regional survey effort")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    cells, Xa = aef_features(a.stats, a.features_cache)
    lat, lon = grid.cell_centers(cells)
    cov = covariate_table(cells, a.data)
    cov_cols = list(cov.columns)
    X = np.concatenate([Xa.astype(np.float32), cov.to_numpy(np.float32)], 1)
    del Xa
    names = aef_names() + cov_cols
    sets = {"aef": np.arange(256), "geo": np.arange(256, X.shape[1]), "aef_geo": np.arange(X.shape[1])}
    log(f"{len(cells):,} cells, {X.shape[1]} features ({len(cov_cols)} covariates)")

    lab, rec = build_labels(cells, a.data)
    pos, conf = lab["pos"], lab["conf"]
    us = in_box(lat, lon, US_BOX)
    w = propensity_weights(cells, pos, conf)
    sample_prob = pu.target_group_propensity(cells, lab["effort"]) if a.target_group else None
    log(f"positives {pos.sum():,} ({pos.mean():.2%}), world-class {lab['world_class'].sum():,}, "
        f"non-US {(pos & ~us).sum():,}, occurrence-only {lab['occurrence_only'].sum():,}, "
        f"OSM metal-mine cells {lab['osm_metal'].sum():,}, footprint cells {lab['mine_footprint'].sum():,}")
    json.dump({"n_records": int(len(rec)), "by_source": rec.source.value_counts().to_dict(),
               "n_pos_cells": int(pos.sum()), "n_world_class_cells": int(lab["world_class"].sum())},
              open(os.path.join(a.out, "labels.json"), "w"), indent=2)

    sys_pos = {s: system_positive(lab, i) for i, s in enumerate(MS.SYSTEMS)} if a.system_heads else {}

    targets_of = lambda te: {  # noqa: E731
        **{f"sys:{s}": sp[te] for s, sp in sys_pos.items()},
        "all": pos[te],
        "non_us": pos[te] & ~us[te],
        "world_class": lab["world_class"][te],
        "occurrence_only": lab["occurrence_only"][te],
        "osm_metal_unlabelled": lab["osm_metal"][te] & ~lab["any_record"][te],
        "no_mine_footprint": pos[te] & ~lab["mine_footprint"][te],
        "contained_cu": lab["cu_kt"][te],  # weighted by kt Cu (Global Copper Deposit Dataset)
    }

    def run_split(tag, tr, te):
        t0 = time.time()
        scores = {"knn": knn_score(lat, lon, tr[pos[tr]], te)}
        for name in ("aef", "geo", "aef_geo"):
            m = fit_gbm(X, sets[name], tr, pos, w, sample_prob, a.cv_bags, fast=True)
            scores[f"gbm_{name}"] = m.predict(_sub(X, te, sets[name]))
        if a.with_nn:
            from earthprosp.train import TrainConfig, fit_features, predict_features
            Xn = _standardised(X, tr)
            cfg = TrainConfig(d=256, epochs=a.nn_epochs, batch=4096, lr=2e-3, prior=a.prior, dropout=0.2)
            nn = fit_features(Xn[tr], pos[tr], w[tr], lab["comp"][tr], cfg, unlabeled_subsample=500_000)
            scores["nn_aef_geo"] = 1 - predict_features(nn, Xn[te])[:, 0]
            del Xn
        # District-scale smoothing of the held-out predictions (uses predictions only).
        ridx = smoothing.ring_index(cells[te], 2)
        scores["gbm_aef_geo_ring2"] = smoothing.ring_mean(scores["gbm_aef_geo"], ridx, self_weight=0.5)
        members = [k for k in scores if k in ("gbm_aef_geo", "nn_aef_geo")]
        if len(members) > 1:
            scores["ensemble"] = np.mean([pu.rank_normalise(scores[k]) for k in members], 0)
        rows = []
        for s, sp in sys_pos.items():  # dedicated per-system heads, scored on their own target
            if sp[tr].sum() >= 100 and sp[te].sum() >= 10:
                hm = fit_gbm(X, sets["aef_geo"], tr, sp, w, sample_prob, a.cv_bags, fast=True)
                rows.append({"split": tag, "model": "sys_head", "target": f"sys:{s}", "n_pos": int(sp[te].sum()),
                             **evaluate.capture_curve(hm.predict(_sub(X, te, sets["aef_geo"])), sp[te])})
        for model, sc in scores.items():
            for tname, t in targets_of(te).items():
                hit, wt = (t > 0, t) if t.dtype != bool else (t, None)
                if hit.sum() >= 10:
                    rows.append({"split": tag, "model": model, "target": tname, "n_pos": int(hit.sum()),
                                 **evaluate.capture_curve(sc, hit, weight=wt)})
        log(f"{tag}: train {len(tr):,} test {len(te):,} ({time.time() - t0:.0f}s) "
            + " ".join(f"{r['model']}={r['capture_auc']:.3f}" for r in rows if r["target"] == "all"))
        return rows

    if not a.skip_cv:
        rows = []
        folds = spatial_cv.buffered_block_folds(cells, lat, lon, k=a.folds, block_res=1, buffer_km=a.buffer_km)
        json.dump({k: {str(q): float(v) for q, v in d.items()} for k, d in
                   spatial_cv.nndm_like_folds(lat, lon, pos, folds).items()},
                  open(os.path.join(a.out, "cv_distance_check.json"), "w"), indent=2)
        for k, (tr, te) in enumerate(folds):
            rows += run_split(f"cv{k}", tr, te)
            pd.DataFrame(rows).to_csv(os.path.join(a.out, "cv_results.csv"), index=False)
        for name, box in HOLDOUTS.items():
            te = np.where(in_box(lat, lon, box))[0]
            if pos[te].sum() < 10:  # region outside this run's cells (regional runs)
                log(f"holdout_{name}: skipped, {int(pos[te].sum())} positive cells")
                continue
            tr = np.setdiff1d(np.arange(len(cells)), te)
            rows += run_split(f"holdout_{name}", tr, te)
            pd.DataFrame(rows).to_csv(os.path.join(a.out, "cv_results.csv"), index=False)
        res = pd.DataFrame(rows)
        res["cv"] = res.split.str.startswith("cv")
        cols = ["capture@1%", "capture@5%", "capture@10%", "capture@20%", "capture_auc"]
        summ = pd.concat([res[res.cv].groupby(["target", "model"])[cols].mean().assign(split="cv_mean"),
                          res[~res.cv].set_index(["target", "model"])[cols + ["split"]]]).round(3)
        summ.to_csv(os.path.join(a.out, "cv_summary.csv"))
        print(summ.to_string(), flush=True)

    final(a, cells, lat, lon, X, names, sets, lab, w, sample_prob, us)


def system_positive(lab, i, min_share=0.5):
    """Positive cells whose mineral-system mix is dominated (>= min_share) by system i."""
    S = lab["sys"]
    share = S[:, i] / np.maximum(S.sum(1), 1e-9)
    return lab["pos"] & lab["has_sys"] & (S.argmax(1) == i) & (share >= min_share)


def _standardised(X, rows):
    mu = np.nanmean(X[rows[::10]], 0)
    sd = np.nanstd(X[rows[::10]], 0) + 1e-6
    out = np.empty(X.shape, np.float16)
    for s in range(0, len(X), 500_000):
        out[s:s + 500_000] = np.nan_to_num((X[s:s + 500_000] - mu) / sd, nan=0.0)
    return out


def final(a, cells, lat, lon, X, names, sets, lab, w, sample_prob, us):
    import joblib
    pos = lab["pos"]
    allr = np.arange(len(cells))
    log("final PU-bagged GBM on all cells")
    cols = sets["aef_geo"]
    m = fit_gbm(X, cols, allr, pos, w, sample_prob, a.bags, seed=1)
    joblib.dump(m, os.path.join(a.out, "gbm_presence.joblib"))
    Xc = _cols(X, cols)
    bags = m.predict_bags(Xc)
    score = bags.mean(0)
    unc = bags.std(0)
    # Prior-matched calibration: shift the log-odds so the mean presence over land equals
    # the assumed deposit density (the PU prior). Scores stay a ranking; this only sets the scale.
    logit = np.log(np.clip(score, 1e-6, 1 - 1e-6) / (1 - np.clip(score, 1e-6, 1 - 1e-6)))
    lo, hi = -20.0, 20.0
    for _ in range(60):
        b = (lo + hi) / 2
        if (1 / (1 + np.exp(-(logit + b)))).mean() > a.prior:
            hi = b
        else:
            lo = b
    p = (1 / (1 + np.exp(-(logit + b)))).astype(np.float32)
    log(f"presence calibrated to mean {p.mean():.3f} (shift {b:.2f})")

    log("district-scale smoothing (ring 2, half own score)")
    p_district = smoothing.ring_mean(p, smoothing.ring_index(cells, 2), self_weight=0.5)

    log("commodity and mineral-system heads")
    prow = np.where(pos)[0]
    comp_m = soft_multiclass(Xc, prow, lab["comp"], lab["conf"], C.N_CLASSES)
    srow = np.where(pos & lab["has_sys"])[0]
    sys_m = soft_multiclass(Xc, srow, lab["sys"], lab["conf"], MS.N_SYSTEMS)
    q = predict_soft(comp_m, Xc)
    qs = predict_soft(sys_m, Xc)
    joblib.dump({"commodity": comp_m, "system": sys_m}, os.path.join(a.out, "gbm_heads.joblib"))
    heads = {}
    if a.system_heads:
        for i, s in enumerate(MS.SYSTEMS):
            sp = system_positive(lab, i)
            if sp.sum() >= 100:
                log(f"system head {s}: {int(sp.sum()):,} positive cells")
                hm = fit_gbm(X, cols, allr, sp, w, sample_prob, max(a.bags // 2, 2), seed=2)
                heads[s] = pu.rank_normalise(hm.predict(Xc)).astype(np.float32)

    log("area of applicability (top-20 features by gain, importance-weighted)")
    imp = m.feature_importance("gain")
    k20 = np.asarray(cols)[np.argsort(-imp)[:20]]
    inside, di, thr = spatial_cv.area_of_applicability(X[pos][:, k20], X[:, k20], weights=np.sort(imp)[::-1][:20],
                                                       max_train=5000)

    log("top drivers by feature family for the top 1% of cells")
    top = np.argsort(-score)[: max(len(score) // 100, 1)]
    contrib = m.contributions(X[top][:, cols])[:, :-1]
    fam = np.array([_family(n) for n in np.array(names)[cols]])
    fams = sorted(set(fam))
    fam_contrib = np.stack([contrib[:, fam == f].sum(1) for f in fams], 1)
    order = np.argsort(-fam_contrib, 1)[:, :3]
    drv = pd.DataFrame({"h3": cells[top].astype(str)})
    for j in range(3):
        drv[f"driver{j + 1}"] = np.array(fams)[order[:, j]]
        drv[f"driver{j + 1}_logodds"] = fam_contrib[np.arange(len(top)), order[:, j]].astype(np.float32)
    drv.to_parquet(os.path.join(a.out, "top1pct_drivers.parquet"), index=False)
    pd.DataFrame(fam_contrib.mean(0)[None], columns=fams).T.rename(columns={0: "mean_logodds_top1pct"}).to_csv(
        os.path.join(a.out, "driver_families_top1pct.csv"))
    imp_df = pd.DataFrame({"feature": np.array(names)[cols], "family": fam, "gain": imp}).sort_values("gain", ascending=False)
    imp_df.to_csv(os.path.join(a.out, "feature_importance.csv"), index=False)
    imp_df.groupby("family")["gain"].sum().sort_values(ascending=False).to_csv(os.path.join(a.out, "family_importance.csv"))

    df = pd.DataFrame({"h3": cells.astype(str), "lat": lat.astype(np.float32), "lon": lon.astype(np.float32),
                       "p_prospective": p, "p_district": p_district, "score_rank": pu.rank_normalise(score), "uncertainty": unc.astype(np.float32),
                       "dissimilarity": di, "in_aoa": inside})
    for i, c in enumerate(C.CLASSES):
        df[f"p_{c}"] = (p * q[:, i]).astype(np.float32)
    for i, s in enumerate(MS.SYSTEMS):
        df[f"sys_{s}"] = qs[:, i].astype(np.float32)
    for s, v in heads.items():
        df[f"head_{s}"] = v  # rank (0-1) from the dedicated per-system PU model
    df["top_class"] = pd.Categorical(np.array(C.CLASSES)[q.argmax(1)], categories=C.CLASSES)
    df["top_system"] = pd.Categorical(np.array(MS.SYSTEMS)[qs.argmax(1)], categories=MS.SYSTEMS)
    df["known_positive"] = pos
    df["world_class_known"] = lab["world_class"]
    df["any_record"] = lab["any_record"]
    df.to_parquet(os.path.join(a.out, "global_v3_res6.parquet"), index=False, compression="zstd")
    top5 = p >= np.quantile(p, 0.95)
    bias = {"us_share_land": float(us.mean()), "us_share_top5pct": float((us & top5).sum() / top5.sum()),
            "aoa_fraction": float(inside.mean()), "aoa_threshold": thr}
    for name, box in HOLDOUTS.items():
        mb = in_box(lat, lon, box)
        bias[f"{name}_world_class_in_global_top5pct"] = \
            f"{int((mb & lab['world_class'] & top5).sum())}/{int((mb & lab['world_class']).sum())}"
    json.dump({"args": vars(a), "n_cells": int(len(cells)), "n_pos": int(pos.sum()), "bias": bias,
               "calibration_shift": float(b)}, open(os.path.join(a.out, "run.json"), "w"), indent=2)
    log(f"bias check: {bias}")
    log("done")


def _family(name: str) -> str:
    if name.startswith("aef_"):
        return "aef_" + name.split("_")[1]
    for p, f in (("wc_", "landcover"), ("dem_", "terrain"), ("slope_", "terrain"), ("s2_", "s2_alteration")):
        if name.startswith(p):
            return f
    return covariates.family_of(name)


if __name__ == "__main__":
    main()
