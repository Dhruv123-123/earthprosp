"""Does a finer grid find deposits better? Res-8 (~0.7 km²) vs res-6 (~36 km²) in one region.

    python scripts/fetch_global.py --bbox -121 31 -108 43 --level 3 --res 8 --out data/wus_r8.npz
    python scripts/downscale.py --stats data/wus_r8.npz --bbox -121 31 -108 43 --out outputs/downscale_wus

Both models are PU-bagged LightGBM trained and tested under the same spatial block folds
(H3 res-2 blocks, 10 km dead zone) inside the region, and both are scored on res-8 cells:

* coarse: features and labels at res 6 (AEF mean/std aggregated from the res-8 cells, ring
  context, the lean covariates), its score copied to every res-8 child. This is what the
  global map can tell you at res 8.
* fine: res-8 AEF mean/std, rings of ~2 and ~5 km, the parent res-6 AEF and covariates,
  trained on res-8 labels.
* blend: geometric mean of the two ranks (a district from the coarse model, a spot inside
  it from the fine one).

Targets: all positive res-8 cells; occurrence-only cells (prospects and showings with no
recorded production); and positive cells with no mapped mine or quarry within ~1.5 km,
which a model cannot find by seeing a pit. "Within-district" capture ranks only the res-8
cells inside the coarse model's top 10% of res-6 cells: how well each model narrows the
search once the district is chosen.
"""

import argparse
import json
import os
import sys
import time

import h3
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import run_v3 as R  # noqa: E402
from earthprosp import evaluate, features, grid, labels, pu, spatial_cv  # noqa: E402

CV = dict(n_estimators=300, learning_rate=0.05)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parents(cells, res):
    return np.array([h3.cell_to_parent(c, res) for c in cells], dtype=object)


def aggregate(child_parent, mean, std, n):
    """Pixel-count-weighted mean / std of children per parent; returns (parents, mean, std)."""
    codes, uniq = pd.factorize(child_parent)
    w = n.astype(np.float64)
    M = np.zeros((len(uniq), mean.shape[1]))
    S = np.zeros_like(M)
    W = np.bincount(codes, w, minlength=len(uniq))
    np.add.at(M, codes, mean * w[:, None])
    np.add.at(S, codes, (std ** 2 + mean ** 2) * w[:, None])
    M /= W[:, None]
    S = np.sqrt(np.maximum(S / W[:, None] - M ** 2, 0))
    return np.asarray(uniq), M.astype(np.float32), S.astype(np.float32)


def lean_covariates(cells6, data_dir, drop):
    """Lean res-6 covariates (the v3L set) for the given res-6 cells."""
    import pyarrow.parquet as pq
    frames = []
    for path, prefix in [(os.path.join(data_dir, "covariates_4489547.parquet"), ""),
                         *[(os.path.join(data_dir, f"{p}_res6.parquet"), "") for p in R.RASTER_PRODUCTS]]:
        if not os.path.exists(path):
            continue
        names = [c for c in pq.read_schema(path).names if c not in ("h3", "__index_level_0__") and c not in drop]
        df = pd.read_parquet(path, columns=names)
        frames.append(df.reindex(pd.Index(cells6)).astype(np.float32))
    return pd.concat(frames, axis=1)


def save_pool_sample(a, c8, lat, lon, X8, pos8, w8, any_rec8, no_fp, cov_cols, frac=0.04):
    """Every cell with a record plus a uniform `frac` of all cells, for training across belts.
    `uniform` marks the random cells, which stand in for the whole belt when scoring it."""
    rng = np.random.default_rng(0)
    uniform = rng.random(len(c8)) < frac
    keep = np.where(uniform | pos8 | any_rec8)[0]
    os.makedirs(a.pool_dir, exist_ok=True)
    name = os.path.basename(a.out.rstrip("/")).replace("downscale_", "")
    np.savez(os.path.join(a.pool_dir, f"{name}.npz"), X=np.asarray(X8[keep], np.float16), pos=pos8[keep],
             w=w8[keep], any_rec=any_rec8[keep], no_fp=no_fp[keep], uniform=uniform[keep],
             lat=lat[keep].astype(np.float32), lon=lon[keep].astype(np.float32),
             cells=c8[keep].astype(str), cov_cols=np.array(cov_cols), n_cells=len(c8))
    log(f"pool sample: {len(keep):,} cells ({uniform.sum():,} uniform)")


def footprint_cells(osm_path, bbox, res, k=2):
    o = pd.read_parquet(osm_path, columns=["class", "lat", "lon"])
    x0, y0, x1, y1 = bbox
    o = o[(o["class"] == "quarry") & o.lon.between(x0, x1) & o.lat.between(y0, y1)]
    out = set()
    for c in grid.latlng_to_cells(o.lat.to_numpy(), o.lon.to_numpy(), res):
        out.update(h3.grid_disk(c, k))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", required=True)
    ap.add_argument("--bbox", type=float, nargs=4, required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--drop-file", default="data/drop_lean.txt")
    ap.add_argument("--neutralize-features", default="wc_built,wc_crop")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--bags", type=int, default=4)
    ap.add_argument("--skip-cv", action="store_true")
    ap.add_argument("--final", action="store_true",
                    help="fit the fine model on every cell and write the res-8 map, blended with the global map")
    ap.add_argument("--global-map", default="outputs/v3L/global_v3_res6.parquet")
    ap.add_argument("--pooled-model", default="",
                    help="joblib from pool_belts.py --fit-final: scored as 'pooled' in the CV, and used for the "
                         "map when the region has fewer than --pool-below positive cells")
    ap.add_argument("--pool-below", type=int, default=400)
    ap.add_argument("--fine-params", default="{}",
                    help='JSON LightGBM params for the fine model, e.g. \'{"num_leaves": 15, "min_child_samples": 20}\'')
    ap.add_argument("--fine-bags", type=int, default=0, help="bags for the fine model (default --bags)")
    ap.add_argument("--pool-dir", default="data/pool",
                    help="write a training sample here for pool_belts.py (empty string to skip)")
    a = ap.parse_args()
    a.fine_params = json.loads(a.fine_params)
    a.fine_bags = a.fine_bags or a.bags
    os.makedirs(a.out, exist_ok=True)
    x0, y0, x1, y1 = a.bbox

    z = np.load(a.stats)
    c8 = z["cells"].astype(object)
    lat, lon = grid.cell_centers(c8)
    keep = (lon >= x0) & (lon <= x1) & (lat >= y0) & (lat <= y1)
    c8, lat, lon = c8[keep], lat[keep], lon[keep]
    m8, s8, n8 = z["mean"][keep].astype(np.float32), z["std"][keep].astype(np.float32), z["n"][keep]
    log(f"{len(c8):,} res-8 cells")

    p6 = parents(c8, 6)
    c6, m6, s6 = aggregate(p6, m8, s8, n8)
    i6 = pd.Index(c6).get_indexer(p6)
    log(f"{len(c6):,} res-6 parents; ring context")
    ctx6 = features.neighbourhood_context(c6, m6, rings=(2, 6))
    ctx8 = features.neighbourhood_context(c8, m8, rings=(2, 6))
    drop = set(open(a.drop_file).read().strip().split(",")) if a.drop_file else set()
    cov = lean_covariates(c6, a.data, drop)
    neutral = [c for c in a.neutralize_features.split(",") if c in cov]
    log(f"{cov.shape[1]} lean covariates")

    X6 = np.concatenate([m6, s6, ctx6, cov.to_numpy(np.float32)], 1)
    X8 = np.concatenate([m8, s8, ctx8, m6[i6], s6[i6], cov.to_numpy(np.float32)[i6]], 1).astype(np.float16)
    cov_off6 = 64 * 4
    cov_off8 = 64 * 6
    nz6 = [cov_off6 + list(cov.columns).index(c) for c in neutral]
    nz8 = [cov_off8 + list(cov.columns).index(c) for c in neutral]
    del ctx8

    rec = labels.global_records_v3(os.path.join(a.data, "raw"), os.path.join(a.data, "minmod_sites.parquet"))
    rec = rec[rec.longitude.between(x0 - 1, x1 + 1) & rec.latitude.between(y0 - 1, y1 + 1)]
    lab8 = labels.cell_labels_v3(rec, res=8, min_conf=1.0).reindex(pd.Index(c8))
    any8 = labels.cell_labels_v3(rec, res=8, min_conf=0.0).reindex(pd.Index(c8))
    lab6 = labels.cell_labels_v3(rec, res=6, min_conf=1.0).reindex(pd.Index(c6))
    pos8 = lab8["positive"].fillna(False).to_numpy(bool)
    any_rec8 = any8["positive"].fillna(False).to_numpy(bool)
    pos6 = lab6["positive"].fillna(False).to_numpy(bool)
    w8 = lab8["conf"].fillna(0).to_numpy(np.float32)
    w6 = lab6["conf"].fillna(0).to_numpy(np.float32)
    fp = footprint_cells(os.path.join(a.data, "overture_mines.parquet"), a.bbox, 8)
    no_fp = pos8 & ~pd.Index(c8).isin(fp)
    log(f"positives: res-8 {pos8.sum():,} ({pos8.mean():.3%}), res-6 {pos6.sum():,} ({pos6.mean():.2%}), "
        f"occurrence-only res-8 {(any_rec8 & ~pos8).sum():,}, positives with no mine/quarry within ~1.5 km {no_fp.sum():,}")

    if a.pool_dir:
        save_pool_sample(a, c8, lat, lon, X8, pos8, w8, any_rec8, no_fp, list(cov.columns))
    pooled = None
    if a.pooled_model:
        import joblib
        pooled = joblib.load(a.pooled_model)
    if a.final:
        final_map(a, c8, lat, lon, p6, X8, nz8, pos8, w8, any_rec8, pooled)
    if a.skip_cv:
        return
    folds = spatial_cv.buffered_block_folds(c8, lat, lon, k=a.folds, block_res=2, buffer_km=10.0)
    models = ("coarse", "fine", "blend") + (("pooled",) if pooled is not None else ())
    rows, oof = [], {k: np.full(len(c8), np.nan, np.float32) for k in models}
    for k, (tr8, te8) in enumerate(folds):
        t0 = time.time()
        te6_set = set(p6[te8])
        tr6_set = set(p6[tr8]) - te6_set
        tr6 = np.where(pd.Index(c6).isin(tr6_set))[0]
        te6 = np.where(pd.Index(c6).isin(te6_set))[0]
        X6n = X6.copy()
        X6n[:, nz6] = 0.0  # predict as if no town or farmland, as in v3
        m = pu.PUBaggingGBM(n_bags=a.bags, seed=k)
        m.params = {**m.params, **CV}
        m.fit(X6, pos6, w6, rows=tr6)
        s6_te = pd.Series(m.predict(X6n[te6]), index=c6[te6])
        coarse = s6_te.reindex(p6[te8]).to_numpy(np.float32)
        Xte = np.asarray(X8[te8], np.float32)
        Xte[:, nz8] = 0.0
        mf = pu.PUBaggingGBM(n_bags=a.fine_bags, seed=k)
        mf.params = {**mf.params, **CV, **a.fine_params}
        mf.fit(X8, pos8, w8, rows=tr8)
        fine = mf.predict(Xte)
        blend = np.sqrt(pu.rank_normalise(coarse) * pu.rank_normalise(fine))
        oof["coarse"][te8], oof["fine"][te8], oof["blend"][te8] = coarse, fine, blend
        scored = [("coarse", coarse), ("fine", fine), ("blend", blend)]
        if pooled is not None:  # trained on other belts only, so every fold is unseen ground
            oof["pooled"][te8] = pooled.predict(Xte)
            scored.append(("pooled", oof["pooled"][te8]))
        district = coarse >= np.quantile(coarse, 0.9)
        targets = {"all": pos8[te8], "occurrence_only": (any_rec8 & ~pos8)[te8], "no_mine_within_1.5km": no_fp[te8]}
        for name, s in scored:
            for t, y in targets.items():
                if y.sum() < 5:
                    continue
                r = {"fold": k, "model": name, "target": t, "n_pos": int(y.sum()),
                     **evaluate.capture_curve(s, y)}
                yd = y[district]
                r["within_district_auc"] = evaluate.capture_curve(s[district], yd)["capture_auc"] if yd.sum() >= 5 else np.nan
                rows.append(r)
        log(f"fold {k}: test {len(te8):,} res-8 cells, {pos8[te8].sum():,} positive "
            f"({time.time() - t0:.0f}s) " + " ".join(
                f"{r['model']}={r['capture_auc']:.3f}/{r['within_district_auc']:.3f}"
                for r in rows if r["fold"] == k and r["target"] == "all"))
        pd.DataFrame(rows).to_csv(os.path.join(a.out, "downscale_folds.csv"), index=False)

    df = pd.DataFrame(rows)
    summ = df.groupby(["target", "model"])[["capture@1%", "capture@5%", "capture_auc", "within_district_auc"]].mean().round(4)
    summ.to_csv(os.path.join(a.out, "downscale_summary.csv"))
    print(summ.to_string())
    pd.DataFrame({"h3": c8, "lat": lat, "lon": lon, **oof, "positive": pos8, "any_record": any_rec8}).to_parquet(
        os.path.join(a.out, "downscale_oof_res8.parquet"), index=False, compression="zstd")
    json.dump({"bbox": a.bbox, "n_res8": int(len(c8)), "n_res6": int(len(c6)), "n_pos8": int(pos8.sum()),
               "n_pos6": int(pos6.sum()), "folds": a.folds, "bags": a.bags,
               "map_model": getattr(a, "map_model", "fine"), "fine_params": a.fine_params, "fine_bags": a.fine_bags, "covariates": list(cov.columns)},
              open(os.path.join(a.out, "downscale.json"), "w"), indent=2)


def final_map(a, c8, lat, lon, p6, X8, nz8, pos8, w8, any_rec8, pooled=None):
    """Res-8 map: the fine model fitted on every cell (or, where the region has few deposits,
    the model pooled across belts), blended with the shipped global map."""
    use_pooled = pooled is not None and pos8.sum() < a.pool_below
    sp = os.path.join(a.out, "downscale_summary.csv")
    if pooled is not None and os.path.exists(sp):  # a CV already ran here: map with whichever won it
        s = pd.read_csv(sp).set_index(["target", "model"]).capture_auc
        if ("all", "pooled") in s.index:
            use_pooled = s[("all", "pooled")] > s[("all", "fine")]
            log(f"CV: fine {s[('all', 'fine')]:.3f}, pooled {s[('all', 'pooled')]:.3f}")
    if use_pooled:
        log("mapping with the pooled model")
        m, a.map_model = pooled, "pooled"
    else:
        log("final fine model on all cells")
        a.map_model = "fine"
        m = pu.PUBaggingGBM(n_bags=2 * a.fine_bags, seed=1)
        m.params = {**m.params, **a.fine_params}
        m.fit(X8, pos8, w8)
    fine = np.empty(len(c8), np.float32)
    for s in range(0, len(c8), 300_000):
        xb = np.asarray(X8[s:s + 300_000], np.float32)
        xb[:, nz8] = 0.0
        fine[s:s + 300_000] = m.predict(xb)
    g = pd.read_parquet(a.global_map, columns=["h3", "p_prospective"]).set_index("h3").p_prospective
    district = g.reindex(p6).to_numpy(np.float32)
    district = np.where(np.isnan(district), np.nanmedian(district), district)
    rf, rd = pu.rank_normalise(fine), pu.rank_normalise(district)
    blend = np.sqrt(rf * rd).astype(np.float32)
    # Greenfield: no mineral record of any kind within ~3 km (res-8 ring 3).
    rec_cells = set(c8[any_rec8])
    near = np.array([any(n in rec_cells for n in h3.grid_disk(c, 3)) for c in c8])
    df = pd.DataFrame({"h3": c8, "lat": lat, "lon": lon, "fine_rank": rf, "district_rank": rd, "blend": blend,
                       "known_positive": pos8, "any_record": any_rec8, "record_within_3km": near})
    df.to_parquet(os.path.join(a.out, "map_res8.parquet"), index=False, compression="zstd")
    top = df[~df.record_within_3km].nlargest(200, "blend")
    # keep one target per ~5 km (res-7 parent) so the list is not one cluster
    top = top.assign(p7=parents(top.h3.to_numpy(), 7)).drop_duplicates("p7").head(50).drop(columns="p7")
    top.round(4).to_csv(os.path.join(a.out, "greenfield_targets_res8.csv"), index=False)
    import render_res8
    render_res8.main(a.out)
    log(f"wrote map_res8.parquet, map_res8.png, {len(top)} greenfield targets")


if __name__ == "__main__":
    main()
