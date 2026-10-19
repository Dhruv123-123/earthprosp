"""Does one res-8 model trained on many belts beat each belt's own model, and does it carry
over to a belt it has never seen?

    python scripts/pool_belts.py --pool data/pool --out outputs/pool_belts

Reads the per-belt samples written by downscale.py (every cell with a record plus a uniform 4%
of all cells). For each belt, under 5 buffered spatial block folds inside it, three models
score the same held-out cells:

* local: trained on the belt's own training folds (what downscale.py's fine model is).
* transfer: trained on every other belt, nothing from this one (leave one belt out).
* pooled: every other belt plus this belt's training folds.

Training uses the positives and the uniform cells (the uniform cells are the unlabeled
background, as in the full-belt model). Scores are estimated against the uniform cells of the
test folds: capture AUC = mean over test positives of the share of uniform cells scoring
below them, top 5% = share of test positives above the uniform cells' 95th percentile.
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from earthprosp import pu, spatial_cv  # noqa: E402

PARAMS = dict(n_estimators=300, learning_rate=0.05)
NEUTRAL = ("wc_built", "wc_crop")
COV_OFF = 64 * 6


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def score(s, pos, uni):
    u = np.sort(s[uni])
    sp = s[pos]
    below = (np.searchsorted(u, sp, "left") + np.searchsorted(u, sp, "right")) / 2 / len(u)
    return {"capture_auc": float(below.mean()), "top5": float((sp > np.quantile(u, 0.95)).mean()),
            "top1": float((sp > np.quantile(u, 0.99)).mean()), "n_pos": int(pos.sum())}


def fit(X, pos, w, rows, seed, bags):
    m = pu.PUBaggingGBM(n_bags=bags, seed=seed)
    m.params = {**m.params, **PARAMS}
    m.fit(X, pos, w, rows=rows)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/pool")
    ap.add_argument("--out", default="outputs/pool_belts")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--bags", type=int, default=4)
    ap.add_argument("--only", default="", help="comma list of belts to evaluate (all trained on)")
    ap.add_argument("--fit-final", default="", metavar="PATH",
                    help="instead of the test, fit one model on every belt and save it (joblib) for downscale.py")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    belts = {}
    for p in sorted(glob.glob(os.path.join(a.pool, "*.npz"))):
        z = np.load(p, allow_pickle=False)
        belts[os.path.basename(p)[:-4]] = {k: z[k] for k in z.files}
    names = list(belts)
    cols = [list(b["cov_cols"]) for b in belts.values()]
    assert all(c == cols[0] for c in cols), "covariate columns differ between belts"
    nz = [COV_OFF + cols[0].index(c) for c in NEUTRAL if c in cols[0]]
    log(f"{len(names)} belts: " + ", ".join(f"{n} {belts[n]['pos'].sum():,}+" for n in names))

    # Training rows of each belt: positives and uniform background.
    X = np.concatenate([b["X"] for b in belts.values()])
    pos = np.concatenate([b["pos"] for b in belts.values()])
    w = np.concatenate([b["w"] for b in belts.values()]).astype(np.float32)
    trainable = np.concatenate([b["pos"] | b["uniform"] for b in belts.values()])
    belt = np.concatenate([np.full(len(b["pos"]), i) for i, b in enumerate(belts.values())])
    log(f"{len(X):,} sample cells, {pos.sum():,} positive")

    if a.fit_final:
        import joblib
        m = fit(X, pos, w, np.where(trainable)[0], seed=7, bags=2 * a.bags)
        joblib.dump(m, a.fit_final)
        log(f"saved pooled model on {len(names)} belts to {a.fit_final}")
        return

    only = [n for n in a.only.split(",") if n] or names
    rows, rows_path = [], os.path.join(a.out, "pool_folds.csv")
    for i, n in enumerate(names):
        if n not in only:
            continue
        b = belts[n]
        idx = np.where(belt == i)[0]
        others = np.where((belt != i) & trainable)[0]
        t0 = time.time()
        mt = fit(X, pos, w, others, seed=100 + i, bags=a.bags)
        folds = spatial_cv.buffered_block_folds(b["cells"].astype(object), b["lat"], b["lon"], k=a.folds,
                                                block_res=2, buffer_km=10.0)
        for k, (tr, te) in enumerate(folds):
            tr_g, te_g = idx[tr][trainable[idx[tr]]], idx[te]
            Xte = np.asarray(X[te_g], np.float32)
            Xte[:, nz] = 0.0
            ml = fit(X, pos, w, tr_g, seed=k, bags=a.bags)
            mp = fit(X, pos, w, np.concatenate([others, tr_g]), seed=k, bags=a.bags)
            p_te, u_te = pos[te_g], b["uniform"][te]
            if p_te.sum() < 5 or u_te.sum() < 100:
                continue
            preds = {"local": ml.predict(Xte), "transfer": mt.predict(Xte), "pooled": mp.predict(Xte)}
            preds["local+transfer"] = np.sqrt(pu.rank_normalise(preds["local"]) * pu.rank_normalise(preds["transfer"]))
            for target, y in (("all", p_te), ("no_mine_within_1.5km", b["no_fp"][te] & p_te)):
                if y.sum() < 5:
                    continue
                for mname, s in preds.items():
                    rows.append({"belt": n, "fold": k, "model": mname, "target": target, **score(s, y, u_te)})
            log(f"{n} fold {k}: " + " ".join(f"{r['model']}={r['capture_auc']:.3f}" for r in rows
                                             if r["belt"] == n and r["fold"] == k and r["target"] == "all"))
            pd.DataFrame(rows).to_csv(rows_path, index=False)
        log(f"{n} done ({time.time() - t0:.0f}s)")

    df = pd.DataFrame(rows)
    per = df.groupby(["target", "belt", "model"])[["capture_auc", "top5", "top1"]].mean().round(4)
    per.to_csv(os.path.join(a.out, "pool_by_belt.csv"))
    piv = per["capture_auc"].unstack("model")
    print(piv.to_string())
    summ = per.groupby(["target", "model"]).median().round(4)
    wins = {m: int((piv.xs("all")[m] > piv.xs("all")["local"]).sum()) for m in piv.columns if m != "local"}
    summ.to_csv(os.path.join(a.out, "pool_summary.csv"))
    print(summ.to_string())
    print("belts beating local (all targets):", wins)


if __name__ == "__main__":
    main()
