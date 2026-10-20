"""How sure is "the 0.7 km² model beats the 36 km² map"? Block-bootstrap intervals per region.

    python scripts/bootstrap_downscale.py --out outputs/bootstrap_downscale

Reads each region's out-of-fold scores (outputs/downscale_<region>/downscale_oof_res8.parquet,
written by downscale.py). Deposits cluster, so cells are resampled in H3 res-3 blocks
(~12,000 km²), not one by one. For fine, blend and (where present) pooled, reports the
capture-AUC difference from coarse: the point estimate, a 90% interval and the share of
resamples where it is above zero. Scores are ranked within their CV fold before folds are
pooled, since each fold's model scores on its own scale.
"""

import argparse
import glob
import os
import sys

import h3
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from earthprosp import pu, spatial_cv  # noqa: E402


def capture_auc(score, y):
    order = np.argsort(-score, kind="stable")
    c = np.cumsum(y[order].astype(np.float64))
    return c.mean() / max(c[-1], 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/bootstrap_downscale")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--block-res", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    for p in sorted(glob.glob("outputs/downscale_*/downscale_oof_res8.parquet")):
        region = p.split("/")[1].replace("downscale_", "")
        d = pd.read_parquet(p).dropna(subset=["coarse", "fine"])
        models = [m for m in ("fine", "blend", "pooled") if m in d and d[m].notna().all()]
        blocks = pd.factorize(np.array([h3.cell_to_parent(c, a.block_res) for c in d.h3]))[0]
        idx = [np.where(blocks == b)[0] for b in range(blocks.max() + 1)]
        y = d.positive.to_numpy(bool)
        # Each fold's model scores on its own scale, so rank within the fold before pooling
        # folds (the same folds downscale.py drew: seed 0, res-2 blocks, 10 km buffer).
        fold = np.full(len(d), -1)
        for k, (_, te) in enumerate(spatial_cv.buffered_block_folds(
                d.h3.to_numpy(object), d.lat.to_numpy(), d.lon.to_numpy(), k=5, block_res=2, buffer_km=10.0)):
            fold[te] = k
        S = {}
        for m in ["coarse"] + models:
            s = d[m].to_numpy(np.float64)
            r = np.empty(len(s))
            for k in np.unique(fold):
                j = fold == k
                r[j] = pu.rank_normalise(s[j])
            S[m] = r
        point = {m: capture_auc(S[m], y) - capture_auc(S["coarse"], y) for m in models}
        boot = {m: [] for m in models}
        for _ in range(a.n):
            take = np.concatenate([idx[i] for i in rng.integers(0, len(idx), len(idx))])
            if y[take].sum() < 5:
                continue
            base = capture_auc(S["coarse"][take], y[take])
            for m in models:
                boot[m].append(capture_auc(S[m][take], y[take]) - base)
        for m in models:
            b = np.array(boot[m])
            rows.append({"region": region, "model": m, "n_blocks": len(idx), "n_pos": int(y.sum()),
                         "diff_vs_coarse": round(point[m], 4), "lo90": round(np.quantile(b, 0.05), 4),
                         "hi90": round(np.quantile(b, 0.95), 4), "p_above_zero": round((b > 0).mean(), 3)})
        print(" ".join(f"{r['region']}/{r['model']} {r['diff_vs_coarse']:+.3f} [{r['lo90']:+.3f}, {r['hi90']:+.3f}]"
                       for r in rows if r["region"] == region), flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(a.out, "bootstrap_vs_coarse.csv"), index=False)
    df = pd.DataFrame(rows)
    f = df[df.model == "fine"]
    print(f"fine > coarse: point {int((f.diff_vs_coarse > 0).sum())}/{len(f)}, "
          f"90% interval above zero {int((f.lo90 > 0).sum())}, below zero {int((f.hi90 < 0).sum())}")


if __name__ == "__main__":
    main()
