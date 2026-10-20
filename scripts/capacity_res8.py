"""Is the res-8 model the right size? Local model variants on every belt's pool sample.

    python scripts/capacity_res8.py --pool data/pool --out outputs/capacity_res8

Same folds and scoring as pool_belts.py (5 buffered spatial block folds per belt, capture AUC
against the uniform cells of the test folds); only the belt's own training folds are used.
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from earthprosp import pu, spatial_cv  # noqa: E402
from pool_belts import COV_OFF, NEUTRAL, score  # noqa: E402

CONFIGS = {  # name: (bag count, unlabeled ratio, LightGBM params on top of pu's defaults)
    "base": (4, 5.0, dict(n_estimators=300, learning_rate=0.05)),
    "bags16": (16, 5.0, dict(n_estimators=300, learning_rate=0.05)),
    "small_trees": (16, 5.0, dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=20)),
    "ratio20": (4, 20.0, dict(n_estimators=300, learning_rate=0.05)),
    "big": (4, 5.0, dict(n_estimators=800, learning_rate=0.03, num_leaves=127, min_child_samples=20)),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/pool")
    ap.add_argument("--out", default="outputs/capacity_res8")
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--skip", default="", help="comma list of belts to leave out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    prev = os.path.join(a.out, "capacity_folds.csv")
    if os.path.exists(prev):  # resume: keep finished belts
        rows = pd.read_csv(prev).to_dict("records")
    done = {r["belt"] for r in rows}
    for p in sorted(glob.glob(os.path.join(a.pool, "*.npz"))):
        n = os.path.basename(p)[:-4]
        if n in a.skip.split(",") or n in done:
            continue
        b = dict(np.load(p))
        X, pos, w, uni = b["X"], b["pos"], b["w"].astype(np.float32), b["uniform"]
        nz = [COV_OFF + list(b["cov_cols"]).index(c) for c in NEUTRAL if c in list(b["cov_cols"])]
        trainable = pos | uni
        folds = spatial_cv.buffered_block_folds(b["cells"].astype(object), b["lat"], b["lon"], k=5,
                                                block_res=2, buffer_km=10.0)
        t0 = time.time()
        for k, (tr, te) in enumerate(folds):
            if pos[te].sum() < 5 or uni[te].sum() < 100:
                continue
            Xte = np.asarray(X[te], np.float32)
            Xte[:, nz] = 0.0
            for c in a.configs.split(","):
                bags, ratio, params = CONFIGS[c]
                m = pu.PUBaggingGBM(n_bags=bags, unlabeled_ratio=ratio, seed=k)
                m.params = {**m.params, **params}
                m.fit(X, pos, w, rows=tr[trainable[tr]])
                rows.append({"belt": n, "fold": k, "config": c, "n_pos_train": int(pos[tr].sum()),
                             **score(m.predict(Xte), pos[te], uni[te])})
        log_rows = [r for r in rows if r["belt"] == n]
        print(f"[{time.strftime('%H:%M:%S')}] {n} ({time.time() - t0:.0f}s): " + " ".join(
            f"{c}={np.mean([r['capture_auc'] for r in log_rows if r['config'] == c]):.3f}"
            for c in a.configs.split(",")), flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(a.out, "capacity_folds.csv"), index=False)
    df = pd.DataFrame(rows)
    per = df.groupby(["belt", "config"])[["capture_auc", "top5", "top1"]].mean().round(4)
    per.to_csv(os.path.join(a.out, "capacity_by_belt.csv"))
    piv = per.capture_auc.unstack("config")
    print(piv.to_string())
    print(per.groupby("config").median().round(4).to_string())
    print("belts beating base:", {c: int((piv[c] > piv["base"]).sum()) for c in piv.columns if c != "base"})


if __name__ == "__main__":
    main()
