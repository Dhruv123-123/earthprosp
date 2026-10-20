"""Tune for transfer: which model settings rank deposits best on a continent it never saw?

Every configuration is fitted with one region hidden and scored on that region. The
three v3 hold-outs (South America, Australia, Africa) choose the winner; two more regions
that no v3 decision used (Canada north of 50N, Central Asia) are reported alongside as an
untouched check, so a setting that only suits the first three shows up as a drop there.
Held-out scores are saved, so rank-average ensembles are scored without refitting.

    python scripts/transfer_v4.py --out outputs/transfer \
        --drop-features $(cat data/drop_lean.txt) --neutralize-features wc_built,wc_crop
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import run_v3 as R  # noqa: E402
from earthprosp import evaluate, grid, pu  # noqa: E402

CHECK_REGIONS = {"canada": (-141, 50, -52, 75), "central_asia": (46, 36, 120, 56)}
CONFIGS = {  # name: (feature set, LightGBM overrides on top of the fast CV params)
    "lean": ("lean", {}),
    "lean_shallow": ("lean", dict(num_leaves=15, min_child_samples=200)),
    "lean_reg": ("lean", dict(num_leaves=31, min_child_samples=500, reg_lambda=10.0, colsample_bytree=0.3)),
    "lean_extra": ("lean", dict(extra_trees=True)),
    "lean_shallow_extra": ("lean", dict(num_leaves=15, min_child_samples=200, extra_trees=True)),
    "aef": ("aef", {}),
    "aef_shallow": ("aef", dict(num_leaves=15, min_child_samples=200)),
}
ENSEMBLES = {"lean+aef": ("lean", "aef"), "lean_shallow+aef_shallow": ("lean_shallow", "aef_shallow"),
             "lean_extra+aef": ("lean_extra", "aef")}


def rank(x):
    return pu.rank_normalise(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", default="data/global_2024.npz")
    ap.add_argument("--data", default="data")
    ap.add_argument("--features-cache", default="data/global_features.npz")
    ap.add_argument("--out", default="outputs/transfer")
    ap.add_argument("--bags", type=int, default=4)
    ap.add_argument("--drop-features", default="")
    ap.add_argument("--neutralize-features", default="")
    ap.add_argument("--configs", default=",".join(CONFIGS))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    cells, Xa = R.aef_features(a.stats, a.features_cache)
    lat, lon = grid.cell_centers(cells)
    X, cov_cols = R.feature_matrix(cells, Xa, a.data)
    del Xa
    names = np.array(R.aef_names() + cov_cols)
    drop = set(filter(None, a.drop_features.split(",")))
    neutral = set(filter(None, a.neutralize_features.split(",")))
    is_aef = np.char.startswith(names.astype(str), "aef_")
    sets = {"aef": np.where(is_aef)[0], "lean": np.array([i for i, n in enumerate(names) if n not in drop])}
    lab, _ = R.build_labels(cells, a.data)
    pos = lab["pos"]
    w = R.propensity_weights(cells, pos, lab["conf"])
    regions = {**R.HOLDOUTS, **CHECK_REGIONS}
    R.log(f"{len(cells):,} cells; lean {len(sets['lean'])} features; regions {list(regions)}")

    score_path = os.path.join(a.out, "scores.npz")
    scores = dict(np.load(score_path)) if os.path.exists(score_path) else {}
    for region, box in regions.items():
        te = np.where(R.in_box(lat, lon, box))[0]
        tr = np.setdiff1d(np.arange(len(cells)), te)
        np.save(os.path.join(a.out, f"te_{region}.npy"), te)
        for name in a.configs.split(","):
            key = f"{region}/{name}"
            if key in scores:
                continue
            fs, over = CONFIGS[name]
            cols = sets[fs]
            t0 = time.time()
            m = pu.PUBaggingGBM(n_bags=a.bags, seed=0)
            m.params = {**m.params, **R.CV_PARAMS, **over}
            m.fit(X, pos, w, rows=tr, cols=cols)
            nidx = [i for i, c in enumerate(cols) if names[c] in neutral]
            scores[key] = m.predict(R._sub(X, te, cols, nidx)).astype(np.float32)
            np.savez(score_path, **scores)
            auc = evaluate.capture_curve(scores[key], pos[te])["capture_auc"]
            R.log(f"{key}: all={auc:.3f} ({time.time() - t0:.0f}s)")

    rows = []
    for region in regions:
        te = np.load(os.path.join(a.out, f"te_{region}.npy"))
        targets = {"all": (pos[te], None), "world_class": (lab["world_class"][te], None),
                   "contained_cu": (lab["cu_kt"][te] > 0, lab["cu_kt"][te])}
        cand = {n: scores[f"{region}/{n}"] for n in CONFIGS if f"{region}/{n}" in scores}
        for e, (p, q) in ENSEMBLES.items():
            if p in cand and q in cand:
                cand[e] = (rank(cand[p]) + rank(cand[q])) / 2
        for n, s in cand.items():
            for t, (y, wt) in targets.items():
                if y.sum() >= 10:
                    rows.append({"region": region, "config": n, "target": t,
                                 "capture_auc": evaluate.capture_curve(s, y, weight=wt)["capture_auc"]})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "transfer_results.csv"), index=False)
    wide = df.pivot_table(index=["target", "config"], columns="region", values="capture_auc")
    tune = list(R.HOLDOUTS)
    wide["tune_mean"] = wide[tune].mean(1)
    check = [c for c in CHECK_REGIONS if c in wide]
    if check:
        wide["check_mean"] = wide[check].mean(1)
    wide = wide.round(4)
    wide.to_csv(os.path.join(a.out, "transfer_summary.csv"))
    print(wide.to_string())
    json.dump({"regions": {k: list(v) for k, v in regions.items()}, "configs": {k: [v[0], v[1]] for k, v in CONFIGS.items()},
               "features_cache": a.features_cache}, open(os.path.join(a.out, "transfer.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
