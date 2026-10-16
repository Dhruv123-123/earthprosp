"""Which public datasets earn their place? Feature-family ablation under spatial CV.

For each covariate family F (gravity, litho, crust, faults, plates, volcano, landcover,
terrain, s2_alteration, ...) fit the v3 PU-bagged GBM on
  * AEF + F           (what F adds to the satellite embedding on its own), and
  * everything but F  (what is lost without it, given all the others),
and report the change in spatial-CV capture AUC against AEF-only and against all
features. Uses the same buffered block folds, labels and weights as run_v3.

    python scripts/ablate_v3.py --out outputs/v3 --folds-used 3
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import run_v3 as R  # noqa: E402
from earthprosp import evaluate, grid, pu, spatial_cv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", default="data/global_2024.npz")
    ap.add_argument("--data", default="data")
    ap.add_argument("--features-cache", default="data/global_features.npz")
    ap.add_argument("--out", default="outputs/v3")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--folds-used", type=int, default=3)
    ap.add_argument("--bags", type=int, default=2)
    ap.add_argument("--buffer-km", type=float, default=50.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    cells, Xa = R.aef_features(a.stats, a.features_cache)
    lat, lon = grid.cell_centers(cells)
    cov = R.covariate_table(cells, a.data)
    X = R.stack_features(Xa, cov)
    del Xa
    names = np.array(R.aef_names() + list(cov.columns))
    fam = np.array([R._family(n) for n in names])
    is_aef = np.char.startswith(names.astype(str), "aef_")
    lab, _ = R.build_labels(cells, a.data)
    pos = lab["pos"]
    us = R.in_box(lat, lon, R.US_BOX)
    w = R.propensity_weights(cells, pos, lab["conf"])

    geo_fams = sorted(set(fam[~is_aef]))
    configs = {"aef": np.where(is_aef)[0], "all": np.arange(len(names))}
    for f in geo_fams:
        configs[f"aef+{f}"] = np.where(is_aef | (fam == f))[0]
        configs[f"all-{f}"] = np.where(fam != f)[0]
    R.log(f"{len(configs)} configurations over {len(geo_fams)} families: {geo_fams}")

    folds = spatial_cv.buffered_block_folds(cells, lat, lon, k=a.folds, block_res=1,
                                            buffer_km=a.buffer_km)[: a.folds_used]
    path = os.path.join(a.out, "ablation_folds.csv")
    rows = pd.read_csv(path).to_dict("records") if os.path.exists(path) else []
    done = {(r["config"], r["fold"]) for r in rows}
    for k, (tr, te) in enumerate(folds):
        targets = {"all": pos[te], "non_us": pos[te] & ~us[te], "world_class": lab["world_class"][te]}
        for name, cols in configs.items():
            if (name, k) in done:
                continue
            t0 = time.time()
            m = R.fit_gbm(X, cols, tr, pos, w, None, a.bags, fast=True)
            sc = m.predict(R._sub(X, te, cols))
            row = {"config": name, "fold": k, "n_features": len(cols)}
            for t, y in targets.items():
                row[t] = evaluate.capture_curve(sc, y)["capture_auc"] if y.sum() >= 10 else np.nan
            rows.append(row)
            pd.DataFrame(rows).to_csv(path, index=False)
            R.log(f"fold {k} {name}: all={row['all']:.3f} non_us={row['non_us']:.3f} ({time.time() - t0:.0f}s)")

    res = pd.DataFrame(rows).groupby("config")[["all", "non_us", "world_class"]].mean()
    base, full = res.loc["aef"], res.loc["all"]
    out = []
    for f in geo_fams:
        out.append({"family": f,
                    **{f"gain_over_aef_{t}": res.loc[f"aef+{f}", t] - base[t] for t in res.columns},
                    **{f"loss_without_{t}": full[t] - res.loc[f"all-{f}", t] for t in res.columns}})
    out = pd.DataFrame(out).sort_values("loss_without_all", ascending=False).round(4)
    out.to_csv(os.path.join(a.out, "ablation.csv"), index=False)
    print(res.round(3).to_string())
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
