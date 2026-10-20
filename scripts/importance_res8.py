"""What does the 0.7 km² model look at? Gain importance by feature family, per region.

    python scripts/importance_res8.py --pool data/pool --out outputs/importance_res8

Fits the small-tree local model on each region's pool sample (all of it, no CV) and sums
LightGBM gain over the feature families of downscale.py's res-8 matrix.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from earthprosp import pu  # noqa: E402

FAMILIES = [("cell AEF mean", 0, 64), ("cell AEF std", 64, 128), ("ring ~2 km", 128, 192),
            ("ring ~5 km", 192, 256), ("parent 36 km² AEF mean", 256, 320), ("parent 36 km² AEF std", 320, 384)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/pool")
    ap.add_argument("--out", default="outputs/importance_res8")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for p in sorted(glob.glob(os.path.join(a.pool, "*.npz"))):
        region = os.path.basename(p)[:-4]
        b = np.load(p)
        X, pos, uni = b["X"], b["pos"], b["uniform"]
        cov = list(b["cov_cols"])
        m = pu.PUBaggingGBM(n_bags=8, seed=0)
        m.params = {**m.params, "n_estimators": 300, "num_leaves": 15, "min_child_samples": 20}
        m.fit(X, pos, b["w"].astype(np.float32), rows=np.where(pos | uni)[0])
        g = m.feature_importance("gain")
        g = g / g.sum()
        fam = {name: g[i:j].sum() for name, i, j in FAMILIES}
        fam["covariates (36 km²)"] = g[384:384 + len(cov)].sum()
        top_cov = sorted(zip(cov, g[384:384 + len(cov)]), key=lambda t: -t[1])[:3]
        rows.append({"region": region, **{k: round(float(v), 3) for k, v in fam.items()},
                     "top_covariates": ", ".join(f"{c} {v:.3f}" for c, v in top_cov)})
        print(region, {k: round(float(v), 2) for k, v in fam.items()}, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "importance_by_family.csv"), index=False)
    print(df.drop(columns="top_covariates").set_index("region").median().round(3).to_string())


if __name__ == "__main__":
    main()
