"""Block-bootstrap confidence intervals for hold-out differences between configurations.

Deposits cluster, so cells are resampled in H3 res-2 blocks (~86,000 km^2), not one by
one. For each region and each configuration, reports the capture-AUC difference from a
reference configuration with a 90% interval over block resamples.

    python scripts/bootstrap_transfer.py --out outputs/transfer --ref lean
"""

import argparse
import os
import sys

import h3
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import run_v3 as R  # noqa: E402


def capture_auc(score, y):
    order = np.argsort(-score)
    c = np.cumsum(y[order].astype(np.float64))
    return c.mean() / max(c[-1], 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/transfer")
    ap.add_argument("--features-cache", default="data/global_features.npz")
    ap.add_argument("--data", default="data")
    ap.add_argument("--ref", default="lean")
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    cells = np.load(a.features_cache)["cells"].astype(object)
    lab, _ = R.build_labels(cells, a.data)
    scores = dict(np.load(os.path.join(a.out, "scores.npz")))
    rng = np.random.default_rng(0)
    rows = []
    regions = sorted({k.split("/")[0] for k in scores})
    for region in regions:
        te = np.load(os.path.join(a.out, f"te_{region}.npy"))
        blocks = pd.factorize(np.array([h3.cell_to_parent(c, 2) for c in cells[te]]))[0]
        members = [np.where(blocks == b)[0] for b in range(blocks.max() + 1)]
        draws = [np.concatenate([members[b] for b in rng.integers(0, len(members), len(members))])
                 for _ in range(a.n)]
        for t, y in {"all": lab["pos"][te], "world_class": lab["world_class"][te]}.items():
            ref = scores[f"{region}/{a.ref}"]
            for key in [k for k in scores if k.startswith(region + "/")]:
                name = key.split("/")[1]
                s = scores[key]
                diff = [capture_auc(s[d], y[d]) - capture_auc(ref[d], y[d]) for d in draws if y[d].sum() > 0]
                lo, hi = np.percentile(diff, [5, 95])
                rows.append({"region": region, "target": t, "config": name,
                             "diff_vs_ref": capture_auc(s, y) - capture_auc(ref, y), "lo90": lo, "hi90": hi})
        print(region, flush=True)
    df = pd.DataFrame(rows).round(4)
    df.to_csv(os.path.join(a.out, f"bootstrap_vs_{a.ref}.csv"), index=False)
    print(df[df.config != a.ref].to_string(index=False))


if __name__ == "__main__":
    main()
