"""Backtest: does spreading targets out find more separate deposits?

    python scripts/backtest_prospects.py --out outputs/backtest_prospects

Uses each region's out-of-fold blend scores (downscale.py), ranked within the CV fold.
Picks N targets four ways (top cells, one per res-7 parent as the greenfield lists do, and
non-maximum suppression at 5 and 10 km) and counts what they would have found: targets
with a known deposit within ~2 km (res-8 ring 2), and distinct deposit clusters hit
(deposit cells grouped by res-6 parent, ~36 km²). Also the hit rate of random cells.
"""

import argparse
import glob
import os
import sys

import h3
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from earthprosp import prospects, pu, spatial_cv  # noqa: E402


def within_fold_rank(d, s):
    fold = np.full(len(d), -1)
    for k, (_, te) in enumerate(spatial_cv.buffered_block_folds(
            d.h3.to_numpy(object), d.lat.to_numpy(), d.lon.to_numpy(), k=5, block_res=2, buffer_km=10.0)):
        fold[te] = k
    r = np.empty(len(s))
    for k in np.unique(fold):
        j = fold == k
        r[j] = pu.rank_normalise(s[j])
    return r


def evaluate(cells, pos_set, pos_parent):
    hits, found = 0, set()
    for c in cells:
        near = [n for n in h3.grid_disk(c, 2) if n in pos_set]
        hits += bool(near)
        found.update(pos_parent[n] for n in near)
    return hits, len(found)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/backtest_prospects")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--only", nargs="*", help="regions to run; others are kept from an earlier CSV")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(0)
    csv = os.path.join(a.out, "backtest.csv")
    rows = pd.read_csv(csv).to_dict("records") if a.only and os.path.exists(csv) else []
    done = {r["region"] for r in rows}
    for p in sorted(glob.glob("outputs/downscale_*/downscale_oof_res8.parquet")):
        region = p.split("/")[1].replace("downscale_", "")
        if region in done or (a.only and region not in a.only):
            continue
        d = pd.read_parquet(p).dropna(subset=["blend"]).reset_index(drop=True)
        r = within_fold_rank(d, d.blend.to_numpy(np.float64))
        pos = d.h3[d.positive].tolist()
        pos_set = set(pos)
        pos_parent = {c: h3.cell_to_parent(c, 6) for c in pos}
        top = np.argsort(-r, kind="stable")[:20000]
        picks = {"top_cells": top[:a.n]}
        p7 = pd.Series([h3.cell_to_parent(c, 7) for c in d.h3.to_numpy()[top]])
        picks["one_per_res7"] = top[~p7.duplicated().to_numpy()][:a.n]
        for km in (5, 10):
            k = prospects.nms(d.lat.to_numpy()[top], d.lon.to_numpy()[top], r[top], km, a.n)
            picks[f"nms_{km}km"] = top[k]
        picks["random"] = rng.choice(len(d), a.n, replace=False)
        row = {"region": region, "n_pos": len(pos), "pos_clusters": len(set(pos_parent.values()))}
        for name, idx in picks.items():
            h, f = evaluate(d.h3.to_numpy()[idx], pos_set, pos_parent)
            row[f"{name}_hits"], row[f"{name}_clusters"] = h, f
        rows.append(row)
        print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)
        pd.DataFrame(rows).to_csv(csv, index=False)
    df = pd.DataFrame(rows)
    for m in ("top_cells", "one_per_res7", "nms_5km", "nms_10km", "random"):
        print(f"{m:13s} median hits {df[f'{m}_hits'].median():5.1f}  clusters {df[f'{m}_clusters'].median():5.1f}  "
              f"total clusters {df[f'{m}_clusters'].sum()}")


if __name__ == "__main__":
    main()
