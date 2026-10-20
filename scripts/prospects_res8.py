"""Separate prospects from each region's 0.7 km² map.

    python scripts/prospects_res8.py --min-km 10 --out outputs/prospects_res8

The per-region greenfield list keeps one cell per res-7 parent, so its 50 targets sit a
median 1.5–2.7 km apart and often describe a handful of anomalies
(outputs/target_clusters). Here candidates are greenfield cells (no record within ~3 km)
in the region's top 1% by blend; non-maximum suppression keeps the best one and drops
everything within --min-km, then repeats. Each prospect reports its footprint (top-1%
greenfield cells within min_km/2) and refit stability where seed runs exist.
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from earthprosp import prospects  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from build_belt_atlas import BELTS, FRONTIER  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/prospects_res8")
    ap.add_argument("--min-km", type=float, default=10.0)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--top", type=float, default=0.01)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    parts, rows = [], []
    for key, (name, styles) in BELTS.items():
        p = f"outputs/downscale_{key}/map_res8.parquet"
        if not os.path.exists(p):
            continue
        m = pd.read_parquet(p)
        cut = m.blend.quantile(1 - a.top)
        c = m[(m.blend >= cut) & ~m.record_within_3km].reset_index(drop=True)
        k = prospects.nms(c.lat, c.lon, c.blend, a.min_km, a.n)
        fp = prospects.footprint(c.lat, c.lon, k, a.min_km / 2)
        t = c.iloc[k].assign(region=name, region_key=key, frontier=key in FRONTIER, deposit_styles=styles,
                             prospect=range(1, len(k) + 1), footprint_cells=fp)
        parts.append(t)
        rows.append({"region_key": key, "candidates": len(c), "prospects": len(k),
                     "median_footprint": float(pd.Series(fp).median()),
                     "single_cell_share": round(float((fp == 1).mean()), 2)})
        print(f"{key:15s} {len(c):6d} candidates -> {len(k)} prospects, median footprint {rows[-1]['median_footprint']:.0f}",
              flush=True)
    cols = ["region", "region_key", "prospect", "h3", "lat", "lon", "blend", "fine_rank", "district_rank",
            "footprint_cells", "seed_rank_std", "seed_top1pct_share", "frontier", "deposit_styles"]
    out = pd.concat(parts)[cols].round(4)
    out.to_csv(os.path.join(a.out, "prospects_all_regions.csv"), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "summary.csv"), index=False)
    print(f"{len(out)} prospects in {len(rows)} regions")


if __name__ == "__main__":
    main()
