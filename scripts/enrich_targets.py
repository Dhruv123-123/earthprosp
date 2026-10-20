"""One greenfield target table for every mapped region, with the likely mineral system.

    python scripts/enrich_targets.py

The 0.7 km² maps rank where to look; the global v3 map's heads say for what. Each target gets
its res-6 parent's most likely mineral system (P(system | deposit)) and top two commodities.
Reads outputs/downscale_<region>/greenfield_targets_res8.csv; writes
outputs/atlas/greenfield_targets_all_regions.csv.
"""
import json
import os
import sys

import h3
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from build_belt_atlas import BELTS, FRONTIER  # noqa: E402

T = "outputs/atlas/greenfield_targets_all_regions.csv"
G = "outputs/v3L/global_v3_res6.parquet"


def add_systems(t, g=None):
    """Adds likely_system, system_share, likely_commodities and global_percentile_36km2."""
    g = pd.read_parquet(G).set_index("h3") if g is None else g
    sys_cols = [c for c in g.columns if c.startswith("sys_")]
    com_cols = [c for c in g.columns if c.startswith("p_") and c not in ("p_prospective", "p_district")]
    par = g.reindex([h3.cell_to_parent(c, 6) for c in t.h3])
    s, c = par[sys_cols].to_numpy(), par[com_cols].to_numpy()
    t["likely_system"] = [sys_cols[i][4:] for i in s.argmax(1)]
    t["system_share"] = s.max(1).round(3)
    order = (-c).argsort(1)
    t["likely_commodities"] = [f"{com_cols[a][2:]}, {com_cols[b][2:]}" for a, b in order[:, :2]]
    t["global_percentile_36km2"] = par.score_rank.to_numpy().round(4)
    return t


def main():
    parts = []
    for k, (name, deps) in BELTS.items():
        d = f"outputs/downscale_{k}"
        if not os.path.exists(f"{d}/greenfield_targets_res8.csv"):
            continue
        x = pd.read_csv(f"{d}/greenfield_targets_res8.csv")
        x.insert(0, "rank_in_region", range(1, len(x) + 1))
        x.insert(0, "region", name)
        x = x.assign(region_key=k, deposit_styles=deps, frontier=k in FRONTIER,
                     map_model=json.load(open(f"{d}/downscale.json")).get("map_model", "fine"))
        parts.append(x)
    t = pd.concat(parts, ignore_index=True)
    t = add_systems(t)
    t.to_csv(T, index=False)
    print(t.groupby("region").likely_system.agg(lambda x: x.value_counts().index[0]).to_string())


if __name__ == "__main__":
    main()
