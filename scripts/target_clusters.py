"""How spread out are each region's 50 greenfield targets?

    python scripts/target_clusters.py --out outputs/target_clusters

Targets are already one per res-7 parent, but nothing stops them bunching in one corner of
a belt. For each region: median distance to the nearest other target, the number of
clusters when targets within 10 km are linked, the share in the largest cluster, and the
share inside the busiest 1° box. Adds a cluster id to each target.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from earthprosp.prospects import haversine_km  # noqa: E402


def link_clusters(D, km):
    """Single-linkage components of the graph with edges where D <= km."""
    n = len(D)
    lab = np.full(n, -1)
    c = 0
    for i in range(n):
        if lab[i] >= 0:
            continue
        stack = [i]
        lab[i] = c
        while stack:
            j = stack.pop()
            for k in np.where((D[j] <= km) & (lab < 0))[0]:
                lab[k] = c
                stack.append(k)
        c += 1
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="outputs/atlas/greenfield_targets_all_regions.csv")
    ap.add_argument("--out", default="outputs/target_clusters")
    ap.add_argument("--link-km", type=float, default=10.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t = pd.read_csv(a.targets)
    rows, parts = [], []
    for key, g in t.groupby("region_key", sort=False):
        lat, lon = g.lat.to_numpy(), g.lon.to_numpy()
        D = haversine_km(lat[:, None], lon[:, None], lat[None], lon[None])
        np.fill_diagonal(D, np.inf)
        lab = link_clusters(D, a.link_km)
        sizes = np.bincount(lab)
        box = pd.Series(list(zip(np.floor(lat), np.floor(lon)))).value_counts()
        rows.append({"region_key": key, "region": g.region.iloc[0], "n": len(g),
                     "nn_km_median": round(float(np.median(D.min(1))), 1),
                     "clusters": int(len(sizes)), "largest_cluster": int(sizes.max()),
                     "largest_share": round(sizes.max() / len(g), 2),
                     "busiest_1deg_share": round(box.iloc[0] / len(g), 2),
                     "boxes_1deg": int(len(box))})
        parts.append(g.assign(cluster=lab, cluster_size=sizes[lab]))
    s = pd.DataFrame(rows).sort_values("largest_share", ascending=False)
    s.to_csv(os.path.join(a.out, "clustering_by_region.csv"), index=False)
    pd.concat(parts).to_csv(os.path.join(a.out, "targets_with_clusters.csv"), index=False)
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
