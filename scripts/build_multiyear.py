"""Average AlphaEarth cell statistics over several years and rebuild the feature cache.

    python scripts/build_multiyear.py --stats data/global_2024.npz data/global_2018.npz \
        --out data/global_features_2y.npz

A single year's embedding carries that year's clouds, snow, crops and fires. Bedrock does
not change between 2018 and 2024, so averaging the years keeps the geology and damps the
rest. Cells are those of the first file; a cell missing from a later year keeps the years
it has. Within-cell std is averaged the same way. Context (ring) features are recomputed
from the averaged means. The result is a drop-in --features-cache for run_v3.py.
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from earthprosp import features  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    z = np.load(a.stats[0])
    cells = z["cells"].astype(object)
    mean = z["mean"].astype(np.float32)
    std = z["std"].astype(np.float32)
    count = np.ones(len(cells), np.float32)
    pos = pd.Index(cells)
    for path in a.stats[1:]:
        y = np.load(path)
        j = pos.get_indexer(y["cells"].astype(object))
        ok = j >= 0
        mean[j[ok]] += y["mean"][ok].astype(np.float32)
        std[j[ok]] += y["std"][ok].astype(np.float32)
        count[j[ok]] += 1
        print(f"{path}: {ok.sum():,} of {len(cells):,} cells matched ({time.time() - t0:.0f}s)", flush=True)
        del y
    mean /= count[:, None]
    std /= count[:, None]
    print(f"years per cell: {pd.Series(count).value_counts().to_dict()}", flush=True)
    ctx = features.neighbourhood_context(cells, mean)
    X = np.concatenate([mean, std, ctx], 1).astype(np.float16)
    np.savez(a.out, cells=cells.astype(str), X=X)
    print(f"wrote {a.out} {X.shape} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
