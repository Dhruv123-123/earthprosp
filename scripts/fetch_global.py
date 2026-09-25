"""Global AEF -> H3 res-6 per-cell statistics (resumable).

    python scripts/fetch_global.py --year 2024 --out data/global_2024.npz

Writes data/global_shards_<year>/shard_*.npz as it goes; re-running skips finished shards.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402

from earthprosp import aef, global_agg  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--index", default="data/aef_index.parquet")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--shards", type=int, default=60)
    ap.add_argument("--limit", type=int, default=None, help="only the first N tiles (testing)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.environ.setdefault("CURL_CA_BUNDLE", "/root/.ccr/ca-bundle.crt")

    tiles = aef.load_index(a.index, a.year)
    if a.limit:
        tiles = tiles.iloc[:a.limit]
    shard_dir = os.path.join(os.path.dirname(a.out) or ".", f"global_shards_{a.year}" + ("_test" if a.limit else ""))
    print(f"{len(tiles):,} tiles -> {shard_dir}", flush=True)
    t0 = time.time()
    global_agg.run(tiles, shard_dir, n_shards=a.shards, workers=a.workers)
    cells, mean, std, n, failed = global_agg.merge(shard_dir)
    np.savez(a.out, cells=cells.astype(str), mean=mean.astype(np.float16), std=std.astype(np.float16), n=n)
    print(f"done: {len(cells):,} cells, {len(failed)} failed tiles, {time.time() - t0:.0f}s", flush=True)
    if len(failed):
        with open(a.out + ".failed.txt", "w") as f:
            f.write("\n".join(failed))


if __name__ == "__main__":
    main()
