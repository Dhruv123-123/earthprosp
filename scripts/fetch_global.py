"""Global AEF -> H3 res-6 per-cell statistics (resumable).

    python scripts/fetch_global.py --year 2024 --out data/global_2024.npz
    python scripts/fetch_global.py --bbox -121 31 -108 43 --level 3 --res 8 --out data/wus_r8.npz

--level picks the COG overview (6 = 1,280 m pixels, 3 = 160 m) and --res the H3 resolution;
finer grids need finer overviews (~20+ pixels per cell). --bbox limits to tiles that
intersect lon0 lat0 lon1 lat1.

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
    ap.add_argument("--level", type=int, default=6)
    ap.add_argument("--res", type=int, default=6)
    ap.add_argument("--bbox", type=float, nargs=4, default=None, metavar=("LON0", "LAT0", "LON1", "LAT1"))
    a = ap.parse_args()
    os.environ.setdefault("CURL_CA_BUNDLE", "/root/.ccr/ca-bundle.crt")

    tiles = aef.load_index(a.index, a.year)
    if a.bbox:
        x0, y0, x1, y1 = a.bbox
        tiles = tiles[(tiles.wgs84_east > x0) & (tiles.wgs84_west < x1) & (tiles.wgs84_north > y0) & (tiles.wgs84_south < y1)]
    if a.limit:
        tiles = tiles.iloc[:a.limit]
    tag = "" if (a.level, a.res, a.bbox) == (6, 6, None) else "_" + os.path.splitext(os.path.basename(a.out))[0]
    shard_dir = os.path.join(os.path.dirname(a.out) or ".", f"global_shards_{a.year}{tag}" + ("_test" if a.limit else ""))
    print(f"{len(tiles):,} tiles -> {shard_dir}", flush=True)
    t0 = time.time()
    global_agg.run(tiles, shard_dir, level=a.level, res=a.res, n_shards=a.shards, workers=a.workers)
    cells, mean, std, n, failed = global_agg.merge(shard_dir)
    np.savez(a.out, cells=cells.astype(str), mean=mean.astype(np.float16), std=std.astype(np.float16), n=n)
    print(f"done: {len(cells):,} cells, {len(failed)} failed tiles, {time.time() - t0:.0f}s", flush=True)
    if len(failed):
        with open(a.out + ".failed.txt", "w") as f:
            f.write("\n".join(failed))


if __name__ == "__main__":
    main()
