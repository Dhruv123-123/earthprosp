"""Stream a global COG mosaic onto H3 res-6 cells (resumable shards); or, with --res 8
--bbox --level, onto res-8 cells in one region.

    python scripts/fetch_rasters.py worldcover --out data/worldcover_res6.parquet
    python scripts/fetch_rasters.py dem        --out data/dem_res6.parquet
    python scripts/fetch_rasters.py s2         --out data/s2_res6.parquet

See earthprosp/cog_agg.py for what each product computes.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from earthprosp import cog_agg  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("product", choices=list(cog_agg.PRODUCTS))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shards", type=int, default=40)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=int, default=6)
    ap.add_argument("--level", type=int, default=None, help="overview level (default: the product's)")
    ap.add_argument("--bbox", type=float, nargs=4, default=None, metavar=("LON0", "LAT0", "LON1", "LAT1"),
                    help="only tiles overlapping this box (for a regional res-8 run)")
    a = ap.parse_args()
    os.environ.setdefault("CURL_CA_BUNDLE", "/root/.ccr/ca-bundle.crt")
    t0 = time.time()
    urls = cog_agg.list_tiles(a.product, a.limit)
    if a.bbox:
        urls = [u for u in urls if cog_agg.tile_overlaps(u, a.bbox)]
    tag = "" if a.res == 6 and not a.bbox else "_" + os.path.basename(a.out).split(".")[0]
    shard_dir = os.path.join(os.path.dirname(a.out) or ".", f"{a.product}_shards{tag}" + ("_test" if a.limit else ""))
    print(f"{len(urls):,} tiles -> {shard_dir}", flush=True)
    cog_agg.run(a.product, urls, shard_dir, n_shards=min(a.shards, len(urls)), workers=a.workers,
                res=a.res, level=a.level)
    df = cog_agg.merge(a.product, shard_dir)
    df.to_parquet(a.out)
    print(f"done: {len(df):,} cells, {df.shape[1]} columns, {time.time() - t0:.0f}s", flush=True)
    print(df.describe().T.round(3).to_string())


if __name__ == "__main__":
    main()
