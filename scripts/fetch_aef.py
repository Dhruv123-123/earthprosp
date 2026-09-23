"""Fetch AEF overview embeddings for a bbox (or the globe) and save per-cell bags.

    python scripts/fetch_aef.py --bbox -120.5 35 -113.5 42.5 --year 2024 --out data/greatbasin_2024.npz
    python scripts/fetch_aef.py --global --year 2024 --out data/global_2024.npz   # ~34k tiles

The index parquet (~800 MB as CSV, smaller as parquet) is cached under data/.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from earthprosp import aef  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"))
    ap.add_argument("--global", dest="glob", action="store_true")
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--overview-level", type=int, default=6, help="6 -> 1.28 km pixels")
    ap.add_argument("--res", type=int, default=6, help="H3 resolution")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--index", default="data/aef_index.parquet")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    if not os.path.exists(a.index):
        print(f"downloading index -> {a.index}")
        os.makedirs(os.path.dirname(a.index), exist_ok=True)
        aef.load_index(aef.INDEX_URL).to_parquet(a.index)
    idx = aef.load_index(a.index, a.year)
    tiles = idx if a.glob else aef.tiles_in_bbox(idx, *a.bbox)
    print(f"{len(tiles)} tiles for year {a.year}")
    t0 = time.time()
    px = aef.read_tiles(tiles, a.overview_level, a.workers)
    print(f"read {len(px.lat):,} pixels in {time.time() - t0:.0f}s")
    if a.bbox and not a.glob:
        w, s, e, n = a.bbox
        keep = (px.lon >= w) & (px.lon <= e) & (px.lat >= s) & (px.lat <= n)
        px = aef.TilePixels(px.lat[keep], px.lon[keep], px.emb[keep])
    cb = aef.pixels_to_bags(px, res=a.res)
    cb.save(a.out)
    print(f"saved {len(cb.cells):,} cells -> {a.out}")


if __name__ == "__main__":
    main()
