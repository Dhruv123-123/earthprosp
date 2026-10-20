"""One small table of the best 0.7 km² cells in every mapped region, for committing and lookups.

    python scripts/export_res8_top.py --frac 0.02 --out outputs/atlas/res8_top.parquet

Reads outputs/downscale_<region>/map_res8.parquet (git-ignored, ~50 MB each) and keeps each
region's top `frac` of cells by blend score.
"""
import argparse
import glob
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=0.02)
    ap.add_argument("--out", default="outputs/atlas/res8_top.parquet")
    a = ap.parse_args()
    keep = ["h3", "lat", "lon", "blend", "fine_rank", "district_rank", "known_positive", "any_record",
            "record_within_3km"]
    parts = []
    for p in sorted(glob.glob("outputs/downscale_*/map_res8.parquet")):
        region = p.split("/")[1].replace("downscale_", "")
        d = pd.read_parquet(p)
        d = d[d.blend >= d.blend.quantile(1 - a.frac)][keep].assign(region=region)
        parts.append(d)
    out = pd.concat(parts, ignore_index=True)
    for c in ("lat", "lon", "blend", "fine_rank", "district_rank"):
        out[c] = out[c].astype("float32")
    out.to_parquet(a.out, index=False, compression="zstd")
    print(f"{len(out):,} cells from {len(parts)} regions -> {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
