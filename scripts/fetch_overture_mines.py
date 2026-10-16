"""Extract every mine / quarry polygon on Earth from Overture Maps (OSM-derived land use).

    python scripts/fetch_overture_mines.py --release 2026-09-23.1 --out data/overture_mines.parquet

Overture's `base/land_use` theme carries OpenStreetMap `landuse=quarry` / mining polygons
under subtype `resource_extraction`, with the original OSM tags in `source_tags`. The
`resource=*` tag (gold, copper, iron_ore, coal, ...) names the commodity for a useful
fraction of them. We keep one row per polygon: bbox centre, bbox area, class and the
tags that matter. Only the needed columns are read, row group by row group, straight
from the public S3 bucket (no credentials).

Uses: (1) a global, current mine-footprint mask for leakage control and the
disturbance-masked evaluation; (2) an independent, commodity-tagged label set to check
the model against (OSM mapping is not used by the USGS compilations).
"""

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.fs as pafs
import pyarrow.parquet as pq

BUCKET = "overturemaps-us-west-2"
KEEP_TAGS = ("resource", "landuse", "name", "operator", "industrial", "man_made", "product",
             "disused", "abandoned", "historic", "mineral", "geological", "start_date")


def s3():
    return pafs.S3FileSystem(anonymous=True, region="us-west-2",
                             endpoint_override="https://s3.us-west-2.amazonaws.com",
                             proxy_options=os.environ.get("HTTPS_PROXY") or None)


def one_file(path: str) -> pd.DataFrame:
    fs = s3()
    f = pq.ParquetFile(path, filesystem=fs)
    out = []
    for rg in range(f.metadata.num_row_groups):
        sub = f.read_row_group(rg, columns=["subtype"])["subtype"]
        mask = pc.equal(sub, "resource_extraction")
        if not pc.any(mask).as_py():
            continue
        t = f.read_row_group(rg, columns=["id", "class", "bbox", "source_tags"]).filter(mask)
        bb = t["bbox"].combine_chunks()
        xmin, xmax = bb.field("xmin").to_numpy(), bb.field("xmax").to_numpy()
        ymin, ymax = bb.field("ymin").to_numpy(), bb.field("ymax").to_numpy()
        lat = (ymin + ymax) / 2
        # bbox area in km^2 (an upper bound on the polygon area; good enough for ranking size)
        area = (xmax - xmin) * 111.32 * np.cos(np.radians(lat)) * (ymax - ymin) * 110.57
        df = pd.DataFrame({"id": t["id"].to_numpy(zero_copy_only=False),
                           "class": t["class"].to_numpy(zero_copy_only=False),
                           "lat": lat, "lon": (xmin + xmax) / 2, "bbox_km2": area})
        tags = t["source_tags"].to_pylist()
        for k in KEEP_TAGS:
            df[k] = [dict(x or []).get(k) for x in tags]
        out.append(df)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default="2026-09-23.1")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="data/overture_mines.parquet")
    a = ap.parse_args()
    fs = s3()
    sel = pafs.FileSelector(f"{BUCKET}/release/{a.release}/theme=base/type=land_use/")
    files = sorted(i.path for i in fs.get_file_info(sel) if i.path.endswith(".parquet"))
    print(f"{len(files)} land_use files", flush=True)
    t0 = time.time()
    parts = []
    with ThreadPoolExecutor(a.workers) as ex:
        for i, df in enumerate(ex.map(one_file, files)):
            parts.append(df)
            print(f"  {i + 1}/{len(files)}: {len(df):,} polygons ({time.time() - t0:.0f}s)", flush=True)
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(a.out, index=False)
    print(f"{len(out):,} resource-extraction polygons -> {a.out}")
    print(out["class"].value_counts().head(10).to_string())
    print(out["resource"].value_counts().head(30).to_string())


if __name__ == "__main__":
    main()
