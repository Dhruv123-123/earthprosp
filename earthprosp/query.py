"""Look up the v3 map at a point, or rank an area.

    python -m earthprosp.query point -22.3 -68.9
    python -m earthprosp.query area --bbox -70 -25 -67 -20 --top 10 --system porphyry_skarn_cu
    python -m earthprosp.query point -22.3 -68.9 --res8   # also the 0.7 km² cell, inside a mapped belt

Reads the res-6 table (`global_v3_res6.parquet`, written by run_v3.py) when it exists,
otherwise the committed res-4 aggregate (`global_v3_res4.parquet`, ~36x coarser).
"""

from __future__ import annotations

import argparse
import os

import h3
import numpy as np
import pandas as pd

DEFAULT_DIR = "outputs/v3L"
RES8_GLOB = "outputs/downscale_*/map_res8.parquet"  # full belt maps (git-ignored)
RES8_TOP = "outputs/atlas/res8_top.parquet"        # committed: each belt's top 2% of cells


def load(out_dir: str = DEFAULT_DIR) -> tuple[pd.DataFrame, int]:
    """(table indexed by H3 cell, resolution)."""
    p6 = os.path.join(out_dir, "global_v3_res6.parquet")
    if os.path.exists(p6):
        return pd.read_parquet(p6).set_index("h3"), 6
    return pd.read_parquet(os.path.join(out_dir, "global_v3_res4.parquet")).set_index("h4"), 4


def point(df: pd.DataFrame, res: int, lat: float, lon: float) -> dict:
    """Scores of the cell containing (lat, lon); commodity and system shares sorted."""
    cell = h3.latlng_to_cell(lat, lon, res)
    if cell not in df.index:
        return {"cell": cell, "found": False}
    r = df.loc[cell]
    out = {"cell": cell, "found": True, "resolution": res,
           "p_prospective": round(float(r.p_prospective), 4), "percentile": round(float(r.score_rank), 4),
           "uncertainty": round(float(r.uncertainty), 4), "in_area_of_applicability": bool(r.in_aoa >= 0.5)}
    comm = {c[2:]: float(r[c]) for c in df.columns if c.startswith("p_") and c not in ("p_prospective", "p_district")}
    syst = {c[4:]: float(r[c]) for c in df.columns if c.startswith("sys_")}
    out["commodities"] = dict(sorted(((k, round(v, 3)) for k, v in comm.items()), key=lambda kv: -kv[1])[:5])
    out["mineral_systems"] = dict(sorted(((k, round(v, 3)) for k, v in syst.items()), key=lambda kv: -kv[1])[:3])
    heads = {c[5:]: round(float(r[c]), 3) for c in df.columns if c.startswith("head_")}
    if heads:
        out["system_head_percentiles"] = heads
    return out


def point_res8(lat: float, lon: float, top_path: str = RES8_TOP, glob_pat: str = RES8_GLOB) -> dict:
    """The 0.7 km² (H3 res 8) cell at (lat, lon) in the belt maps: its blend score (a rank in
    its belt, 1 = top) when a full map covers it, else whether it is in a belt's top 2%."""
    import glob
    cell = h3.latlng_to_cell(lat, lon, 8)
    for p in sorted(glob.glob(glob_pat)):
        d = pd.read_parquet(p, filters=[("h3", "==", cell)])
        if len(d):
            r = d.iloc[0]
            out = {"cell": cell, "region": p.split(os.sep)[-2].replace("downscale_", ""),
                   "blend_rank": round(float(r.blend), 4), "fine_rank": round(float(r.fine_rank), 4),
                   "district_rank": round(float(r.district_rank), 4), "known_deposit": bool(r.known_positive),
                   "record_within_3km": bool(r.record_within_3km)}
            if "seed_top1pct_share" in d:
                out["top1pct_in_refits"] = round(float(r.seed_top1pct_share), 2)
            return out
    if os.path.exists(top_path):
        d = pd.read_parquet(top_path, filters=[("h3", "==", cell)])
        if len(d):
            return {"cell": cell, "region": d.region.iloc[0], "in_top_2pct": True,
                    "blend_rank": round(float(d.blend.iloc[0]), 4)}
    return {"cell": cell, "in_top_2pct": False}


def area(df: pd.DataFrame, res: int, bbox: tuple[float, float, float, float], top: int = 10,
         system: str | None = None, greenfield: bool = False) -> pd.DataFrame:
    """Top cells inside bbox = (lon_min, lat_min, lon_max, lat_max), by presence or by
    P(deposit) x P(system | deposit)."""
    if "lat" in df.columns:
        lat, lon = df.lat.to_numpy(), df.lon.to_numpy()
    else:
        ll = np.array([h3.cell_to_latlng(c) for c in df.index])
        lat, lon = ll[:, 0], ll[:, 1]
    x0, y0, x1, y1 = bbox
    m = (lon >= x0) & (lon <= x1) & (lat >= y0) & (lat <= y1)
    d = df[m].assign(lat=lat[m], lon=lon[m])
    if greenfield and "any_record" in d:
        d = d[~d.any_record]
    key = d.p_prospective * d[f"sys_{system}"] if system else d.p_prospective
    cols = ["lat", "lon", "p_prospective", "score_rank", "uncertainty"] + ([f"sys_{system}"] if system else [])
    return d.assign(score=key).nlargest(top, "score")[cols + ["score"]].round(4)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DEFAULT_DIR)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("point")
    p.add_argument("lat", type=float)
    p.add_argument("lon", type=float)
    p.add_argument("--res8", action="store_true", help="also look up the 0.7 km² belt maps")
    q = sub.add_parser("area")
    q.add_argument("--bbox", type=float, nargs=4, required=True, metavar=("LON0", "LAT0", "LON1", "LAT1"))
    q.add_argument("--top", type=int, default=10)
    q.add_argument("--system", default=None)
    q.add_argument("--greenfield", action="store_true", help="only cells with no mineral record")
    a = ap.parse_args()
    df, res = load(a.out)
    if a.cmd == "point":
        import json
        out = point(df, res, a.lat, a.lon)
        if a.res8:
            out["res8"] = point_res8(a.lat, a.lon)
        print(json.dumps(out, indent=2))
    else:
        print(area(df, res, tuple(a.bbox), a.top, a.system, a.greenfield).to_string())


if __name__ == "__main__":
    main()
